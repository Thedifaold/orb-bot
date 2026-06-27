"""
ORB Trading Bot — Big Daddy Max Strategy
Detecta setups de Opening Range Breakout en tiempo real
Usa Claude como motor de análisis + Telegram para notificaciones
"""

import os
import asyncio
import logging
from datetime import datetime, time
import pytz
import anthropic
import requests
import yfinance as yf
import pandas as pd

# ── Configuración ─────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY")

# Par(es) a monitorear — cambia según tu preferencia
SYMBOLS = ["EURUSD=X", "GC=F", "SI=F"]   # Forex & Metales

# Sesiones (hora UTC)
SESSIONS = {
    "Asia":     {"open": time(0,  0), "close": time(8,  0)},
    "Londres":  {"open": time(7,  0), "close": time(16, 0)},
    "NewYork":  {"open": time(13, 0), "close": time(21, 0)},
}

SCAN_INTERVAL_SECONDS = 60   # Escanear cada 60 seg
RR_RATIO = 2.0               # Risk/Reward mínimo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ── Sesión activa ─────────────────────────────────────────────────────────────
def get_active_session() -> str | None:
    now_utc = datetime.utcnow().time()
    for name, hours in SESSIONS.items():
        if hours["open"] <= now_utc < hours["close"]:
            return name
    return None


# ── Datos de mercado ──────────────────────────────────────────────────────────
def fetch_candles(symbol: str, interval: str = "15m", periods: int = 20) -> pd.DataFrame:
    """Descarga las últimas N velas del intervalo solicitado."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval=interval)
        if df.empty:
            log.warning(f"Sin datos para {symbol} en {interval}")
            return pd.DataFrame()
        return df.tail(periods)
    except Exception as e:
        log.error(f"Error al obtener datos de {symbol}: {e}")
        return pd.DataFrame()


def build_candle_summary(df: pd.DataFrame) -> str:
    """Convierte el DataFrame en texto legible para Claude."""
    lines = []
    for ts, row in df.iterrows():
        lines.append(
            f"{ts.strftime('%H:%M')} | O:{row['Open']:.5f} H:{row['High']:.5f} "
            f"L:{row['Low']:.5f} C:{row['Close']:.5f}"
        )
    return "\n".join(lines)


# ── Prompt ORB para Claude ────────────────────────────────────────────────────
ORB_SYSTEM_PROMPT = """
Eres un trader profesional especializado en la estrategia ORB (Opening Range Breakout)
desarrollada por Big Daddy Max. Tu único trabajo es analizar datos de velas OHLC
y detectar setups válidos siguiendo estas reglas al pie de la letra:

REGLAS DE LA ESTRATEGIA ORB:
1. RANGO DE APERTURA: La primera vela de 15M de la sesión marca el ORB.
   - ORB High = máximo de esa vela
   - ORB Low  = mínimo de esa vela

2. BREAKOUT (en 5M): Una vela de 5M debe CERRAR completamente fuera del ORB.
   - Cierre por encima de ORB High → posible LONG
   - Cierre por debajo de ORB Low  → posible SHORT

3. RETEST: El precio debe regresar a la zona rota y mostrar rechazo.
   - En LONG: precio baja de nuevo al ORB High y rebota al alza
   - En SHORT: precio sube de nuevo al ORB Low y cae

4. ENTRADA: Justo al otro lado de la zona rota tras el retest.

5. GESTIÓN:
   - SL: al otro lado de la zona ORB
   - TP: relación mínima 1:2 riesgo/beneficio

RESPONDE SIEMPRE EN ESTE FORMATO JSON EXACTO (sin markdown):
{
  "setup_valido": true|false,
  "direccion": "LONG"|"SHORT"|"NINGUNA",
  "orb_high": <número>,
  "orb_low": <número>,
  "breakout_confirmado": true|false,
  "retest_completado": true|false,
  "entrada": <número>|null,
  "stop_loss": <número>|null,
  "take_profit": <número>|null,
  "rr_ratio": <número>|null,
  "razon": "<explicación breve en español>"
}
"""

def analyze_orb_with_claude(
    symbol: str,
    session: str,
    candles_15m: str,
    candles_5m: str,
    orb_high: float,
    orb_low: float
) -> dict:
    """Envía datos a Claude y recibe análisis ORB estructurado."""
    user_message = f"""
Símbolo: {symbol}
Sesión activa: {session}
Hora UTC: {datetime.utcnow().strftime('%H:%M')}

ORB DEFINIDO:
- ORB High: {orb_high:.5f}
- ORB Low:  {orb_low:.5f}

VELAS 15M (últimas 10):
{candles_15m}

VELAS 5M (últimas 20):
{candles_5m}

Analiza si hay un setup ORB válido y responde en JSON.
"""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=ORB_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        raw = response.content[0].text.strip()
        import json
        return json.loads(raw)
    except Exception as e:
        log.error(f"Error en análisis Claude: {e}")
        return {"setup_valido": False, "razon": str(e)}


# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(message: str):
    """Envía mensaje de texto a Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram no configurado — solo log local.")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        log.error(f"Error enviando Telegram: {e}")


def format_signal(symbol: str, session: str, analysis: dict) -> str:
    """Formatea la alerta de Telegram."""
    direction_emoji = "🟢" if analysis["direccion"] == "LONG" else "🔴"
    return f"""
{direction_emoji} <b>SEÑAL ORB DETECTADA</b>

📊 Par: <b>{symbol}</b>
🌍 Sesión: <b>{session}</b>
⏰ Hora UTC: {datetime.utcnow().strftime('%H:%M')}

━━━━━━━━━━━━━━━━━━
🎯 Dirección: <b>{analysis['direccion']}</b>
📍 Entrada:    <code>{analysis['entrada']}</code>
🛑 Stop Loss:  <code>{analysis['stop_loss']}</code>
✅ Take Profit: <code>{analysis['take_profit']}</code>
📐 R/R:        <b>1:{analysis['rr_ratio']}</b>

🔲 ORB High: <code>{analysis['orb_high']}</code>
🔲 ORB Low:  <code>{analysis['orb_low']}</code>
━━━━━━━━━━━━━━━━━━
💬 {analysis['razon']}

⚠️ <i>Solo informativo. Gestiona tu riesgo.</i>
"""


# ── Loop principal ────────────────────────────────────────────────────────────
def compute_orb(df_15m: pd.DataFrame) -> tuple[float, float]:
    """Extrae ORB High y ORB Low de la primera vela de 15M del día."""
    if df_15m.empty:
        return 0.0, 0.0
    first = df_15m.iloc[0]
    return float(first["High"]), float(first["Low"])


async def scan_symbol(symbol: str, session: str):
    """Escanea un símbolo buscando setup ORB."""
    log.info(f"Escaneando {symbol} — Sesión: {session}")

    df_15m = fetch_candles(symbol, interval="15m", periods=10)
    df_5m  = fetch_candles(symbol, interval="5m",  periods=20)

    if df_15m.empty or df_5m.empty:
        log.warning(f"Datos insuficientes para {symbol}")
        return

    orb_high, orb_low = compute_orb(df_15m)

    if orb_high == 0.0:
        log.warning(f"No se pudo calcular ORB para {symbol}")
        return

    candles_15m = build_candle_summary(df_15m)
    candles_5m  = build_candle_summary(df_5m)

    analysis = analyze_orb_with_claude(
        symbol, session, candles_15m, candles_5m, orb_high, orb_low
    )

    log.info(f"{symbol} → setup_valido={analysis.get('setup_valido')} | {analysis.get('razon','')}")

    if analysis.get("setup_valido") and analysis.get("rr_ratio", 0) >= RR_RATIO:
        msg = format_signal(symbol, session, analysis)
        send_telegram(msg)
        log.info(f"✅ Señal enviada para {symbol}")
    else:
        log.info(f"⏳ Sin setup válido para {symbol}")


async def main_loop():
    log.info("🤖 ORB Bot iniciado — Big Daddy Max Strategy")
    send_telegram("🤖 <b>ORB Bot activado</b>\nMonitoreando: " + ", ".join(SYMBOLS))

    while True:
        session = get_active_session()
        if session:
            tasks = [scan_symbol(sym, session) for sym in SYMBOLS]
            await asyncio.gather(*tasks)
        else:
            log.info("⏸ Fuera de sesión activa — esperando...")

        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main_loop())
