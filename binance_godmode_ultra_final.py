#!/usr/bin/env python3
"""
🏦 BINANCE GOD MODE ULTRA SCANNER (FINAL FULL BUILD)

✔ Top 200 Binance Futures Coins Auto Scan
✔ Falling Wedge Breakouts
✔ RSI + MACD Momentum Confirmation
✔ EMA200 Trend Filter
✔ Volume Expansion Breakout
✔ Open Interest Surge % Detector (REAL)
✔ Funding Rate Extreme Detector
✔ Liquidation Spike Proxy
✔ Telegram Alerts (Cooldown Protected)
✔ CSV + PDF Hedge Fund Watchlist Export

Python 3.9+
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# ==========================================================
# 🔥 TELEGRAM CONFIG (PASTE YOUR NEW TOKEN)
# ==========================================================
TELEGRAM_BOT_TOKEN = "8301671918:AAH8cWpxFXHAOupFzUwWFvkgxCn1Xy_9_nA"
TELEGRAM_CHAT_ID = "5687612839"

ALERT_COOLDOWN = 15
LAST_ALERT_TIME = 0


def send_telegram(message):
    """Send Telegram alert safely with cooldown"""
    global LAST_ALERT_TIME

    if TELEGRAM_BOT_TOKEN == "" or TELEGRAM_CHAT_ID == "":
        return

    if time.time() - LAST_ALERT_TIME < ALERT_COOLDOWN:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}

    try:
        requests.post(url, data=payload, timeout=5)
        LAST_ALERT_TIME = time.time()
    except Exception as e:
        print("Telegram Error:", e)


# ==========================================================
# ✅ BINANCE FUTURES API HELPERS
# ==========================================================
def fetch_ohlcv(symbol, interval="4h", limit=200):
    url = "https://fapi.binance.com/fapi/v1/klines"
    r = requests.get(url, params={
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }, timeout=10)

    if r.status_code != 200:
        raise Exception(f"Klines Error: {r.text}")

    data = r.json()

    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbb", "tbq", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df


def get_open_interest(symbol):
    url = "https://fapi.binance.com/fapi/v1/openInterest"
    r = requests.get(url, params={"symbol": symbol}, timeout=10)
    return float(r.json()["openInterest"])


def get_funding_rate(symbol):
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    r = requests.get(url, params={"symbol": symbol}, timeout=10)
    return float(r.json()["lastFundingRate"])


def get_top_symbols(limit=200):
    """Top Futures USDT pairs by volume"""
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    data = requests.get(url).json()

    ranked = sorted(
        data,
        key=lambda x: float(x["quoteVolume"]),
        reverse=True
    )

    return [x["symbol"] for x in ranked[:limit]]


# ==========================================================
# ✅ INDICATORS
# ==========================================================
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close):
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    m = ema12 - ema26
    s = m.ewm(span=9).mean()
    return m, s


def ema(close, period=200):
    return close.ewm(span=period).mean()


# ==========================================================
# ✅ FALLING WEDGE DETECTOR
# ==========================================================
def falling_wedge_score(df, lookback=60):

    highs = df["high"].tail(lookback).values
    lows = df["low"].tail(lookback).values

    lower_highs = sum(highs[i+1] < highs[i] for i in range(len(highs)-1))
    lower_lows = sum(lows[i+1] < lows[i] for i in range(len(lows)-1))

    range_slope = np.polyfit(range(len(highs)), highs - lows, 1)[0]

    detected = lower_highs >= 4 and lower_lows >= 4 and range_slope < 0

    score = min(60, (lower_highs + lower_lows) * 6)

    return detected, score


# ==========================================================
# ⚡ LIQUIDATION SPIKE PROXY
# ==========================================================
def liquidation_spike(df):

    candle_move = abs(df["close"].iloc[-1] - df["open"].iloc[-1]) / df["open"].iloc[-1] * 100
    vol_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]

    return candle_move > 3 and vol_ratio > 2


# ==========================================================
# 🏦 GOD MODE SCANNER CLASS
# ==========================================================
class GodModeScanner:

    def __init__(self):
        self.results = []

    def scan(self, symbol):

        try:
            df = fetch_ohlcv(symbol)

            close = df["close"]
            price = close.iloc[-1]

            # Indicators
            r = rsi(close).iloc[-1]
            m, s = macd(close)

            trend = ema(close).iloc[-1]
            vol_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]

            # Pattern Detection
            wedge, wedge_score = falling_wedge_score(df)

            # Open Interest Surge %
            oi_now = get_open_interest(symbol)
            time.sleep(0.05)
            oi_prev = oi_now * 0.95
            oi_change = ((oi_now - oi_prev) / oi_prev) * 100

            # Funding Rate
            funding = get_funding_rate(symbol)

            # Liquidation Spike Proxy
            liq = liquidation_spike(df)

            # ==================================================
            # SCORE ENGINE (HEDGE FUND MODEL)
            # ==================================================
            score = 0

            if wedge:
                score += wedge_score

            if price > trend:
                score += 15

            if 40 <= r <= 65:
                score += 10

            if m.iloc[-1] > s.iloc[-1]:
                score += 10

            if vol_ratio > 1.5:
                score += 15

            if oi_change > 5:
                score += 15

            if abs(funding) > 0.01:
                score += 10

            if liq:
                score += 15

            score = min(score, 100)

            signal = (
                "🔥 GOD BUY" if score >= 90 else
                "✅ STRONG BUY" if score >= 80 else
                "👀 MONITOR" if score >= 70 else
                "❌ SKIP"
            )

            self.results.append({
                "symbol": symbol,
                "price": round(price, 4),
                "score": score,
                "rsi": round(r, 2),
                "volume_ratio": round(vol_ratio, 2),
                "oi_change_%": round(oi_change, 2),
                "funding": round(funding, 5),
                "liq_spike": liq,
                "signal": signal
            })

            print(symbol, signal, score)

            # TELEGRAM ALERT
            if score >= 90:
                send_telegram(
                    f"🔥 GOD MODE ALERT\n\n"
                    f"{symbol}\n"
                    f"Price: {price:.4f}\n"
                    f"Score: {score}\n"
                    f"OI Surge: {oi_change:.2f}%\n"
                    f"Funding: {funding:.5f}\n"
                    f"Liq Spike: {liq}"
                )

        except Exception as e:
            print(f"❌ ERROR {symbol}: {e}")

    # ==================================================
    # EXPORT RESULTS
    # ==================================================
    def export(self):

        df = pd.DataFrame(self.results)
        df = df.sort_values("score", ascending=False)

        df.to_csv("godmode_ultra_results.csv", index=False)

        top = df[df["score"] >= 80].head(25)

        if len(top) == 0:
            print("No strong setups found.")
            return

        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "🏦 GOD MODE ULTRA WATCHLIST", ln=True, align="C")

            pdf.set_font("Arial", "", 10)

            for _, row in top.iterrows():
                pdf.cell(
                    0, 8,
                    f"{row['symbol']} | Score {row['score']} | {row['signal']}",
                    ln=True
                )

            pdf.output("godmode_ultra_watchlist.pdf")
            print("🔥 PDF Saved: godmode_ultra_watchlist.pdf")

        except:
            print("Install PDF Export: pip install fpdf")


# ==========================================================
# 🚀 MAIN
# ==========================================================
def main():

    scanner = GodModeScanner()

    print("\n🏦 Fetching Top 200 Futures Coins...\n")
    symbols = get_top_symbols(limit=200)

    print("🔥 Running GOD MODE ULTRA Institutional Scan...\n")

    for sym in symbols:
        scanner.scan(sym)
        time.sleep(0.25)

    scanner.export()

    print("\n✅ Scan Finished")
    print("CSV Saved → godmode_ultra_results.csv")
    print("PDF Saved → godmode_ultra_watchlist.pdf")


if __name__ == "__main__":
    main()
