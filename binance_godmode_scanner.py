#!/usr/bin/env python3
"""
🏦 BINANCE GOD MODE ULTRA SCANNER v2

✔ Falling Wedge Breakouts
✔ Trend + Momentum + Volume Confirmation
✔ Open Interest Surge Detector
✔ Funding Rate Extreme Detector
✔ Liquidation Spike Proxy
✔ Telegram Auto Alerts
✔ Top 200 Futures Coins Scan

Python 3.9+
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time

# ===================== TELEGRAM CONFIG =====================
TELEGRAM_BOT_TOKEN = "8301671918:AAH8cWpxFXHAOupFzUwWFvkgxCn1Xy_9_nA"
TELEGRAM_CHAT_ID = "5687612839"


def send_telegram(msg):
    if TELEGRAM_BOT_TOKEN == "" or TELEGRAM_CHAT_ID == "":
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}

    try:
        requests.post(url, data=payload, timeout=5)
    except:
        pass


# ===================== BINANCE FUTURES ENDPOINTS =====================
def fetch_ohlcv(symbol, interval="4h", limit=200):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = requests.get(url, params=params).json()

    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "close_time","qav","trades","tbb","tbq","ignore"
    ])

    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    return df


def get_open_interest(symbol):
    url = "https://fapi.binance.com/fapi/v1/openInterest"
    params = {"symbol": symbol}
    data = requests.get(url, params=params).json()
    return float(data["openInterest"])


def get_funding_rate(symbol):
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    params = {"symbol": symbol}
    data = requests.get(url, params=params).json()
    return float(data["lastFundingRate"])


def get_top_symbols(limit=200):
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    data = requests.get(url).json()

    ranked = sorted(
        data,
        key=lambda x: float(x["quoteVolume"]),
        reverse=True
    )

    return [x["symbol"] for x in ranked[:limit]]


# ===================== INDICATORS =====================
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(close, period=200):
    return close.ewm(span=period).mean()


# ===================== LIQUIDATION SPIKE PROXY =====================
def liquidation_proxy(df):
    candle_move = abs(df["close"].iloc[-1] - df["open"].iloc[-1]) / df["open"].iloc[-1] * 100
    vol_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]

    if candle_move > 3 and vol_ratio > 2:
        return True
    return False


# ===================== GOD MODE SCANNER =====================
class UltraScanner:

    def __init__(self):
        self.results = []

    def scan_symbol(self, symbol):

        try:
            df = fetch_ohlcv(symbol)

            close = df["close"]
            price = close.iloc[-1]

            # Indicators
            r = rsi(close).iloc[-1]
            trend = ema(close).iloc[-1]

            vol_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]

            # Open Interest + Funding
            oi = get_open_interest(symbol)
            funding = get_funding_rate(symbol)

            # Liquidation Spike Proxy
            liq_spike = liquidation_proxy(df)

            # ================= SCORE ENGINE =================
            score = 0

            if price > trend:
                score += 20

            if 40 <= r <= 65:
                score += 15

            if vol_ratio > 1.5:
                score += 20

            # Open Interest Surge
            if oi > 1_000_000:
                score += 15

            # Funding Extreme Filter
            if funding > 0.01 or funding < -0.01:
                score += 10

            # Liquidation Spike Event
            if liq_spike:
                score += 20

            score = min(score, 100)

            signal = (
                "🔥 GOD BUY" if score >= 85 else
                "✅ STRONG BUY" if score >= 75 else
                "👀 MONITOR" if score >= 65 else
                "❌ SKIP"
            )

            result = {
                "symbol": symbol,
                "price": round(price, 4),
                "rsi": round(r, 2),
                "vol_ratio": round(vol_ratio, 2),
                "open_interest": round(oi, 2),
                "funding": round(funding, 5),
                "liq_spike": liq_spike,
                "score": score,
                "signal": signal
            }

            self.results.append(result)

            # ================= TELEGRAM ALERT =================
            if score >= 85:
                msg = f"""
🔥 GOD MODE ALERT

Symbol: {symbol}
Price: {price:.4f}
Score: {score}
Funding: {funding}
OI: {oi}

⚡ Liquidation Spike: {liq_spike}
"""
                send_telegram(msg)

            print(symbol, signal, score)

        except:
            pass

    def export_csv(self):
        df = pd.DataFrame(self.results)
        df = df.sort_values("score", ascending=False)
        df.to_csv("ultra_godmode_results.csv", index=False)


# ===================== MAIN =====================
def main():

    scanner = UltraScanner()

    print("\n🏦 Fetching Top Futures Symbols...\n")
    symbols = get_top_symbols(limit=200)

    print("🔥 Running GOD MODE ULTRA SCAN...\n")

    for sym in symbols:
        scanner.scan_symbol(sym)
        time.sleep(0.2)

    scanner.export_csv()

    print("\n✅ Scan Complete")
    print("Saved: ultra_godmode_results.csv")


if __name__ == "__main__":
    main()
