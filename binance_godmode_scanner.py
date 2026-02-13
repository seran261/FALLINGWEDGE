#!/usr/bin/env python3
"""
🏦 BINANCE HEDGE FUND GOD MODE SCANNER

✔ Auto Top 200 Futures Coins
✔ Falling Wedge Breakouts
✔ Trend + Momentum + Volume Confirmation
✔ Institutional Score Ranking
✔ CSV + PDF Watchlist Report

Python 3.9+
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import logging

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ================= BINANCE DATA =================
def fetch_ohlcv(symbol="BTCUSDT", interval="4h", limit=300):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)

    if r.status_code != 200:
        raise Exception("Binance API Error")

    data = r.json()

    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "close_time","qav","trades","tbb","tbq","ignore"
    ])

    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    return df


# ================= TOP FUTURES COINS =================
def get_top_futures_symbols(limit=200):
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    data = requests.get(url).json()

    usdt_pairs = [x for x in data if "USDT" in x["symbol"]]

    ranked = sorted(
        usdt_pairs,
        key=lambda x: float(x["quoteVolume"]),
        reverse=True
    )

    symbols = [x["symbol"] for x in ranked[:limit]]
    return symbols


# ================= INDICATORS =================
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


# ================= PATTERN ENGINE =================
def falling_wedge_score(df, lookback=60):

    highs = df["high"].tail(lookback).values
    lows = df["low"].tail(lookback).values

    lower_highs = sum(highs[i+1] < highs[i] for i in range(len(highs)-1))
    lower_lows = sum(lows[i+1] < lows[i] for i in range(len(lows)-1))

    slope = np.polyfit(range(len(highs)), highs, 1)[0]

    detected = lower_highs >= 4 and lower_lows >= 4 and slope < 0

    score = min(60, (lower_highs + lower_lows) * 6)

    return detected, score


# ================= GOD MODE SCANNER =================
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

            # Pattern
            wedge, wedge_score = falling_wedge_score(df)

            # Breakout Strength Candle
            candle_strength = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100

            # ================= SCORE ENGINE =================
            score = 0

            # Pattern Weight
            if wedge:
                score += wedge_score

            # Trend Filter
            if price > trend:
                score += 15

            # Momentum Filter
            if 40 <= r <= 65:
                score += 10
            if m.iloc[-1] > s.iloc[-1]:
                score += 10

            # Volume Breakout
            if vol_ratio > 1.5:
                score += 15

            # Candle Breakout Strength
            if candle_strength > 1:
                score += 10

            score = min(score, 100)

            signal = (
                "🔥 GOD BUY" if score >= 85 else
                "✅ STRONG BUY" if score >= 75 else
                "👀 MONITOR" if score >= 65 else
                "❌ SKIP"
            )

            self.results.append({
                "symbol": symbol,
                "price": round(price, 4),
                "rsi": round(r, 2),
                "vol_ratio": round(vol_ratio, 2),
                "score": round(score, 1),
                "signal": signal
            })

            logger.info(f"{symbol} → {signal} ({score})")

        except:
            pass

    # ================= EXPORT =================
    def export(self):

        df = pd.DataFrame(self.results)
        df = df.sort_values("score", ascending=False)

        df.to_csv("godmode_scan_results.csv", index=False)

        top = df[df["score"] >= 75].head(20)

        if len(top) == 0:
            print("No Hedge Fund Setups Found.")
            return

        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "🏦 GOD MODE BINANCE WATCHLIST", ln=True, align="C")

            pdf.set_font("Arial", "", 10)

            for _, row in top.iterrows():
                pdf.cell(
                    0, 8,
                    f"{row['symbol']} | Score: {row['score']} | {row['signal']}",
                    ln=True
                )

            pdf.output("godmode_watchlist.pdf")
            print("🔥 PDF Saved: godmode_watchlist.pdf")

        except:
            print("Install fpdf → pip install fpdf")


# ================= MAIN =================
def main():

    scanner = GodModeScanner()

    print("\n🏦 Fetching Top 200 Binance Futures Coins...\n")
    symbols = get_top_futures_symbols(limit=200)

    print("🔥 Running GOD MODE Institutional Scan...\n")

    for sym in symbols:
        scanner.scan(sym)

    scanner.export()

    print("\n✅ GOD MODE Scan Finished!")
    print("CSV → godmode_scan_results.csv")
    print("PDF → godmode_watchlist.pdf")


if __name__ == "__main__":
    main()
