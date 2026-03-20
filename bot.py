import discord
import alpaca_trade_api as tradeapi
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==== KEYS ====
DISCORD_TOKEN = "DISCORD_TOKEN"
API_KEY = "your_alpaca_key"
SECRET_KEY = "your_secret"
BASE_URL = "https://paper-api.alpaca.markets"

# ==== DISCORD ====
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# ==== API ====
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

# ==== GLOBALS ====
WATCHLIST = ["AAPL","TSLA","NVDA","AMD","META","MSFT","AMZN","GOOGL","SPY","QQQ"]
last_signals = {}
signals_db = pd.DataFrame(columns=["symbol","action","price","time","tp","sl","closed","result"])

scaler = StandardScaler()
analyzer = SentimentIntensityAnalyzer()

# =========================
# MARKET HOURS FILTER
# =========================
def market_open():
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)

    if now.weekday() > 4:
        return False

    open_time = now.replace(hour=9, minute=30, second=0)
    close_time = now.replace(hour=16, minute=0, second=0)

    return open_time <= now <= close_time

# =========================
# NEWS (ALPACA)
# =========================
def get_news(symbol):
    try:
        news = api.get_news(symbol, limit=3)
        scores = []
        for n in news:
            text = n.headline + " " + (n.summary or "")
            scores.append(analyzer.polarity_scores(text)["compound"])
        return np.mean(scores) if scores else 0
    except:
        return 0

# =========================
# AI MODEL
# =========================
def ai_predict(close, volume):
    try:
        if len(close) < 40:
            return 0.5

        X, y = [], []

        for i in range(25, len(close)-1):
            w = close[i-25:i]

            returns = np.diff(w) / w[:-1]
            momentum = w[-1] - w[0]
            vol = np.std(w)
            vol_spike = volume[i] / np.mean(volume[i-25:i])

            features = np.concatenate([returns, [momentum, vol, vol_spike]])
            X.append(features)
            y.append(1 if close[i+1] > close[i] else 0)

        X = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=300)
        model.fit(X, y)

        w = close[-25:]
        returns = np.diff(w) / w[:-1]
        momentum = w[-1] - w[0]
        vol = np.std(w)
        vol_spike = volume[-1] / np.mean(volume[-25:])

        latest = scaler.transform([np.concatenate([returns, [momentum, vol, vol_spike]])])

        return float(model.predict_proba(latest)[0][1])

    except:
        return 0.5

# =========================
# INDICATORS
# =========================
def indicators(df):
    close = df['close']
    volume = df['volume']

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss

    rsi = 100 - (100/(1+rs))
    ma9 = close.rolling(9).mean()
    ma21 = close.rolling(21).mean()

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()

    vol_spike = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]
    trend_strength = abs(ma9.iloc[-1] - ma21.iloc[-1]) / close.iloc[-1]

    return {
        "rsi": float(np.nan_to_num(rsi.iloc[-1])),
        "ma_short": float(ma9.iloc[-1]),
        "ma_long": float(ma21.iloc[-1]),
        "macd": float(macd.iloc[-1]),
        "signal": float(signal.iloc[-1]),
        "vol_spike": float(vol_spike),
        "trend_strength": float(trend_strength)
    }

# =========================
# RISK
# =========================
def risk(price, action):
    if action == "BUY":
        return price*1.006, price*0.997
    else:
        return price*0.994, price*1.003

# =========================
# SIGNAL
# =========================
def get_signal(symbol):
    try:
        df1 = api.get_bars(symbol, "1Min", limit=120).df
        df5 = api.get_bars(symbol, "5Min", limit=120).df
        df15 = api.get_bars(symbol, "15Min", limit=120).df

        if len(df1)<50 or len(df5)<50 or len(df15)<50:
            return None, None

        ind1 = indicators(df1)
        ind5 = indicators(df5)
        ind15 = indicators(df15)

        close = df1['close'].values
        volume = df1['volume'].values

        ai = np.clip(ai_predict(close, volume),0,1)
        news = np.clip(get_news(symbol),-1,1)

        score = 0
        confirm = 0

        if ai>0.6: score+=2; confirm+=1
        elif ai<0.4: score-=2; confirm+=1

        if ind1["rsi"]<30: score+=1; confirm+=1
        elif ind1["rsi"]>70: score-=1; confirm+=1

        trend1 = ind1["ma_short"]>ind1["ma_long"]
        trend5 = ind5["ma_short"]>ind5["ma_long"]
        trend15 = ind15["ma_short"]>ind15["ma_long"]

        if trend1==trend5==trend15:
            score += 2 if trend1 else -2
            confirm+=2

        score += 1 if ind1["macd"]>ind1["signal"] else -1

        if ind1["vol_spike"]>1.5:
            score+=1; confirm+=1

        if news>0.2: score+=1
        elif news<-0.2: score-=1

        if score>0 and news<-0.4: return None,None
        if score<0 and news>0.4: return None,None

        if ind1["trend_strength"]<0.002:
            return None,None

        confidence = int(min((confirm/6)*100,100))
        price = float(df1['close'].iloc[-1])

        if score>=4:
            action="BUY"
        elif score<=-4:
            action="SELL"
        else:
            return None,None

        tp,sl = risk(price,action)

        global signals_db
        signals_db.loc[len(signals_db)] = [symbol,action,price,pd.Timestamp.now(),tp,sl,False,None]

        msg = f"{symbol} {action} | ${price:.2f} | AI {ai*100:.0f}% | Conf {confidence}% | TP {tp:.2f} | SL {sl:.2f}"
        return action, msg

    except Exception as e:
        print(symbol, e)
        return None, None

# =========================
# UPDATE TRADES
# =========================
def update_trades():
    global signals_db
    for i,row in signals_db.iterrows():
        if row["closed"]: continue
        try:
            price = api.get_latest_trade(row["symbol"]).price
            if row["action"]=="BUY":
                if price>=row["tp"]:
                    signals_db.at[i,"closed"]=True
                    signals_db.at[i,"result"]="WIN"
                elif price<=row["sl"]:
                    signals_db.at[i,"closed"]=True
                    signals_db.at[i,"result"]="LOSS"
            else:
                if price<=row["tp"]:
                    signals_db.at[i,"closed"]=True
                    signals_db.at[i,"result"]="WIN"
                elif price>=row["sl"]:
                    signals_db.at[i,"closed"]=True
                    signals_db.at[i,"result"]="LOSS"
        except:
            pass

# =========================
# STATS
# =========================
def stats():
    closed = signals_db[signals_db["closed"]==True]
    if len(closed)==0:
        return "No trades yet"
    wins = len(closed[closed["result"]=="WIN"])
    total = len(closed)
    return f"Trades: {total} | Wins: {wins} | Winrate: {wins/total*100:.1f}%"

# =========================
# LOOP
# =========================
async def stock_loop():
    await client.wait_until_ready()
    channel = discord.utils.get(client.get_all_channels(), name="general")

    while not client.is_closed():

        if not market_open():
            print("Market closed...")
            await asyncio.sleep(300)
            continue

        update_trades()

        for stock in WATCHLIST:
            signal, message = get_signal(stock)

            if signal and last_signals.get(stock) != signal:
                last_signals[stock] = signal
                await channel.send(message)

        await asyncio.sleep(60)

# =========================
# DISCORD
# =========================
@client.event
async def on_ready():
    print("BOT READY")
    client.loop.create_task(stock_loop())

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!signal"):
        symbol = message.content.split()[1].upper()
        _, msg = get_signal(symbol)
        await message.channel.send(msg if msg else f"{symbol}: HOLD")

    if message.content.startswith("!stats"):
        await message.channel.send(stats())

# RUN
client.run(DISCORD_TOKEN)
