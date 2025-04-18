import os
import base64
import io
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, BaseCrossValidator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LSTM, GRU, Conv1D,
                                     Reshape, Bidirectional, MultiHeadAttention, Lambda)
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, session as flask_session, redirect 
from flask_cors import CORS
import stripe
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBRegressor
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay
import math
from scipy.stats import norm
import matplotlib
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
import threading
import queue
import signal
matplotlib.use('Agg')

# Initialize Flask
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    }
})
app.secret_key = "ureallythoughtyouwereabouttogethissecretkeyyoudumbidiot"
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production
app.config['SESSION_COOKIE_HTTPONLY'] = True

# Stripe API Key (LIVE)
# stripe.api_key = "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"
# Stripe API Key (TEST)
stripe.api_key = "sk_test_51R9A09Pjg9aJM7FIgCBeiJJWLhiLs7Z835fEid4JS8NusmxN3ltePtzLVea4jsRqFcHjqFFRZtEt1PR0a1b4edpR00htboUeNS"

# Mock database (replace with real database in production)
users_db = {}
DEVELOPER_KEY = "47906"

# Dictionary to store active prediction tasks
active_predictions = {}
prediction_lock = threading.Lock()

class PredictionTask:
    def __init__(self, ticker):
        self.ticker = ticker
        self.cancelled = False
        self.thread = None
        self.result_queue = queue.Queue()
        
    def run(self):
        if self.cancelled:
            return
            
        try:
            # Train the ML models
            print(f"Training ML models for {self.ticker}...")
            if self.cancelled:
                return
            models = train_ml_model(self.ticker, window=60)
            
            if self.cancelled:
                return
                
            # Get ML-based prediction
            print(f"Getting ML prediction for {self.ticker}...")
            ml_pred = predict_next_day_close_ml(self.ticker, models)
            
            if self.cancelled:
                return
                
            # Run backtesting
            print(f"Running backtesting for {self.ticker}...")
            backtest_results = backtest_model_ml(self.ticker, models, window=60, test_period_days=63)
            
            if self.cancelled:
                return
                
            # Get basic prediction as fallback
            print(f"Getting basic prediction for {self.ticker}...")
            equal_pred = predict_next_day_close(self.ticker)
            
            if self.cancelled:
                return

            results = {
                "ticker": self.ticker,
                "ml_prediction": {
                    "direction": ml_pred["prediction"],
                    "confidence": ml_pred["confidence_percentage"] / 100,
                    "target_price": ml_pred["predicted_price"],
                    "predicted_move": ml_pred["predicted_pct_move"],
                    "predicted_date": ml_pred["predicted_date"]
                },
                "basic_prediction": equal_pred,
                "backtest_results": {
                    "statistics": backtest_results["stats"],
                    "plots": {
                        "prediction_vs_actual": backtest_results["plot_pred_vs_actual"],
                        "cumulative_return": backtest_results["plot_cumulative_return"],
                        "price_comparison": backtest_results["plot_price_comparison"]
                    }
                },
                "model_metrics": models["model_metrics"]
            }
            
            self.result_queue.put(("success", results))
            
        except Exception as e:
            if not self.cancelled:
                self.result_queue.put(("error", str(e)))
        finally:
            with prediction_lock:
                if self.ticker in active_predictions:
                    del active_predictions[self.ticker]

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        
    def cancel(self):
        self.cancelled = True
        # Clear any remaining items in the queue
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

# ---------------------------- Authentication Routes -----------------------------
@app.route("/api/auth/check", methods=["GET"])
def check_auth():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({"error": "No token provided"}), 401
        
        # Verify token
        try:
            payload = jwt.decode(token, app.secret_key, algorithms=["HS256"])
            return jsonify({
                "isAuthenticated": True,
                "user": {"email": payload["email"]}
            })
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
            
        if email in users_db:
            return jsonify({"error": "Account already exists"}), 409
            
        # Hash password and store user
        hashed_password = generate_password_hash(password)
        users_db[email] = {
            "password": hashed_password,
            "is_subscribed": False
        }
        
        return jsonify({
            "success": True,
            "message": "Account created successfully"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/auth/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
            
        # Check if user exists
        if email not in users_db:
            return jsonify({"error": "Account not found"}), 404
            
        # Verify password
        if not check_password_hash(users_db[email]["password"], password):
            return jsonify({"error": "Incorrect password"}), 401
            
        # Generate JWT token
        token = jwt.encode(
            {
                "email": email,
                "exp": datetime.utcnow() + relativedelta(days=7)
            },
            app.secret_key,
            algorithm="HS256"
        )
        
        return jsonify({
            "success": True,
            "message": "Login successful",
            "token": token
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/auth/developer", methods=["POST"])
def developer_auth():
    try:
        data = request.get_json()
        key = data.get("key")
        
        if not key or key != DEVELOPER_KEY:
            return jsonify({"error": "Invalid developer key"}), 403
            
        # Generate developer JWT token
        token = jwt.encode(
            {
                "email": "developer@vecaid.ai",
                "is_developer": True,
                "exp": datetime.utcnow() + relativedelta(days=7)
            },
            app.secret_key,
            algorithm="HS256"
        )
        
        return jsonify({
            "success": True,
            "message": "Developer authentication successful",
            "token": token
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/auth/logout", methods=["POST"])
def logout():
    try:
        flask_session.clear()
        return jsonify({"message": "Logged out successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------- Info Route -----------------------------
@app.route("/api/info", methods=["POST"])
def get_info():
    try:
        # Get the data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        if 'ticker' not in data:
            return jsonify({"error": "Ticker symbol is required"}), 400

        ticker = data['ticker'].upper()
        
        # Check if there's an existing prediction task for this ticker
        with prediction_lock:
            if ticker in active_predictions:
                return jsonify({"error": "A prediction is already running for this ticker"}), 409
            
            # Create and start new prediction task
            task = PredictionTask(ticker)
            active_predictions[ticker] = task
            
        task.start()
        
        # Wait for result with timeout
        try:
            result_type, result = task.result_queue.get(timeout=300)  # 5-minute timeout
            if result_type == "success":
                return jsonify(result)
            else:
                return jsonify({"error": result}), 500
        except queue.Empty:
            with prediction_lock:
                if ticker in active_predictions:
                    del active_predictions[ticker]
            return jsonify({"error": "Prediction timed out"}), 504

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/info/cancel/<ticker>", methods=["POST"])
def cancel_prediction(ticker):
    ticker = ticker.upper()
    with prediction_lock:
        if ticker in active_predictions:
            task = active_predictions[ticker]
            task.cancel()
            del active_predictions[ticker]
            return jsonify({"message": f"Prediction cancelled for {ticker}"})
    return jsonify({"message": f"No active prediction found for {ticker}"}), 404

# ---------------------------- Subscription Routes -----------------------------
@app.route("/api/subscription/check", methods=["GET"])
def check_subscription():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({"isSubscribed": False})
            
        # Verify token
        try:
            payload = jwt.decode(token, app.secret_key, algorithms=["HS256"])
            email = payload["email"]
            
            # Check if user is developer (they get automatic subscription)
            if payload.get("is_developer"):
                return jsonify({"isSubscribed": True})
                
            # Check if user exists and is subscribed
            if email in users_db:
                return jsonify({"isSubscribed": users_db[email]["is_subscribed"]})
            
            return jsonify({"isSubscribed": False})

        except jwt.InvalidTokenError:
            return jsonify({"isSubscribed": False})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/subscription/create-checkout", methods=["POST"])
def create_checkout_session():
    try:
        # Create Stripe checkout session
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': 'price_1234567890',  # Replace with your actual price ID
                'quantity': 1,
            }],
            mode='subscription',
            success_url=request.host_url + 'subscription/success',
            cancel_url=request.host_url + 'subscription',
        )
        return jsonify({"url": session.url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/subscription/success", methods=["GET"])
def subscription_success():
    try:
        session_id = request.args.get("session_id")
        if not session_id:
            return jsonify({"error": "No session ID provided"}), 400
            
        # Retrieve checkout session
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        
        # Update user subscription status
        customer_email = checkout_session.customer_details.email
        if customer_email in users_db:
            users_db[customer_email]["is_subscribed"] = True
            
        return jsonify({
            "success": True,
            "message": "Subscription activated successfully"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------- Prediction Routes -----------------------------
@app.route("/api/prediction", methods=["POST"])
def get_prediction():
    try:
        # Verify authentication and subscription
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({"error": "Authentication required"}), 401
            
        try:
            payload = jwt.decode(token, app.secret_key, algorithms=["HS256"])
            email = payload["email"]
            
            # Allow access for developers
            if not payload.get("is_developer"):
                # Check subscription for regular users
                if email not in users_db or not users_db[email]["is_subscribed"]:
                    return jsonify({"error": "Subscription required"}), 403
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
            
        # Get prediction data
        data = request.get_json()
        if not data or 'ticker' not in data:
            return jsonify({"error": "Ticker symbol is required"}), 400
            
        ticker = data['ticker'].upper()
        
        # Retrieve session & subscription
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        print("✅ Checkout session:", checkout_session)

        subscription = stripe.Subscription.retrieve(checkout_session.subscription)
        print("✅ Stripe subscription:", subscription)

        # ✅ Preserve existing email from login if available
        if 'email' not in flask_session:
            flask_session['email'] = checkout_session.customer_email or "adhitanya@gmail.com"

        flask_session['stripe_customer_id'] = checkout_session.customer or "test_customer_id"
        flask_session['stripe_subscription_id'] = subscription.id or "test_sub_id"
        flask_session['subscribed'] = True

        print("✅ Final session:", dict(flask_session))

        return jsonify({"message": "✅ You're now subscribed to Vecaid Premium!"})
    except Exception as e:
        print("❌ Error in /success:", e)

        # ✅ Still allow predictions by setting fallback session
        if 'email' not in flask_session:
            flask_session['email'] = "adhitanya@gmail.com"
        flask_session['subscribed'] = True

        return jsonify({"warning": "Session fallback used", "error": str(e)}), 200






@app.route("/cancel")
def cancel():
    return jsonify({"message": "❌ Subscription canceled. Try again anytime."})

@app.route("/me")
def me():
    return jsonify({
        "email": flask_session.get("email"),
        "subscribed": flask_session.get("subscribed"),
        "stripe_customer_id": flask_session.get("stripe_customer_id"),
        "stripe_subscription_id": flask_session.get("stripe_subscription_id"),
        "session": dict(flask_session)  # Optional: full raw session
    })



# ---------------------------- Locked -----------------------------
@app.route("/locked")
def locked():
    return jsonify({"message": "🚫 You need to subscribe to access predictions."})




###############################################################################
# Global Constants
###############################################################################
FUND_WEIGHT = 0.3
OPTIONS_WEIGHT = 0.1
SIGMA = 2.5  # Example sigma from historical residuals

###############################################################################
# YEAR-MONTH TIME SERIES SPLIT
###############################################################################


class YearMonthTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=4, year_len=504, month_len=42):
        self.n_splits = n_splits
        self.year_len = year_len
        self.month_len = month_len

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        for i in range(1, self.n_splits + 1):
            train_end = i * self.year_len
            test_start = train_end
            test_end = train_end + self.month_len
            if test_end > n_samples:
                break
            yield np.arange(0, train_end), np.arange(test_start, test_end)

###############################################################################
# DATA HELPER FUNCTIONS
###############################################################################


def get_multi_timeframe_data(ticker, start_date, end_date, intervals=["1d", "1wk"]):
    dfs = []
    for interval in intervals:
        df = yf.download(ticker, start=start_date, end=end_date,
                         interval=interval, progress=False)
        if df.empty:
            print(f"No {interval} data found for {ticker} in the given range.")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.levels[0]:
                df = df[ticker]
            else:
                df = df.xs(ticker, level=1, axis=1)
        suffix = "_" + interval.replace("1", "")
        df.columns = [c + suffix for c in df.columns]
        df.reset_index(inplace=True)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()  # Return empty if no intervals found
    merged_df = dfs[0]
    for i in range(1, len(dfs)):
        merged_df = pd.merge(merged_df, dfs[i], on="Date", how="outer")
    merged_df.set_index("Date", inplace=True)
    merged_df.sort_index(inplace=True)
    return merged_df


def get_intraday_features(ticker, days=60, interval="5m"):
    intraday = yf.download(
        ticker, period=f"{days}d", interval=interval, progress=False)
    if intraday.empty:
        print(
            f"No intraday {interval} data found for {ticker} in last {days} days.")
        return pd.DataFrame()
    if isinstance(intraday.columns, pd.MultiIndex):
        if ticker in intraday.columns.levels[0]:
            intraday = intraday[ticker]
        else:
            intraday = intraday.xs(ticker, level=1, axis=1)
    intraday['Date'] = intraday.index.date
    grouped = intraday.groupby('Date')
    daily_agg = pd.DataFrame()
    daily_agg['intraday_high'] = grouped['High'].max()
    daily_agg['intraday_low'] = grouped['Low'].min()
    daily_agg['intraday_range'] = daily_agg['intraday_high'] - \
        daily_agg['intraday_low']
    daily_agg['intraday_vol'] = grouped['Volume'].sum()
    daily_agg['intraday_avg_vol'] = grouped['Volume'].mean()
    daily_agg.index = pd.to_datetime(daily_agg.index)
    daily_agg.sort_index(inplace=True)
    return daily_agg

###############################################################################
# SENTIMENT FUNCTION
###############################################################################


def get_sentiment(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return 0
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find_all('h3')
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(
            h.get_text())['compound'] for h in headlines]
        return sum(scores) / len(scores) if scores else 0
    except Exception as e:
        print("Error during sentiment scraping:", e)
        return 0

###############################################################################
# ATR and ADX
###############################################################################


def compute_atr(data, period=14):
    tr = []
    for i in range(1, len(data)):
        high = data['High'].iloc[i]
        low = data['Low'].iloc[i]
        prev_close = data['Close'].iloc[i-1]
        tr.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    atr = pd.Series(tr).rolling(window=period).mean()
    atr = pd.concat([pd.Series([np.nan]), atr], ignore_index=True)
    return atr


def adx(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr = pd.DataFrame({
        'tr': np.maximum(high - low, np.maximum(np.abs(high - close.shift()), np.abs(low - close.shift())))
    })
    atr = tr['tr'].rolling(window=period).mean()
    up_move = high.diff()
    down_move = low.diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).sum() / atr)
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx_val = pd.Series(dx).rolling(window=period).mean()
    data['DI_plus'] = plus_di
    data['DI_minus'] = minus_di
    data['ADX'] = adx_val
    return data

###############################################################################
# SHORT-WINDOW UTILS and TECHNICAL INDICATORS
###############################################################################


def short_atr(data, period=5):
    tr = []
    for i in range(1, len(data)):
        high = data['High'].iloc[i]
        low = data['Low'].iloc[i]
        prev_close = data['Close'].iloc[i-1]
        tr.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    if len(tr) < period:
        return 0
    return np.mean(tr[-period:])


def short_momentum_signal(data, period=5, threshold=1.5):
    if len(data) < period:
        return 0
    mom_val = (data['Close'].iloc[-1] - data['Close'].iloc[-period]
               ) / data['Close'].iloc[-period] * 100
    if mom_val > threshold:
        return 1
    elif mom_val < -threshold:
        return -1
    return 0


def linear_regression_trend(data, period=20):
    if len(data) < period:
        return 0
    x = np.arange(period)
    y = data['Close'].iloc[-period:].values
    slope = np.polyfit(x, y, 1)[0]
    norm_slope = slope / np.mean(y)
    return np.sign(norm_slope)


def supertrend(data, period=10, multiplier=3):
    atr = compute_atr(data, period)
    hl2 = (data['High'] + data['Low']) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    last_close = data['Close'].iloc[-1]
    if last_close > upperband.iloc[-1]:
        return 1
    elif last_close < lowerband.iloc[-1]:
        return -1
    return 0


def bollinger_bands(data, period=20, num_std=2):
    bb_middle = data['Close'].rolling(window=period).mean()
    bb_std = data['Close'].rolling(window=period).std()
    bb_upper = bb_middle + num_std * bb_std
    bb_lower = bb_middle - num_std * bb_std
    data['BB_middle'] = bb_middle
    data['BB_upper'] = bb_upper
    data['BB_lower'] = bb_lower
    data['BB_width'] = (bb_upper - bb_lower) / bb_middle
    data['SMA20'] = bb_middle
    data['upper_band'] = bb_upper
    data['lower_band'] = bb_lower
    return data


def compute_mfi(data, period=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(
        typical_price <= typical_price.shift(1), 0)
    pos_mf = positive_flow.rolling(window=period).sum()
    neg_mf = negative_flow.rolling(window=period).sum()
    mfi = 100 - (100 / (1 + pos_mf / neg_mf))
    return mfi


def compute_force_index(data, period=1):
    force_index = data['Close'].diff(period) * data['Volume']
    return force_index


def stochastic_oscillator(data, k_period=14, d_period=3):
    lowest_low = data['Low'].rolling(window=k_period).min()
    highest_high = data['High'].rolling(window=k_period).max()
    data['%K'] = 100 * ((data['Close'] - lowest_low) /
                        (highest_high - lowest_low))
    data['%D'] = data['%K'].rolling(window=d_period).mean()
    return data


def chaikin_money_flow(data, period=20):
    denom = (data['High'] - data['Low']).replace(0, np.nan)
    ad = ((data['Close'] - data['Low']) -
          (data['High'] - data['Close'])) / denom * data['Volume']
    cmf = ad.rolling(window=period, min_periods=1).sum() / \
        data['Volume'].rolling(window=period, min_periods=1).sum()
    return cmf


def commodity_channel_index(data, period=20):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    md = tp.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * md)
    return cci


def fwma(data, period=20):
    weights = np.arange(1, period+1)

    def weighted_mean(x):
        if len(x) < period:
            return np.nan
        return np.dot(x, weights) / weights.sum()
    data['FWMA'] = data['Close'].rolling(
        window=period).apply(weighted_mean, raw=True)
    return data


def tema(data, period=20):
    ema1 = data['Close'].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    data['TEMA'] = 3*ema1 - 3*ema2 + ema3
    return data


def get_technical_indicators(data):
    data = data.copy()
    data['HLC3'] = (data['High'] + data['Low'] + data['Close']) / 3
    data = bollinger_bands(data, period=20, num_std=2)
    data = tema(data, period=20)
    data = fwma(data, period=20)

    data['std'] = data['Close'].rolling(window=20).std()
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / roll_down
    data['RSI'] = 100 - (100 / (1 + RS))
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['typical_price'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (data['typical_price'] * data['Volume']
                    ).cumsum() / data['Volume'].cumsum()
    data['ROC_short'] = data['Close'].pct_change(periods=3) * 100
    data = stochastic_oscillator(data)
    data = adx(data)
    data['ROC'] = data['Close'].pct_change() * 100
    data['CMF'] = chaikin_money_flow(data, period=20)
    data['CCI'] = commodity_channel_index(data, period=20)
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()

    # OBV
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv

    data['vol_SMA20'] = data['Volume'].rolling(window=20).mean()
    data['Momentum'] = data['Close'].pct_change(periods=10) * 100
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['Donchian_High'] = data['High'].rolling(window=20).max()
    data['Donchian_Low'] = data['Low'].rolling(window=20).min()
    data['Williams_R'] = -100 * ((data['High'].rolling(window=14).max() - data['Close']) /
                                 (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min()))
    data['ATR'] = compute_atr(data, period=14)
    data['MFI'] = compute_mfi(data, period=14)
    data['Force_Index'] = compute_force_index(data)
    return data

###############################################################################
# STRATEGY FUNCTIONS
###############################################################################


def stochastic_strategy(latest):
    if latest.get('%K', 50) < 20 and latest.get('%D', 50) < 20:
        return 1
    elif latest.get('%K', 50) > 80 and latest.get('%D', 50) > 80:
        return -1
    return 0


def adx_strategy(latest, threshold=25):
    adx_val = latest.get('ADX', 0)
    di_plus = latest.get('DI_plus', 0)
    di_minus = latest.get('DI_minus', 0)
    if adx_val > threshold:
        if di_plus > di_minus:
            return 1
        elif di_minus > di_plus:
            return -1
    return 0


def roc_strategy(latest, roc_threshold=5):
    roc_val = latest.get('ROC', 0)
    if roc_val > roc_threshold:
        return 1
    elif roc_val < -roc_threshold:
        return -1
    return 0


def bb_strategy(latest):
    lower_band = latest.get('lower_band', 0)
    upper_band = latest.get('upper_band', 0)
    close = latest.get('Close', 0)
    if close == 0:
        return 0
    if (close - lower_band)/close < 0.02:
        return 1
    elif (upper_band - close)/close < 0.02:
        return -1
    return 0


def rsi_strategy(latest, lower=30, upper=70):
    rsi_val = latest.get('RSI', 50)
    if rsi_val < lower:
        return 1
    elif rsi_val > upper:
        return -1
    return 0


def macd_strategy(latest):
    macd_val = latest.get('MACD', 0)
    signal_val = latest.get('Signal_Line', 0)
    return 1 if macd_val > signal_val else -1


def sma_crossover_strategy(latest):
    sma50 = latest.get('SMA50', 0)
    sma200 = latest.get('SMA200', 0)
    return 1 if sma50 > sma200 else -1


def obv_strategy(latest):
    current_obv = latest.get('OBV', 0)
    prev_obv = current_obv * 0.99
    if current_obv > prev_obv:
        return 1
    elif current_obv < prev_obv:
        return -1
    return 0


def volume_strategy(latest):
    vol_sma20 = latest.get('vol_SMA20', 0)
    volume = latest.get('Volume', 0)
    close = latest.get('Close', 0)
    sma20 = latest.get('SMA20', 0)
    if vol_sma20 > 0 and volume > 1.5 * vol_sma20:
        if close > sma20:
            return 1
        elif close < sma20:
            return -1
    return 0


def momentum_strategy_strategy(latest, threshold=2):
    mom = latest.get('Momentum', 0)
    if pd.isna(mom):
        return 0
    if mom > threshold:
        return 1
    elif mom < -threshold:
        return -1
    return 0


def bollinger_b_percent_strategy(latest):
    lower_band = latest.get('lower_band', 0)
    upper_band = latest.get('upper_band', 0)
    close = latest.get('Close', 0)
    if upper_band - lower_band == 0:
        return 0
    b_percent = (close - lower_band)/(upper_band - lower_band)
    if b_percent > 0.8:
        return -1
    elif b_percent < 0.2:
        return 1
    return 0


def keltner_channel_strategy(latest, data, atr_multiplier=1.5):
    ema20 = latest.get('EMA20', 0)
    atr_val = latest.get('ATR', 0)
    close = latest.get('Close', 0)
    upper = ema20 + atr_multiplier * atr_val
    lower = ema20 - atr_multiplier * atr_val
    if close > upper:
        return 1
    elif close < lower:
        return -1
    return 0


def donchian_breakout_strategy(latest):
    close = latest.get('Close', 0)
    donch_high = latest.get('Donchian_High', 0)
    donch_low = latest.get('Donchian_Low', 0)
    if close >= donch_high:
        return 1
    elif close <= donch_low:
        return -1
    return 0


def williams_r_strategy(latest):
    w_val = latest.get('Williams_R', -50)
    if w_val < -80:
        return 1
    elif w_val > -20:
        return -1
    return 0


def atr_breakout_strategy(latest):
    high = latest.get('High', 0)
    low = latest.get('Low', 0)
    atr_val = latest.get('ATR', 0)
    if atr_val == 0:
        return 0
    current_tr = high - low
    if current_tr > 1.2*atr_val:
        return 1
    elif current_tr < 0.8*atr_val:
        return -1
    return 0


def vwap_reversion_strategy(latest, threshold=0.01):
    vwap = latest.get('VWAP', 0)
    close = latest.get('Close', 0)
    if vwap == 0:
        return 0
    deviation = (close - vwap)/vwap
    if deviation > threshold:
        return -1
    elif deviation < -threshold:
        return 1
    return 0


def ema_crossover_short_strategy(data):
    if len(data) < 2:
        return 0
    ema9 = data['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = data['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
    return 1 if ema9 > ema21 else -1


def pivot_point_strategy(prev_day):
    return (prev_day['High'] + prev_day['Low'] + prev_day['Close'])/3


def volume_surge_strategy(latest, surge_factor=1.5):
    vol_sma20 = latest.get('vol_SMA20', 0)
    volume = latest.get('Volume', 0)
    if vol_sma20 == 0:
        return 0
    if volume > surge_factor*vol_sma20:
        return 1
    return 0


def opening_range_breakout_strategy(data):
    if len(data) < 2:
        return 0
    prev_day = data.iloc[-2]
    pivot = (prev_day['High'] + prev_day['Low'])/2
    current_close = data.iloc[-1]['Close']
    if current_close > pivot:
        return 1
    elif current_close < pivot:
        return -1
    return 0


def squeeze_momentum_indicator(latest, threshold=0.1):
    bb_width = latest.get('BB_width', 0)
    mom = latest.get('Momentum', 0)
    if bb_width < threshold:
        return 1 if mom > 0 else -1
    return 0


def lorenzian_classification_signal(data, period=20, gamma=1.0, threshold=0.5):
    if len(data) < period:
        return 0
    sma = data['Close'].rolling(window=period).mean()
    diff = data['Close'].iloc[-1] - sma.iloc[-1]
    lorentz = 1/(1 + (diff/gamma)**2) if gamma != 0 else 0
    if lorentz < threshold:
        return np.sign(-diff)
    return 0


def roc_momentum_strategy(data, period=10, threshold=5):
    if len(data) < period:
        return 0
    roc_val = data['Close'].pct_change(periods=period).iloc[-1] * 100
    if roc_val > threshold:
        return 1
    elif roc_val < -threshold:
        return -1
    return 0


def volume_roc_strategy(data, period=10, threshold=10):
    if len(data) < period:
        return 0
    vol0 = data['Volume'].iloc[-1]
    vol_prev = data['Volume'].iloc[-period]
    if vol_prev == 0:
        return 0
    vol_roc = (vol0 - vol_prev)/abs(vol_prev) * 100
    if vol_roc > threshold:
        return 1
    elif vol_roc < -threshold:
        return -1
    return 0

###############################################################################
# COMPOSITE INDICATOR SIGNAL
###############################################################################


def composite_indicator_signal(latest, data=None):
    def capped(val):
        return max(-1, min(1, val))

    base_signals = [
        stochastic_strategy(latest),
        adx_strategy(latest),
        roc_strategy(latest),
        bb_strategy(latest),
        rsi_strategy(latest),
        macd_strategy(latest),
        sma_crossover_strategy(latest),
        obv_strategy(latest),
        volume_strategy(latest),
        momentum_strategy_strategy(latest),
        vwap_reversion_strategy(latest),
        bollinger_b_percent_strategy(latest),
        williams_r_strategy(latest),
        atr_breakout_strategy(latest),
        squeeze_momentum_indicator(latest)
    ]
    base_signals = [capped(s) for s in base_signals]
    total_signal = sum(base_signals)

    if data is not None and len(data) > 1:
        short_vol = short_atr(data, period=5)
        avg_atr_20 = data['ATR'].iloc[-20:].mean() if len(data) >= 20 else 0
        high_vol = (short_vol > avg_atr_20)

        adx_val = latest.get('ADX', 0)
        trending = (adx_val > 25)
        trend_weight = 1.5 if trending else 0.5
        reversion_weight = 1.5 if not trending else 0.5

        short_mom_s = short_momentum_signal(data)
        linreg_s = linear_regression_trend(data)
        supertrend_s = supertrend(data)
        lorenz_s = lorenzian_classification_signal(data)
        roc_mom_s = roc_momentum_strategy(data)
        vol_roc_s = volume_roc_strategy(data)
        orb_s = opening_range_breakout_strategy(data)
        vol_surge_s = volume_surge_strategy(latest, surge_factor=1.5)

        extras = []
        extras.append(capped(short_mom_s * reversion_weight))
        extras.append(capped(linreg_s * trend_weight))
        extras.append(capped(supertrend_s * trend_weight))
        extras.append(capped(lorenz_s))
        extras.append(capped(roc_mom_s * reversion_weight))
        extras.append(capped(vol_roc_s))
        extras.append(capped(orb_s * trend_weight))
        extras.append(capped(vol_surge_s * (1.0 if high_vol else 0.5)))

        total_signal += sum(extras)

    return total_signal

###############################################################################
# Markov Chain & Mean Reversion
###############################################################################


def markov_chain_signal(data):
    if len(data) < 2:
        return 0
    returns = data['Close'].pct_change().dropna()
    if returns.empty:
        return 0
    states = (returns > 0).astype(int)
    transitions = pd.crosstab(states.shift(1), states)
    if transitions.sum().sum() == 0:
        return 0
    markov_prob = transitions.div(transitions.sum(axis=1), axis=0)
    last_state = 1 if returns.iloc[-1] > 0 else 0
    if (last_state in markov_prob.index) and (1 in markov_prob.columns):
        prob_up = markov_prob.loc[last_state, 1]
    else:
        prob_up = 0.5
    if prob_up > 0.65:
        return 1
    elif prob_up < 0.45:
        return -1
    return 0


def mean_reversion_signal(latest):
    sma20 = latest.get('SMA20', np.nan)
    std_val = latest.get('std', 0)
    close = latest.get('Close', 0)
    if pd.isna(sma20) or std_val == 0:
        return 0
    z = (close - sma20)/std_val
    if z > 1.0:
        return -1
    elif z < -1.0:
        return 1
    return 0

###############################################################################
# BASIC TECHNICAL SIGNAL
###############################################################################


def technical_signal(latest):
    signal = 0
    rsi_val = latest.get('RSI', 50)
    if rsi_val < 30:
        signal += 1
    elif rsi_val > 70:
        signal -= 1

    macd_val = latest.get('MACD', 0)
    sig_val = latest.get('Signal_Line', 0)
    signal += (1 if macd_val > sig_val else -1)

    close = latest.get('Close', 0)
    sma20 = latest.get('SMA20', 0)
    if close > sma20:
        signal += 1
    else:
        signal -= 1

    lower_band = latest.get('lower_band', 0)
    upper_band = latest.get('upper_band', 0)
    if close < lower_band:
        signal += 1
    elif close > upper_band:
        signal -= 1

    deviation = 0
    if sma20 != 0:
        deviation = (close - sma20)/sma20
    if deviation > 0.05:
        signal -= 1
    elif deviation < -0.05:
        signal += 1

    signal += composite_indicator_signal(latest, data=None)
    return signal

###############################################################################
# OPTIONS & FUNDAMENTALS
###############################################################################


def options_signal(ticker, current_price):
    try:
        ticker_obj = yf.Ticker(ticker)
        expirations = ticker_obj.options
        if len(expirations) == 0:
            return 0
        expiry = expirations[0]
        opt_chain = ticker_obj.option_chain(expiry)
        calls = opt_chain.calls
        puts = opt_chain.puts
    except Exception as e:
        print("Error retrieving options chain:", e)
        return 0
    if calls.empty and puts.empty:
        return 0

    options_sig = 0
    if not calls.empty and not puts.empty:
        high_call = calls.loc[calls['volume'].idxmax()]
        high_put = puts.loc[puts['volume'].idxmax()]
        if high_call['volume'] >= high_put['volume']:
            options_sig = 1 if current_price < high_call['strike'] else -1
        else:
            options_sig = -1 if current_price > high_put['strike'] else 1
    elif not calls.empty:
        high_call = calls.loc[calls['volume'].idxmax()]
        options_sig = 1 if current_price < high_call['strike'] else -1
    elif not puts.empty:
        high_put = puts.loc[puts['volume'].idxmax()]
        options_sig = -1 if current_price > high_put['strike'] else 1
    return options_sig


def get_fundamental_signal(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        fundamentals = ticker_obj.info
    except:
        return 0
    if not fundamentals or len(fundamentals) == 0:
        return 0
    signal = 0
    pe = fundamentals.get('trailingPE', None)
    if pe is not None:
        if pe < 15:
            signal += 1
        elif pe > 25:
            signal -= 1
    sentiment_score = get_sentiment(ticker)
    if sentiment_score > 0.2:
        signal += 1
    elif sentiment_score < -0.2:
        signal -= 1
    return signal

###############################################################################
# BASIC PREDICTION FUNCTION
###############################################################################


def predict_next_day_close(ticker):
    data = yf.download(ticker, period="60d", interval="1d")
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker} in last 60 days.")
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.levels[0]:
            data = data[ticker]
        else:
            data = data.xs(ticker, level=1, axis=1)
    data = get_technical_indicators(data)
    latest = data.iloc[-1]
    current_price = latest['Close']
    tech_sig = technical_signal(latest)
    markov_sig = markov_chain_signal(data)
    mean_rev_sig = mean_reversion_signal(latest)
    opt_sig = options_signal(ticker, current_price)
    fund_sig = get_fundamental_signal(ticker)

    total_signal = tech_sig + markov_sig + mean_rev_sig + opt_sig + fund_sig
    predicted_pct_move = total_signal * 0.5
    predicted_price = current_price*(1 + predicted_pct_move/100)
    z_score = abs(predicted_pct_move)/SIGMA
    confidence = min(100, norm.cdf(z_score)*100)
    direction = "higher" if predicted_price > current_price else "lower"
    diff = predicted_price - current_price
    predicted_date = data.index[-1] + BDay(1)
    return {
        "prediction": direction,
        "predicted_difference": diff,
        "confidence_percentage": confidence,
        "predicted_price": predicted_price,
        "predicted_date": predicted_date.strftime('%Y-%m-%d')
    }

###############################################################################
# MODEL BUILDERS
###############################################################################


def build_gru_model(input_dim):
    inputs = Input(shape=(input_dim,), name='gru_input')
    x = Reshape((1, input_dim))(inputs)
    x = GRU(64, return_sequences=False)(x)
    output = Dense(1)(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    return model


def build_cnn_lstm_model(input_dim):
    from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D
    inputs = Input(shape=(input_dim,), name='cnn_lstm_input')
    x = Reshape((input_dim, 1))(inputs)
    x = Conv1D(32, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, return_sequences=False)(x)
    output = Dense(1)(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    return model


def build_cnn_lstm_variant_model(input_dim):
    from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D
    inputs = Input(shape=(input_dim,), name='cnn_lstm_variant_input')
    x = Reshape((input_dim, 1))(inputs)
    x = Conv1D(32, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, return_sequences=False)(x)
    output = Dense(1)(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    return model


def build_bidirectional_lstm_model(input_dim):
    from tensorflow.keras.layers import LSTM, Bidirectional
    inputs = Input(shape=(input_dim,), name='bilstm_input')
    x = Reshape((1, input_dim))(inputs)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    output = Dense(1)(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    return model


def build_dnn_model(input_dim):
    inputs = Input(shape=(input_dim,), name='dnn_input')
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    return model


def build_transformer_model(input_dim):
    from tensorflow.keras.layers import MultiHeadAttention, Flatten
    inputs = Input(shape=(input_dim,), name='transformer_input')
    x = Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)
    attn_output = MultiHeadAttention(num_heads=2, key_dim=input_dim)(x, x)
    x = Flatten()(attn_output)
    x = Dense(64, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    return model

###############################################################################
# BAYESIAN OPTIMIZATION FOR XGBOOST
###############################################################################


def bayesian_optimize_xgb(X, y):
    from sklearn.model_selection import cross_val_score, KFold

    def xgb_cv(max_depth, learning_rate, subsample, colsample_bytree):
        max_depth = int(round(max_depth))
        model = XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='reg:squarederror',
            random_state=42,
            n_estimators=100
        )
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X, y, scoring='neg_mean_squared_error', cv=cv)
        return scores.mean()

    optimizer = BayesianOptimization(
        f=xgb_cv,
        pbounds={
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.5, 1)
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=25)
    return optimizer.max

###############################################################################
# TRAIN FINAL ENSEMBLE
###############################################################################


def train_ml_model(ticker, window=60):
    print("Starting training of the ML model...")
    today = datetime.today()
    start_date = (today - relativedelta(years=7)).strftime('%Y-%m-%d')
    split_date = (today - relativedelta(months=3)).strftime('%Y-%m-%d')

    print("Downloading daily data...")
    daily_data = yf.download(ticker, start=start_date,
                             end=split_date, interval="1d")
    if daily_data.empty:
        raise ValueError(
            f"No daily data found for {ticker} from {start_date} to {split_date}.")

    if isinstance(daily_data.columns, pd.MultiIndex):
        if ticker in daily_data.columns.levels[0]:
            daily_data = daily_data[ticker]
        else:
            daily_data = daily_data.xs(ticker, level=1, axis=1)

    print("Downloading weekly data...")
    weekly_data = get_multi_timeframe_data(
        ticker, start_date, split_date, intervals=["1wk"])
    if weekly_data.empty:
        print("No weekly data found; using empty placeholder for weekly_data_ff.")
        weekly_data_ff = pd.DataFrame(index=daily_data.index)
    else:
        weekly_data_ff = weekly_data.resample('1d').ffill()

    daily_data = daily_data.reset_index()
    weekly_data_ff = weekly_data_ff.reset_index()

    data_full = pd.merge(daily_data, weekly_data_ff,
                         on="Date", how='left', suffixes=('', '_wk'))
    data_full.set_index("Date", inplace=True)
    data_full.ffill(inplace=True)

    print("Computing technical indicators on daily+weekly merged data...")
    data = get_technical_indicators(data_full)

    print("Downloading intraday 5-minute data...")
    intraday_agg = get_intraday_features(ticker, days=60, interval="5m")
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    data_merged = pd.merge(data, intraday_agg, how="left",
                           left_index=True, right_index=True)
    data_merged.fillna(0, inplace=True)

    features = []
    targets = []
    print("Creating feature vectors and target values...")
    for i in range(window, len(data_merged) - 1):
        window_data = data_merged.iloc[i-window:i]
        latest_window = window_data.iloc[-1]
        current_price = latest_window['Close']

        tech_sig = technical_signal(latest_window)
        cmf = latest_window.get('CMF', 0)
        cci = latest_window.get('CCI', 0)
        intraday_range = latest_window.get('intraday_range', 0)
        stochastic_sig = stochastic_strategy(latest_window)
        adx_sig = adx_strategy(latest_window)
        roc_sig = roc_strategy(latest_window)
        bb_sig = bb_strategy(latest_window)
        rsi_sig = rsi_strategy(latest_window)
        macd_sig = macd_strategy(latest_window)
        sma_cross_sig = sma_crossover_strategy(latest_window)
        obv_sig = obv_strategy(latest_window)
        volume_sig = volume_strategy(latest_window)
        momentum_sig = momentum_strategy_strategy(latest_window)
        hlc3_val = latest_window.get('HLC3', 0)
        tema_val = latest_window.get('TEMA', 0)
        fwma_val = latest_window.get('FWMA', 0)
        bb_width_val = latest_window.get('BB_width', 0)

        feature_vector = [
            tech_sig, cmf, cci, intraday_range,
            stochastic_sig, adx_sig, roc_sig, bb_sig,
            rsi_sig, macd_sig, sma_cross_sig, obv_sig,
            volume_sig, momentum_sig, hlc3_val, tema_val, fwma_val, bb_width_val
        ]
        features.append(feature_vector)

        next_day_close = data_merged.iloc[i+1]['Close']
        pct_change = ((next_day_close - current_price)/current_price)*100
        targets.append(pct_change)

    X = np.array(features)
    y = np.array(targets)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Applying PCA (retain 95% variance)...")
    pca = PCA(n_components=0.95, svd_solver='full')
    X_pca = pca.fit_transform(X_scaled)
    input_dim = X_pca.shape[1]
    print(f"PCA reduced input dimension to: {input_dim}")

    print("Starting Bayesian optimization for XGBoost hyperparameters...")
    best_params = bayesian_optimize_xgb(X_pca, y)
    print("Best XGBoost parameters found:", best_params)
    xgb_best_params = best_params["params"]
    xgb_best_params["max_depth"] = int(round(xgb_best_params["max_depth"]))
    xgb_best_params["objective"] = 'reg:squarederror'
    xgb_best_params["random_state"] = 42
    xgb_best_params["n_estimators"] = 100

    print("Building and training ensemble models...")
    gru_model = build_gru_model(input_dim)
    gru_model.fit(X_pca, y, epochs=200, batch_size=32,
                  verbose=1, validation_split=0.2)

    cnn_lstm_model = build_cnn_lstm_model(input_dim)
    cnn_lstm_model.fit(X_pca, y, epochs=200, batch_size=32,
                       verbose=1, validation_split=0.2)

    cnn_lstm_variant = build_cnn_lstm_variant_model(input_dim)
    cnn_lstm_variant.fit(X_pca, y, epochs=200, batch_size=32,
                         verbose=1, validation_split=0.2)

    bilstm_model = build_bidirectional_lstm_model(input_dim)
    bilstm_model.fit(X_pca, y, epochs=200, batch_size=32,
                     verbose=1, validation_split=0.2)

    dnn_model = build_dnn_model(input_dim)
    dnn_model.fit(X_pca, y, epochs=200, batch_size=32,
                  verbose=1, validation_split=0.2)

    transformer_model = build_transformer_model(input_dim)
    transformer_model.fit(X_pca, y, epochs=200, batch_size=32,
                          verbose=1, validation_split=0.2)

    from sklearn.metrics import mean_squared_error

    xgb_model_1 = XGBRegressor(**xgb_best_params)
    xgb_model_1.fit(X_pca, y)

    xgb_model_2 = XGBRegressor(
        max_depth=xgb_best_params["max_depth"],
        learning_rate=xgb_best_params["learning_rate"]*1.1,
        subsample=xgb_best_params["subsample"],
        colsample_bytree=xgb_best_params["colsample_bytree"],
        objective='reg:squarederror',
        random_state=42,
        n_estimators=100
    )
    xgb_model_2.fit(X_pca, y)

    xgb_model_3 = XGBRegressor(
        max_depth=xgb_best_params["max_depth"],
        learning_rate=xgb_best_params["learning_rate"]*0.9,
        subsample=xgb_best_params["subsample"],
        colsample_bytree=xgb_best_params["colsample_bytree"],
        objective='reg:squarederror',
        random_state=42,
        n_estimators=100
    )
    xgb_model_3.fit(X_pca, y)

    xgb_model_4 = XGBRegressor(
        max_depth=xgb_best_params["max_depth"],
        learning_rate=xgb_best_params["learning_rate"]*1.2,
        subsample=xgb_best_params["subsample"],
        colsample_bytree=xgb_best_params["colsample_bytree"],
        objective='reg:squarederror',
        random_state=42,
        n_estimators=100
    )
    xgb_model_4.fit(X_pca, y)

    xgb_model_5 = XGBRegressor(
        max_depth=xgb_best_params["max_depth"],
        learning_rate=xgb_best_params["learning_rate"]*0.8,
        subsample=xgb_best_params["subsample"],
        colsample_bytree=xgb_best_params["colsample_bytree"],
        objective='reg:squarederror',
        random_state=42,
        n_estimators=100
    )
    xgb_model_5.fit(X_pca, y)

    xgb_models = [xgb_model_1, xgb_model_2,
                  xgb_model_3, xgb_model_4, xgb_model_5]
    xgb_ensemble_preds = np.mean(
        [m.predict(X_pca).reshape(-1, 1) for m in xgb_models], axis=0)
    rmse_xgb = np.sqrt(mean_squared_error(y, xgb_ensemble_preds))

    rmse_gru = np.sqrt(mean_squared_error(y, gru_model.predict(X_pca)))
    rmse_cnn_lstm = np.sqrt(mean_squared_error(
        y, cnn_lstm_model.predict(X_pca)))
    rmse_cnn_lstm_variant = np.sqrt(
        mean_squared_error(y, cnn_lstm_variant.predict(X_pca)))
    rmse_bilstm = np.sqrt(mean_squared_error(y, bilstm_model.predict(X_pca)))
    rmse_dnn = np.sqrt(mean_squared_error(y, dnn_model.predict(X_pca)))
    rmse_transformer = np.sqrt(mean_squared_error(
        y, transformer_model.predict(X_pca)))

    p_gru_ = gru_model.predict(X_pca).reshape(-1, 1)
    p_cnn_lstm_ = cnn_lstm_model.predict(X_pca).reshape(-1, 1)
    p_cnn_lstm_var_ = cnn_lstm_variant.predict(X_pca).reshape(-1, 1)
    p_bilstm_ = bilstm_model.predict(X_pca).reshape(-1, 1)
    p_dnn_ = dnn_model.predict(X_pca).reshape(-1, 1)
    p_trans_ = transformer_model.predict(X_pca).reshape(-1, 1)
    p_xgb_ = xgb_ensemble_preds.reshape(-1, 1)

    combined_preds = np.concatenate([p_gru_, p_cnn_lstm_, p_cnn_lstm_var_,
                                     p_bilstm_, p_dnn_, p_trans_, p_xgb_], axis=1)

    X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        combined_preds, y, test_size=0.2, random_state=42
    )

    meta_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    meta_model.fit(X_meta_train, y_meta_train, eval_set=[
                   (X_meta_val, y_meta_val)], verbose=True)

    rmse_meta = np.sqrt(mean_squared_error(
        y_meta_val, meta_model.predict(X_meta_val)))

    model_metrics = {
        "gru_rmse": rmse_gru,
        "cnn_lstm_rmse": rmse_cnn_lstm,
        "cnn_lstm_variant_rmse": rmse_cnn_lstm_variant,
        "bilstm_rmse": rmse_bilstm,
        "dnn_rmse": rmse_dnn,
        "transformer_rmse": rmse_transformer,
        "xgb_rmse": rmse_xgb,
        "meta_rmse": rmse_meta
    }
    print("Model performance (RMSE on training data):")
    for k, v in model_metrics.items():
        print(f"{k}: {v}")

    print("Training complete.")
    return {
        "gru_model": gru_model,
        "cnn_lstm_model": cnn_lstm_model,
        "cnn_lstm_variant": cnn_lstm_variant,
        "bilstm_model": bilstm_model,
        "dnn_model": dnn_model,
        "transformer_model": transformer_model,
        "xgb_models": xgb_models,
        "meta_model": meta_model,
        "scaler": scaler,
        "pca": pca,
        "model_metrics": model_metrics
    }

###############################################################################
# PREDICT FINAL (ML-WEIGHTED)
###############################################################################


def predict_next_day_close_ml(ticker, models):
    print("Starting ML-weighted realtime prediction...")
    data = yf.download(ticker, period="60d", interval="1d")
    if data.empty:
        raise ValueError(f"No daily data found for {ticker} in last 60 days.")
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.levels[0]:
            data = data[ticker]
        else:
            data = data.xs(ticker, level=1, axis=1)
    data = get_technical_indicators(data)

    intraday_agg = get_intraday_features(ticker, days=60, interval="5m")
    data.index = pd.to_datetime(data.index)
    data_merged = pd.merge(data, intraday_agg, how="left",
                           left_index=True, right_index=True)
    data_merged.fillna(0, inplace=True)

    latest = data_merged.iloc[-1]
    current_price = latest['Close']

    tech_sig = technical_signal(latest)
    cmf = latest.get('CMF', 0)
    cci = latest.get('CCI', 0)
    intraday_range = latest.get('intraday_range', 0)
    stochastic_sig = stochastic_strategy(latest)
    adx_sig = adx_strategy(latest)
    roc_sig = roc_strategy(latest)
    bb_sig = bb_strategy(latest)
    rsi_sig = rsi_strategy(latest)
    macd_sig = macd_strategy(latest)
    sma_cross_sig = sma_crossover_strategy(latest)
    obv_sig = obv_strategy(latest)
    volume_sig = volume_strategy(latest)
    momentum_sig = momentum_strategy_strategy(latest)
    hlc3_val = latest.get('HLC3', 0)
    tema_val = latest.get('TEMA', 0)
    fwma_val = latest.get('FWMA', 0)
    bb_width_val = latest.get('BB_width', 0)

    feature_vector = np.array([[tech_sig, cmf, cci, intraday_range,
                                stochastic_sig, adx_sig, roc_sig, bb_sig,
                                rsi_sig, macd_sig, sma_cross_sig, obv_sig,
                                volume_sig, momentum_sig, hlc3_val, tema_val, fwma_val, bb_width_val]])

    scaler = models["scaler"]
    pca = models["pca"]
    X_input = scaler.transform(feature_vector)
    X_input_pca = pca.transform(X_input)

    gru_model = models["gru_model"]
    cnn_lstm_model = models["cnn_lstm_model"]
    cnn_lstm_variant = models["cnn_lstm_variant"]
    bilstm_model = models["bilstm_model"]
    dnn_model = models["dnn_model"]
    transformer_model = models["transformer_model"]
    xgb_models = models["xgb_models"]
    meta_model = models["meta_model"]

    p_gru = gru_model.predict(X_input_pca).reshape(-1, 1)
    p_cnn_lstm = cnn_lstm_model.predict(X_input_pca).reshape(-1, 1)
    p_cnn_lstm_variant = cnn_lstm_variant.predict(X_input_pca).reshape(-1, 1)
    p_bilstm = bilstm_model.predict(X_input_pca).reshape(-1, 1)
    p_dnn = dnn_model.predict(X_input_pca).reshape(-1, 1)
    p_trans = transformer_model.predict(X_input_pca).reshape(-1, 1)

    xgb_preds = []
    for m in xgb_models:
        xgb_preds.append(m.predict(X_input_pca).reshape(-1, 1))
    p_xgb = np.mean(xgb_preds, axis=0)

    combined = np.concatenate(
        [p_gru, p_cnn_lstm, p_cnn_lstm_variant, p_bilstm, p_dnn, p_trans, p_xgb], axis=1)
    final_pred = meta_model.predict(combined)[0]

    opt_sig = options_signal(ticker, current_price)
    fund_sig = get_fundamental_signal(ticker)
    final_pred_adj = final_pred + FUND_WEIGHT*fund_sig + OPTIONS_WEIGHT*opt_sig
    predicted_price = current_price*(1 + final_pred_adj/100)
    z_score = abs(final_pred_adj)/SIGMA
    confidence = min(100, norm.cdf(z_score)*100)
    direction = "higher" if predicted_price > current_price else "lower"
    predicted_date = data_merged.index[-1] + BDay(1)

    print("Realtime prediction complete.")
    return {
        "prediction": direction,
        "predicted_difference": predicted_price - current_price,
        "confidence_percentage": confidence,
        "predicted_price": predicted_price,
        "predicted_pct_move": final_pred_adj,
        "predicted_date": predicted_date.strftime('%Y-%m-%d')
    }

###############################################################################
# HELPER: Save a Matplotlib Figure to Base64
###############################################################################


def save_plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return img_base64

###############################################################################
# BACKTEST
###############################################################################


def backtest_model_ml(ticker, models, window=60, test_period_days=63):
    print("Starting backtesting...")
    total_days = test_period_days + window
    data = yf.download(ticker, period=f"{total_days}d", interval="1d")
    if data.empty:
        raise ValueError(
            f"No data found for {ticker} in last {total_days} days.")
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.levels[0]:
            data = data[ticker]
        else:
            data = data.xs(ticker, level=1, axis=1)

    data = get_technical_indicators(data)
    intraday_agg = get_intraday_features(ticker, days=60, interval="5m")
    data.index = pd.to_datetime(data.index)
    data_merged = pd.merge(data, intraday_agg, how="left",
                           left_index=True, right_index=True)
    data_merged.fillna(0, inplace=True)

    dates, predicted_moves, actual_moves = [], [], []
    cumulative_returns = [1]
    predicted_prices, actual_prices = [], []

    scaler = models["scaler"]
    pca = models["pca"]
    gru_model = models["gru_model"]
    cnn_lstm_model = models["cnn_lstm_model"]
    cnn_lstm_variant = models["cnn_lstm_variant"]
    bilstm_model = models["bilstm_model"]
    dnn_model = models["dnn_model"]
    transformer_model = models["transformer_model"]
    xgb_models = models["xgb_models"]
    meta_model = models["meta_model"]

    print("Beginning backtest loop over", len(
        data_merged) - window - 1, "days...")
    for i in range(window, len(data_merged) - 1):
        window_data = data_merged.iloc[i-window:i]
        latest_window = window_data.iloc[-1]
        current_price = latest_window['Close']

        tech_sig = technical_signal(latest_window)
        cmf = latest_window.get('CMF', 0)
        cci = latest_window.get('CCI', 0)
        intraday_range = latest_window.get('intraday_range', 0)
        stochastic_sig = stochastic_strategy(latest_window)
        adx_sig = adx_strategy(latest_window)
        roc_sig = roc_strategy(latest_window)
        bb_sig = bb_strategy(latest_window)
        rsi_sig = rsi_strategy(latest_window)
        macd_sig = macd_strategy(latest_window)
        sma_cross_sig = sma_crossover_strategy(latest_window)
        obv_sig = obv_strategy(latest_window)
        volume_sig = volume_strategy(latest_window)
        momentum_sig = momentum_strategy_strategy(latest_window)
        hlc3_val = latest_window.get('HLC3', 0)
        tema_val = latest_window.get('TEMA', 0)
        fwma_val = latest_window.get('FWMA', 0)
        bb_width_val = latest_window.get('BB_width', 0)

        feature_vector = np.array([[tech_sig, cmf, cci, intraday_range,
                                    stochastic_sig, adx_sig, roc_sig, bb_sig,
                                    rsi_sig, macd_sig, sma_cross_sig, obv_sig,
                                    volume_sig, momentum_sig, hlc3_val, tema_val, fwma_val, bb_width_val]])

        X_input = scaler.transform(feature_vector)
        X_input_pca = pca.transform(X_input)

        p_gru = gru_model.predict(X_input_pca).reshape(-1, 1)
        p_cnn_lstm = cnn_lstm_model.predict(X_input_pca).reshape(-1, 1)
        p_cnn_lstm_variant = cnn_lstm_variant.predict(
            X_input_pca).reshape(-1, 1)
        p_bilstm = bilstm_model.predict(X_input_pca).reshape(-1, 1)
        p_dnn = dnn_model.predict(X_input_pca).reshape(-1, 1)
        p_transformer = transformer_model.predict(X_input_pca).reshape(-1, 1)

        xgb_preds = []
        for m in xgb_models:
            xgb_preds.append(m.predict(X_input_pca).reshape(-1, 1))
        p_xgb = np.mean(xgb_preds, axis=0)

        combined = np.concatenate([p_gru, p_cnn_lstm, p_cnn_lstm_variant,
                                   p_bilstm, p_dnn, p_transformer, p_xgb], axis=1)
        predicted_pct_move_ml = meta_model.predict(combined)[0]

        final_pred = predicted_pct_move_ml
        predicted_moves.append(final_pred)
        pred_direction = 1 if final_pred > 0 else -1 if final_pred < 0 else 0
        pred_price = current_price*(1 + final_pred/100)
        predicted_prices.append(pred_price)

        next_day_close = data_merged.iloc[i+1]['Close']
        actual_pct_move = ((next_day_close - current_price)/current_price)*100
        actual_moves.append(actual_pct_move)
        actual_prices.append(next_day_close)
        dates.append(data_merged.index[i+1])

        trade_return = 0
        if pred_direction != 0:
            correct_direction = 1 if actual_pct_move > 0 else -1 if actual_pct_move < 0 else 0
            if pred_direction == correct_direction:
                trade_return = abs(actual_pct_move)/100
            else:
                trade_return = -abs(actual_pct_move)/100

        cumulative_returns.append(cumulative_returns[-1]*(1 + trade_return))

        if (i - window) % 10 == 0:
            print(
                f"Day {data_merged.index[i+1]}: Current Price={current_price:.2f}, Predicted Move={final_pred:.2f}%")

    predictions_array = np.array(predicted_moves)
    actuals_array = np.array(actual_moves)
    accuracy = np.mean(np.sign(predictions_array) ==
                       np.sign(actuals_array))*100
    mae = np.mean(np.abs(predictions_array - actuals_array))
    mse = np.mean((predictions_array - actuals_array)**2)
    price_diffs = np.array(predicted_prices) - np.array(actual_prices)
    rmse_price = np.sqrt(np.mean(price_diffs**2))

    # Generate plots and save as base64 strings
    plt.figure(figsize=(14, 6))
    plt.plot(dates, predictions_array, label="Predicted % Move",
             color='#203123')  # Dark green
    plt.plot(dates, actuals_array, label="Actual % Move",
             color='#FFC87A')  # Light grey
    plt.xlabel("Date")
    plt.ylabel("Percent Move")
    plt.title(f"Backtest: Predicted vs. Actual % Moves for {ticker}")
    plt.legend()
    plt.grid(True)
    img1 = save_plot_to_base64()

    plt.figure(figsize=(14, 6))
    plt.plot(dates, cumulative_returns[1:], label="Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title(f"Cumulative Return from Trading Strategy on {ticker}")
    plt.legend()
    plt.grid(True)
    img2 = save_plot_to_base64()

    plt.figure(figsize=(14, 6))
    plt.plot(dates, predicted_prices, label="Predicted Price", color='#203123')
    plt.plot(dates, actual_prices, label="Actual Price", color='#FFC87A')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Predicted vs. Actual Price for {ticker}")
    plt.legend()
    plt.grid(True)
    img3 = save_plot_to_base64()

    stats = {
        "accuracy_percentage": accuracy,
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "rmse_price": rmse_price,
    }
    print("Backtesting Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("Backtesting complete.")
    return {
        "stats": stats,
        "plot_pred_vs_actual": img1,
        "plot_cumulative_return": img2,
        "plot_price_comparison": img3
    }

###############################################################################
# FLASK ROUTES
###############################################################################


@app.route("/", methods=["GET"])
def index():
    """
    Health check or welcome route for backend.
    This no longer serves HTML. React handles the frontend.
    """
    print("Vecaid backend is running!")
    return jsonify({"message": "✅ Vecaid backend is running!"})



# ---------------------------- Start Server -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
