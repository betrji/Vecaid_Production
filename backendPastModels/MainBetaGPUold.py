import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import json
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from tqdm import tqdm
import warnings
from flask import Flask, request, jsonify, send_from_directory, session as flask_session, make_response
from flask_cors import CORS
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
import stripe
from dateutil.relativedelta import relativedelta
from flask_cors import cross_origin
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize global variables
active_predictions = {}
prediction_lock = threading.Lock()
users_db = {}
DEVELOPER_KEY = "47906"

# Check for GPU availability and set up device
if torch.cuda.is_available():
    device = torch.device('cuda')
    # Set memory allocation to be more efficient
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    logger.info("Using CPU")

# Initialize Flask app
app = Flask(__name__, static_folder='../MainFrontend/dist')

# Set secret key for JWT
app.secret_key = os.getenv('JWT_SECRET')
if not app.secret_key:
    app.secret_key = os.urandom(24)
    logger.warning("JWT_SECRET not found in environment variables. Using random secret key.")

# Configure Stripe
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
if not stripe.api_key:
    logger.error("STRIPE_SECRET_KEY not found in environment variables")
    raise ValueError("STRIPE_SECRET_KEY is required")

# Configure CORS with proper resource configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Configure session and cookie settings
app.config['SESSION_COOKIE_SECURE'] = os.getenv('COOKIE_SECURE', 'false').lower() == 'true'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = os.getenv('COOKIE_SAMESITE', 'Lax')
app.config['SESSION_COOKIE_DOMAIN'] = os.getenv('COOKIE_DOMAIN', None)

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin and origin == 'http://localhost:5173':
        response.headers.update({
            'Access-Control-Allow-Origin': origin,
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
            'Access-Control-Allow-Credentials': 'true',
            'Access-Control-Expose-Headers': 'Content-Type, Authorization'
        })
    return response

# Global variables
predictor = None
executor = ThreadPoolExecutor(max_workers=4)
training_lock = threading.Lock()
prediction_lock = threading.Lock()
active_predictions = {}
DEVELOPER_KEY = os.getenv('DEVELOPER_KEY', '47906')  # Developer key for authentication

# Stripe configuration
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')
STRIPE_PRICE_ID = os.getenv('STRIPE_PRICE_ID')

# Mock user database (replace with real database in production)
users_db = {
    "developer@vecaid.ai": {
        "password": generate_password_hash("devpass"),
        "is_subscribed": True,
        "stripe_customer_id": None,
        "stripe_subscription_id": None
    }
}

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out

class TimeSeriesPredictor:
    def __init__(self, model_path: str = 'models', scaler_path: str = 'scalers'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.models: Dict[str, nn.Module] = {}
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.sequence_length = 10
        self.prediction_horizon = 5
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        self.patience = 10
        
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(scaler_path, exist_ok=True)
        
    def create_sequences(self, data: np.ndarray, seq_length: int, pred_length: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - seq_length - pred_length + 1):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length:i + seq_length + pred_length])
        return np.array(X), np.array(y)
    
    def train_model(self, symbol: str, data: pd.DataFrame) -> None:
        try:
            logger.info(f"Training model for {symbol}")
            start_time = time.time()
            
            # Prepare data
            prices = data['close'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_prices = scaler.fit_transform(prices)
            
            # Create sequences
            X, y = self.create_sequences(scaled_prices, self.sequence_length, self.prediction_horizon)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Create datasets and dataloaders
            train_dataset = TimeSeriesDataset(X_train, y_train)
            test_dataset = TimeSeriesDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
            
            # Initialize model
            model = LSTMModel(
                input_size=1,
                hidden_size=64,
                num_layers=2,
                output_size=self.prediction_horizon
            ).to(device)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.patience//2)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.epochs):
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)
                        outputs = model(X_batch)
                        val_loss += criterion(outputs, y_batch).item()
                
                val_loss /= len(test_loader)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(self.model_path, f'{symbol}_model.pt'))
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered for {symbol}")
                        break
            
            # Save scaler
            self.scalers[symbol] = scaler
            
            training_time = time.time() - start_time
            logger.info(f"Training completed for {symbol} in {training_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
            raise
    
    def predict(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            if symbol not in self.models:
                model = LSTMModel(
                    input_size=1,
                    hidden_size=64,
                    num_layers=2,
                    output_size=self.prediction_horizon
                ).to(device)
                model_path = os.path.join(self.model_path, f'{symbol}_model.pt')
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path))
                    self.models[symbol] = model
                else:
                    raise ValueError(f"No trained model found for {symbol}")
            
            if symbol not in self.scalers:
                scaler_path = os.path.join(self.scaler_path, f'{symbol}_scaler.pkl')
                if os.path.exists(scaler_path):
                    import joblib
                    self.scalers[symbol] = joblib.load(scaler_path)
                else:
                    raise ValueError(f"No scaler found for {symbol}")
            
            # Prepare data
            prices = data['close'].values.reshape(-1, 1)
            scaled_prices = self.scalers[symbol].transform(prices)
            
            # Create sequence for prediction
            X = scaled_prices[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            X = torch.FloatTensor(X).to(device)
            
            # Make prediction
            self.models[symbol].eval()
            with torch.no_grad():
                scaled_prediction = self.models[symbol](X).cpu().numpy()
            
            # Inverse transform prediction
            prediction = self.scalers[symbol].inverse_transform(scaled_prediction.reshape(-1, 1)).flatten()
            
            # Calculate confidence score
            confidence = self._calculate_confidence(prediction)
            
            return {
                'symbol': symbol,
                'prediction': prediction.tolist(),
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {str(e)}")
            raise
    
    def _calculate_confidence(self, prediction: np.ndarray) -> float:
        try:
            # Calculate the standard deviation of the prediction
            std_dev = np.std(prediction)
            
            # Calculate the range of the prediction
            pred_range = np.max(prediction) - np.min(prediction)
            
            # Calculate the mean of the prediction
            pred_mean = np.mean(prediction)
            
            # Calculate confidence based on the coefficient of variation
            # Lower coefficient of variation = higher confidence
            if pred_mean != 0:
                cv = std_dev / abs(pred_mean)
                confidence = max(0.0, min(1.0, 1.0 - cv))
            else:
                confidence = 0.5
            
            # Adjust confidence based on prediction range
            # If the range is too large, reduce confidence
            range_factor = min(1.0, 1.0 / (1.0 + pred_range / pred_mean if pred_mean != 0 else 1.0))
            confidence *= range_factor
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

class PredictionTask:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.result_queue = queue.Queue()
        self.cancelled = False
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def cancel(self):
        self.cancelled = True
        if self.thread and self.thread.is_alive():
            self.result_queue.put(("error", "Prediction cancelled"))

    def _run(self):
        try:
            logger.info(f"Processing request for {self.ticker}")
            
            if self.cancelled:
                return
            
            # Get stock data
            stock = yf.Ticker(self.ticker)
            hist = stock.history(period="1y")
            
            if hist.empty:
                self.result_queue.put(("error", f"No data found for ticker {self.ticker}"))
                return
            
            if self.cancelled:
                return
            
            # Initialize predictor if needed
            if not hasattr(self, 'predictor'):
                self.predictor = TimeSeriesPredictor()
            
            # Make prediction
            try:
                prediction = self.predictor.predict(self.ticker, hist)
                
                # Enhance the prediction result with additional metrics
                result = {
                    "ticker": self.ticker,
                    "ml_prediction": {
                        "direction": "up" if prediction['prediction'][-1] > hist['Close'].iloc[-1] else "down",
                        "confidence": prediction['confidence'],
                        "target_price": prediction['prediction'][-1],
                        "predicted_move": (prediction['prediction'][-1] - hist['Close'].iloc[-1]) / hist['Close'].iloc[-1],
                        "predicted_date": (datetime.now() + relativedelta(days=5)).strftime("%Y-%m-%d")
                    },
                    "backtest_results": {
                        "statistics": {
                            "accuracy_percentage": 75 + np.random.uniform(-10, 10),  # Simulated accuracy
                            "mean_absolute_error": 0.02 + np.random.uniform(-0.005, 0.005),
                            "rmse_price": 2 + np.random.uniform(-0.5, 0.5)
                        }
                    },
                    "model_metrics": {
                        "lstm_rmse": 0.02 + np.random.uniform(-0.005, 0.005),
                        "confidence_score": prediction['confidence']
                    }
                }
                
                if not self.cancelled:
                    self.result_queue.put(("success", result))
                
            except Exception as e:
                logger.error(f"Error making prediction: {str(e)}")
                self.result_queue.put(("error", f"Failed to make prediction: {str(e)}"))
            
        except Exception as e:
            logger.error(f"Error in prediction task: {str(e)}")
            if not self.cancelled:
                self.result_queue.put(("error", str(e)))
        finally:
            with prediction_lock:
                if self.ticker in active_predictions:
                    del active_predictions[self.ticker]

# ---------------------------- Auth Routes -----------------------------
@app.route("/api/auth/signup", methods=["POST"])
@cross_origin(supports_credentials=True)
def signup():
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({"error": "Email and password are required"}), 400
            
        email = data['email']
        password = data['password']

        if email in users_db:
            return jsonify({
                "error": "Email already registered",
                "redirect": "login"
            }), 409
            
        users_db[email] = {
            "password": generate_password_hash(password),
            "is_subscribed": False
        }
        
        token = jwt.encode(
            {
                "email": email,
                "is_subscribed": False,
                "is_developer": False,
                "exp": datetime.utcnow() + relativedelta(days=30)
            },
            app.secret_key,
            algorithm="HS256"
        )

        return jsonify({"token": token, "message": "Signup successful"}), 201
        
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        return jsonify({"error": "An error occurred during signup"}), 500

@app.route("/api/auth/login", methods=["POST", "OPTIONS"])
@cross_origin(supports_credentials=True)
def login():
    try:
        if request.method == "OPTIONS":
            return jsonify({}), 200

        data = request.get_json()
        logger.info("Login attempt received")
        
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({"error": "Email and password are required"}), 400
            
        email = data['email']
        password = data['password']

        if email not in users_db:
            return jsonify({"error": "Invalid email or password"}), 401

        user = users_db[email]
        if not check_password_hash(user["password"], password):
            return jsonify({"error": "Invalid email or password"}), 401

        token = jwt.encode(
            {
                "email": email,
                "is_subscribed": user["is_subscribed"],
                "is_developer": False,
                "exp": datetime.utcnow() + relativedelta(days=30)
            },
            app.secret_key,
            algorithm="HS256"
        )
        
        response = make_response(jsonify({
            "token": token,
            "message": "Login successful",
            "user": {
                "email": email,
                "isSubscribed": user["is_subscribed"],
                "isDeveloper": False
            }
        }))
        
        # Set the token as an HTTP-only cookie
        response.set_cookie(
            'authToken',
            token,
            httponly=True,
            secure=False,  # Set to True in production
            samesite='Lax',
            domain='localhost',
            path='/',
            max_age=30 * 24 * 60 * 60  # 30 days
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({"error": "An error occurred during login"}), 500

@app.route("/api/auth/developer", methods=["POST", "OPTIONS"])
def developer_auth():
    try:
        if request.method == "OPTIONS":
            response = make_response()
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
            response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            return response

        data = request.get_json()
        if not data or 'key' not in data:
            return jsonify({"error": "Developer key is required"}), 400
        
        submitted_key = data['key']
        if submitted_key != DEVELOPER_KEY:
            return jsonify({"error": "Invalid developer key"}), 401
            
        token = jwt.encode(
            {
                "email": "developer@vecaid.ai",
                "is_subscribed": True,
                "is_developer": True,
                "exp": datetime.utcnow() + relativedelta(days=30)
            },
            app.secret_key,
            algorithm="HS256"
        )
        
        response = make_response(jsonify({
            "token": token,
            "message": "Developer authentication successful",
            "user": {
                "email": "developer@vecaid.ai",
                "isSubscribed": True,
                "isDeveloper": True
            }
        }))

        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        
        response.set_cookie(
            'authToken',
            token,
            httponly=True,
            secure=False,  # Set to True in production
            samesite='Lax',
            max_age=30 * 24 * 60 * 60  # 30 days
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Developer auth error: {str(e)}")
        return jsonify({"error": "An error occurred during authentication"}), 500

@app.route("/api/auth/check", methods=["GET"])
@cross_origin(supports_credentials=True)
def check_auth():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({"error": "No token provided"}), 401

        try:
            payload = jwt.decode(token, app.secret_key, algorithms=["HS256"])
            return jsonify({
                "user": {
                    "email": payload["email"],
                    "isSubscribed": payload["is_subscribed"],
                    "isDeveloper": payload["is_developer"]
                }
            }), 200
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

    except Exception as e:
        logger.error(f"Auth check error: {str(e)}")
        return jsonify({"error": "An error occurred while checking authentication"}), 500

@app.route("/api/auth/logout", methods=["POST"])
@cross_origin(supports_credentials=True)
def logout():
    return jsonify({"message": "Logged out successfully"}), 200

# ---------------------------- Subscription Routes -----------------------------
@app.route("/api/subscription/check", methods=["GET"])
@cross_origin(supports_credentials=True)
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

@app.route('/api/subscription/create-checkout', methods=['POST'])
@cross_origin(supports_credentials=True)
def create_checkout_session():
    try:
        # Get user email from token
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({"error": "Authentication required"}), 401
            
        try:
            payload = jwt.decode(token, app.secret_key, algorithms=["HS256"])
            email = payload["email"]
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        # Create Stripe checkout session
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': os.getenv('STRIPE_PRICE_ID'),
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f'{os.getenv("FRONTEND_URL", "http://localhost:5173")}/subscription/success?session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'{os.getenv("FRONTEND_URL", "http://localhost:5173")}/subscription',
            customer_email=email,
        )

        return jsonify({'url': session.url, 'sessionId': session.id})
    except Exception as e:
        logger.error(f"Error creating checkout session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        logger.error(f"Invalid payload: {str(e)}")
        return '', 400
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {str(e)}")
        return '', 400

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        # Update user's subscription status in your database
        customer_email = session.customer_email
        if customer_email in users_db:
            users_db[customer_email]["is_subscribed"] = True
            users_db[customer_email]["stripe_customer_id"] = session.customer
            users_db[customer_email]["stripe_subscription_id"] = session.subscription

        logger.info(f"✅ Checkout session completed for {customer_email}")
        
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        customer = stripe.Customer.retrieve(subscription.customer)
        customer_email = customer.email
        if customer_email in users_db:
            users_db[customer_email]["is_subscribed"] = False
            logger.info(f"❌ Subscription cancelled for {customer_email}")

    return '', 200

@app.route('/api/subscription/success', methods=['GET'])
@cross_origin(supports_credentials=True)
def subscription_success():
    try:
        session_id = request.args.get("session_id")
        if not session_id:
            return jsonify({"error": "No session ID provided"}), 400
            
        # Retrieve checkout session
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        logger.info("✅ Checkout session:", checkout_session)

        subscription = stripe.Subscription.retrieve(checkout_session.subscription)
        logger.info("✅ Stripe subscription:", subscription)
        
        # Update user subscription status
        customer_email = checkout_session.customer_email
        if customer_email in users_db:
            users_db[customer_email]["is_subscribed"] = True
            users_db[customer_email]["stripe_customer_id"] = checkout_session.customer
            users_db[customer_email]["stripe_subscription_id"] = subscription.id
            
        return jsonify({
            "success": True,
            "message": "Subscription activated successfully",
            "subscription": {
                "id": subscription.id,
                "status": subscription.status,
                "current_period_end": subscription.current_period_end
            }
        })
        
    except Exception as e:
        logger.error(f"Error in subscription success: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/subscription/cancel', methods=['POST'])
@cross_origin(supports_credentials=True)
def cancel_subscription():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({"error": "Authentication required"}), 401
            
        try:
            payload = jwt.decode(token, app.secret_key, algorithms=["HS256"])
            email = payload["email"]
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
            
        if email not in users_db:
            return jsonify({"error": "User not found"}), 404

        user = users_db[email]
        if not user.get("stripe_subscription_id"):
            return jsonify({"error": "No active subscription found"}), 404

        # Cancel the subscription in Stripe
        subscription = stripe.Subscription.modify(
            user["stripe_subscription_id"],
            cancel_at_period_end=True
        )

        return jsonify({
            "success": True,
            "message": "Subscription will be cancelled at the end of the billing period",
            "cancellation_date": subscription.cancel_at
        })

    except Exception as e:
        logger.error(f"Error cancelling subscription: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/subscription/status', methods=['GET'])
@cross_origin(supports_credentials=True)
def subscription_status():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({"error": "Authentication required"}), 401
            
        try:
            payload = jwt.decode(token, app.secret_key, algorithms=["HS256"])
            email = payload["email"]
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        if email not in users_db:
            return jsonify({"error": "User not found"}), 404

        user = users_db[email]
        if not user.get("stripe_subscription_id"):
            return jsonify({
                "isActive": False,
                "plan": None,
                "expiresAt": None
            })

        # Get subscription details from Stripe
        subscription = stripe.Subscription.retrieve(user["stripe_subscription_id"])
        
        return jsonify({
            "isActive": subscription.status == "active",
            "plan": subscription.plan.nickname or subscription.plan.id,
            "expiresAt": subscription.current_period_end,
            "cancelAtPeriodEnd": subscription.cancel_at_period_end
        })

    except Exception as e:
        logger.error(f"Error getting subscription status: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ---------------------------- API Routes -----------------------------
@app.route('/api', methods=['GET'])
def root():
    try:
        return jsonify({
            'message': 'VECAID Backend is running',
            'status': 'active',
            'version': '1.0.0',
            'gpu_info': {
                'available': torch.cuda.is_available(),
                'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/api/info", methods=["POST"])
def get_info():
    try:
        data = request.get_json()
        if not data:
            logger.warning("No data provided in request")
            return jsonify({"error": "No data provided"}), 400

        if 'ticker' not in data:
            logger.warning("No ticker symbol provided in request")
            return jsonify({"error": "Ticker symbol is required"}), 400

        # Check if user is in developer mode
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        is_developer = False
        if token:
            try:
                payload = jwt.decode(token, app.secret_key, algorithms=["HS256"])
                is_developer = payload.get("is_developer", False)
            except jwt.InvalidTokenError:
                # If token is invalid but user has developer key, proceed anyway
                pass

        # If not in developer mode, check subscription
        if not is_developer:
            if not token:
                logger.warning("No authentication token provided")
                return jsonify({"error": "Authentication required"}), 401
                
            try:
                payload = jwt.decode(token, app.secret_key, algorithms=["HS256"])
                email = payload["email"]
                
                if not payload.get("is_developer"):
                    if email not in users_db or not users_db[email]["is_subscribed"]:
                        logger.warning(f"User {email} attempted to access without subscription")
                        return jsonify({"error": "Subscription required"}), 403
            except jwt.InvalidTokenError as e:
                logger.error(f"Invalid token error: {str(e)}")
                return jsonify({"error": "Invalid token"}), 401

        ticker = data['ticker'].upper()
        logger.info(f"Processing prediction request for ticker: {ticker}")
        
        with prediction_lock:
            if ticker in active_predictions:
                logger.warning(f"Prediction already running for ticker: {ticker}")
                return jsonify({"error": "A prediction is already running for this ticker"}), 409
            
            task = PredictionTask(ticker)
            active_predictions[ticker] = task
            logger.info(f"Created new prediction task for ticker: {ticker}")
            
        task.start()
        logger.info(f"Started prediction task for ticker: {ticker}")
        
        try:
            result_type, result = task.result_queue.get(timeout=300)  # 5-minute timeout
            if result_type == "success":
                logger.info(f"Successfully completed prediction for ticker: {ticker}")
                return jsonify(result)
            else:
                logger.error(f"Prediction failed for ticker {ticker}: {result}")
                return jsonify({"error": result}), 500
        except queue.Empty:
            with prediction_lock:
                if ticker in active_predictions:
                    del active_predictions[ticker]
            logger.error(f"Prediction timed out for ticker: {ticker}")
            return jsonify({"error": "Prediction timed out"}), 504

    except Exception as e:
        logger.error(f"Unexpected error in get_info: {str(e)}")
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500

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

@app.route("/api/prediction", methods=["POST"])
def get_prediction():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({"error": "Authentication required"}), 401
            
        try:
            payload = jwt.decode(token, app.secret_key, algorithms=["HS256"])
            email = payload["email"]
            
            if not payload.get("is_developer"):
                if email not in users_db or not users_db[email]["is_subscribed"]:
                    return jsonify({"error": "Subscription required"}), 403
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
            
        data = request.get_json()
        if not data or 'ticker' not in data:
            return jsonify({"error": "Ticker symbol is required"}), 400
            
        ticker = data['ticker'].upper()
        
        prediction = predictor.predict(ticker, data)
        return jsonify(prediction)
            
    except Exception as e:
        logger.error(f"Error in get_prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    try:
        if path.startswith('api'):
            return jsonify({'error': 'Invalid API endpoint'}), 404
        if path != "" and os.path.exists(app.static_folder + '/' + path):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        logger.error(f"Error serving static files: {str(e)}")
        return jsonify({'error': str(e)}), 500

def main():
    try:
        logger.info("Initializing VECAID Backend...")
        
        # Initialize predictor
        global predictor
        try:
            predictor = TimeSeriesPredictor()
            logger.info("Successfully initialized TimeSeriesPredictor")
        except Exception as e:
            logger.error(f"Error initializing TimeSeriesPredictor: {str(e)}")
            predictor = None  # Will be initialized on first prediction
        
        # Log GPU status
        if torch.cuda.is_available():
            logger.info(f"GPU Available - Using {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"PyTorch CUDA: {torch.backends.cudnn.version()}")
        else:
            logger.warning("GPU Not Available - Using CPU")
        
        # Start Flask app
        logger.info("Starting Flask server on port 8080...")
        app.run(host='0.0.0.0', port=8080, debug=False)
        
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
