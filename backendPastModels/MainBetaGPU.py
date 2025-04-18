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
from flask import Flask, request, jsonify, send_from_directory, session as flask_session, make_response, Response
from flask_cors import CORS, cross_origin
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
import stripe
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import pickle
from sklearn.metrics import mean_squared_error

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize global variables and configuration
DEVELOPER_KEY = os.getenv('DEVELOPER_KEY')
if not DEVELOPER_KEY:
    logger.error("DEVELOPER_KEY not found in environment variables")
    raise ValueError("DEVELOPER_KEY is required")

JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
if not JWT_SECRET_KEY:
    logger.error("JWT_SECRET_KEY not found in environment variables")
    raise ValueError("JWT_SECRET_KEY is required")

STRIPE_API_KEY = os.getenv('STRIPE_API_KEY')
if not STRIPE_API_KEY:
    logger.error("STRIPE_API_KEY not found in environment variables")
    raise ValueError("STRIPE_API_KEY is required")

STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')
if not STRIPE_WEBHOOK_SECRET:
    logger.error("STRIPE_WEBHOOK_SECRET not found in environment variables")
    raise ValueError("STRIPE_WEBHOOK_SECRET is required")

STRIPE_PRICE_ID = os.getenv('STRIPE_PRICE_ID')
if not STRIPE_PRICE_ID:
    logger.error("STRIPE_PRICE_ID not found in environment variables")
    raise ValueError("STRIPE_PRICE_ID is required")

try:
    ALLOWED_ORIGINS = json.loads(os.getenv('ALLOWED_ORIGINS', '[]'))
    if not ALLOWED_ORIGINS:
        logger.warning("ALLOWED_ORIGINS not found or empty, defaulting to localhost origins")
        ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
except json.JSONDecodeError:
    logger.error("Invalid JSON format in ALLOWED_ORIGINS environment variable")
    raise ValueError("ALLOWED_ORIGINS must be a valid JSON array")

# Model configuration
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', 'models')
SCALER_SAVE_PATH = os.getenv('SCALER_SAVE_PATH', 'scalers')
SEQUENCE_LENGTH = int(os.getenv('SEQUENCE_LENGTH', '60'))
PREDICTION_HORIZON = int(os.getenv('PREDICTION_HORIZON', '5'))

# Initialize other global variables
active_predictions = {}
prediction_lock = threading.Lock()
training_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=4)
predictor = None

# Mock user database (replace with real database in production)
users_db = {
    "developer@vecaid.ai": {
        "password": generate_password_hash("devpass"),
        "is_subscribed": True,
        "stripe_customer_id": None,
        "stripe_subscription_id": None
    }
}

# Check for GPU availability and set up device
cuda_info = {
    'cuda_available': torch.cuda.is_available(),
    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
}

if cuda_info['cuda_available']:
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info(f"=== CUDA Information ===")
    logger.info(f"CUDA available: {cuda_info['cuda_available']}")
    logger.info(f"Using GPU: {cuda_info['gpu_name']}")
    logger.info(f"CUDA Version: {cuda_info['cuda_version']}")
    logger.info(f"GPU Count: {cuda_info['gpu_count']}")
    logger.info(f"Current Device: {cuda_info['current_device']}")
else:
    device = torch.device('cpu')
    logger.info("Using CPU - CUDA not available")

# Initialize Flask app
app = Flask(__name__, static_folder='../MainFrontend/dist')
app.secret_key = JWT_SECRET_KEY

# Configure Stripe
stripe.api_key = STRIPE_API_KEY

# Configure CORS with proper resource configuration
CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type", "Authorization"],
        "max_age": 3600
    }
})

# Configure session and cookie settings
app.config.update(
    SESSION_COOKIE_SECURE=os.getenv('SESSION_COOKIE_SECURE', 'false').lower() == 'true',
    SESSION_COOKIE_HTTPONLY=os.getenv('SESSION_COOKIE_HTTPONLY', 'true').lower() == 'true',
    SESSION_COOKIE_SAMESITE=os.getenv('SESSION_COOKIE_SAMESITE', 'Lax'),
    SESSION_COOKIE_DOMAIN=os.getenv('SESSION_COOKIE_DOMAIN', 'localhost')
)

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin in ALLOWED_ORIGINS:
        response.headers.update({
            'Access-Control-Allow-Origin': origin,
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
            'Access-Control-Allow-Credentials': 'true',
            'Access-Control-Max-Age': '3600',
            'Access-Control-Expose-Headers': 'Content-Type, Authorization'
        })
    return response

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
        self.models: Dict[str, Any] = {}  # Store both PyTorch and XGBoost models
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.sequence_length = SEQUENCE_LENGTH
        self.prediction_horizon = PREDICTION_HORIZON
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        self.patience = 10
        
        # Ensure GPU is properly configured
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(f"TimeSeriesPredictor initialized with GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("TimeSeriesPredictor running on CPU")
        
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(scaler_path, exist_ok=True)

    def get_stock_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Fetch real-time and historical stock data using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval="1d")
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data and create features"""
        # Technical indicators
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['MACD'], data['Signal'] = self.calculate_macd(data['Close'])
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = self.calculate_bollinger_bands(data['Close'])
        
        # Additional features
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        # Drop NaN values
        data = data.dropna()
        
        # Create feature matrix
        features = np.column_stack((
            data['Close'].values,
            data['Volume'].values,
            data['RSI'].values,
            data['MACD'].values,
            data['BB_upper'].values,
            data['BB_lower'].values,
            data['Returns'].values,
            data['Volatility'].values,
            data['Volume_MA'].values
        ))
        
        return features, data['Close'].values

    def train_model(self, ticker: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Train both LSTM and XGBoost models"""
        start_time = time.time()
        features, targets = self.preprocess_data(data)
        
        # Scale the data
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        self.scalers[ticker] = scaler
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, self.sequence_length, self.prediction_horizon)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Train LSTM model
        lstm_model, best_val_loss = self.train_lstm(X_train, y_train, X_val, y_val)
        
        # Train XGBoost model
        xgb_model = self.train_xgboost(X_train.reshape(X_train.shape[0], -1), y_train,
                                      X_val.reshape(X_val.shape[0], -1), y_val)
        
        # Save models
        self.models[ticker] = {
            'lstm': lstm_model,
            'xgb': xgb_model,
            'scaler': scaler
        }
        
        return {
            'training_time': time.time() - start_time,
            'final_loss': float(best_val_loss)
        }

    def train_lstm(self, X_train, y_train, X_val, y_val) -> Tuple[nn.Module, float]:
        """Train LSTM model with early stopping"""
        model = LSTMModel(
            input_size=X_train.shape[2],
            hidden_size=128,
            num_layers=2,
            output_size=self.prediction_horizon
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    val_loss += criterion(outputs, y_batch).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"{self.model_path}/{ticker}_lstm.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        return model, best_val_loss

    def train_xgboost(self, X_train, y_train, X_val, y_val) -> Any:
        """Train XGBoost model"""
        import xgboost as xgb
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        return model

    def backtest(self, ticker: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform backtesting on historical data"""
        features, targets = self.preprocess_data(data)
        features_scaled = self.scalers[ticker].transform(features)
        
        predictions = []
        actuals = []
        
        # Create sequences for backtesting
        for i in range(len(features_scaled) - self.sequence_length - self.prediction_horizon):
            X = features_scaled[i:i+self.sequence_length].reshape(1, self.sequence_length, -1)
            y_true = targets[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon]
            
            # Get predictions from both models
            lstm_pred = self.models[ticker]['lstm'](torch.FloatTensor(X).to(device)).cpu().detach().numpy()
            xgb_pred = self.models[ticker]['xgb'].predict(
                xgb.DMatrix(X.reshape(1, -1))
            ).reshape(-1, self.prediction_horizon)
            
            # Ensemble predictions (simple average)
            y_pred = (lstm_pred + xgb_pred) / 2
            
            predictions.append(y_pred[0])
            actuals.append(y_true)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actuals - predictions))
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(predictions[1:] - predictions[:-1]) == 
                                 np.sign(actuals[1:] - actuals[:-1]))
        accuracy = direction_correct / (len(predictions) - 1)
        
        return {
            'statistics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'accuracy_percentage': accuracy * 100
            },
            'predictions': predictions.tolist(),
            'actuals': actuals.tolist()
        }

    def create_visualization(self, predictions: np.ndarray, actuals: np.ndarray, ticker: str) -> str:
        """Create visualization of predictions vs actuals"""
        try:
            import matplotlib.pyplot as plt
            import io
            import base64

            plt.figure(figsize=(12, 6))
            plt.plot(actuals, label='Actual Price', color='blue', alpha=0.7)
            plt.plot(predictions, label='Predicted Price', color='red', alpha=0.7)
            plt.title(f'{ticker} - Price Prediction vs Actual')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return ""

    def predict(self, ticker: str) -> Dict[str, Any]:
        """Make predictions using both models and return comprehensive results"""
        try:
            # Get real-time data
            data = self.get_stock_data(ticker)
            
            # Train models if not already trained
            if ticker not in self.models:
                training_info = self.train_model(ticker, data)
            else:
                training_info = {'training_time': 0, 'final_loss': 0}
            
            # Calculate technical indicators
            technical_indicators = self.calculate_technical_indicators(data)
            
            # Prepare latest data for prediction
            features, _ = self.preprocess_data(data)
            features_scaled = self.scalers[ticker].transform(features)
            X = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            X_tensor = torch.FloatTensor(X).to(device)
            
            # Get predictions from both models
            self.models[ticker]['lstm'].eval()  # Set model to evaluation mode
            with torch.no_grad():
                lstm_pred = self.models[ticker]['lstm'](X_tensor).cpu().numpy()
            
            xgb_pred = self.models[ticker]['xgb'].predict(
                xgb.DMatrix(X.reshape(1, -1))
            ).reshape(-1, self.prediction_horizon)
            
            # Ensemble predictions
            final_prediction = (lstm_pred + xgb_pred) / 2
            
            # Perform backtesting
            backtest_results = self.backtest(ticker, data)
            
            # Create visualization
            viz_data = self.create_visualization(
                backtest_results['predictions'],
                backtest_results['actuals'],
                ticker
            )
            
            # Determine prediction direction and confidence
            current_price = data['Close'].iloc[-1]
            predicted_price = final_prediction[0][0]
            price_change = predicted_price - current_price
            direction = 'up' if price_change > 0 else 'down'
            confidence = backtest_results['statistics']['accuracy_percentage'] / 100
            
            return {
                'ticker': ticker,
                'training_info': training_info,
                'ml_prediction': {
                    'direction': direction,
                    'confidence': confidence,
                    'target_price': float(predicted_price),
                    'predicted_move': float(price_change / current_price),
                    'predicted_date': (datetime.now() + relativedelta(days=1)).isoformat()
                },
                'backtest_results': backtest_results,
                'technical_indicators': technical_indicators,
                'visualization': viz_data
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {ticker}: {str(e)}")
            raise

    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for the latest data point"""
        try:
            rsi = self.calculate_rsi(data['Close'])[-1]
            macd, signal = self.calculate_macd(data['Close'])
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
            
            return {
                'rsi': float(rsi),
                'macd': float(macd[-1]),
                'signal': float(signal[-1]),
                'bollinger_bands': {
                    'upper': float(bb_upper[-1]),
                    'middle': float(bb_middle[-1]),
                    'lower': float(bb_lower[-1])
                }
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD and Signal line"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd.values, signal_line.values

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        middle_band = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        return upper_band.values, middle_band.values, lower_band.values

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
            logger.info(f"Downloading historical data for {self.ticker}")
            stock = yf.Ticker(self.ticker)
            hist = stock.history(period="2y")  # Extended period for better training
            
            if hist.empty:
                self.result_queue.put(("error", f"No data found for ticker {self.ticker}"))
                return
            
            if self.cancelled:
                return
            
            # Initialize predictor if needed
            global predictor
            if predictor is None:
                predictor = TimeSeriesPredictor()
                if torch.cuda.is_available():
                    logger.info(f"Initialized predictor with GPU: {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning("Initialized predictor with CPU")
            
            try:
                # Initialize progress
                self.result_queue.put(("progress", {
                    "stage": "initializing",
                    "message": f"Analyzing {self.ticker}",
                    "progress": 0
                }))

                # Check if we need to train a new model
                needs_training = self.ticker not in predictor.models or self.ticker not in predictor.scalers
                training_info = None
                
                if needs_training:
                    self.result_queue.put(("progress", {
                        "stage": "training_start",
                        "message": f"Starting model training for {self.ticker} using {'GPU' if torch.cuda.is_available() else 'CPU'}",
                        "progress": 10
                    }))
                    
                    # Train the model
                    training_info = predictor.train_model(self.ticker, hist)
                    
                    self.result_queue.put(("progress", {
                        "stage": "training_complete",
                        "message": f"Model training completed in {training_info['training_time']:.2f} seconds",
                        "progress": 50
                    }))
                
                # Start prediction process
                self.result_queue.put(("progress", {
                    "stage": "prediction",
                    "message": "Running predictions and backtesting...",
                    "progress": 60
                }))
                
                # Get prediction results
                prediction_results = predictor.predict(self.ticker)
                
                if not self.cancelled:
                    # Send completion message
                    self.result_queue.put(("progress", {
                        "stage": "complete",
                        "message": "Analysis completed successfully",
                        "progress": 100
                    }))
                    
                    # Ensure all required fields are present
                    final_result = {
                        "ticker": self.ticker,
                        "training_info": {
                            "model_type": "Ensemble (LSTM + XGBoost)",
                            "training_time": float(training_info['training_time'] if training_info else 0),
                            "final_loss": float(training_info['final_loss'] if training_info else 0),
                            "gpu_used": torch.cuda.is_available()
                        },
                        "ml_prediction": prediction_results["ml_prediction"],
                        "backtest_results": prediction_results["backtest_results"],
                        "technical_indicators": prediction_results["technical_indicators"],
                        "visualization": prediction_results["visualization"]
                    }
                    
                    # Send final result
                    self.result_queue.put(("success", final_result))
                
            except Exception as e:
                logger.error(f"Error in model operations: {str(e)}")
                self.result_queue.put(("error", f"Failed to process prediction: {str(e)}"))
            
        except Exception as e:
            logger.error(f"Error in prediction task: {str(e)}")
            self.result_queue.put(("error", str(e)))
        finally:
            with prediction_lock:
                if self.ticker in active_predictions:
                    del active_predictions[self.ticker]
                    
            # Clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            response = make_response()
            response.headers.update({
                'Access-Control-Allow-Origin': request.headers.get('Origin', 'http://localhost:5173'),
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                'Access-Control-Allow-Credentials': 'true',
                'Access-Control-Max-Age': '3600'
            })
            return response

        data = request.get_json()
        logger.info("Login attempt received")
        logger.debug(f"Login data: {data}")  # Add debug logging
        
        if not data:
            logger.warning("No data received in login request")
            return jsonify({"error": "No data provided"}), 400
            
        if 'email' not in data or 'password' not in data:
            logger.warning("Missing email or password in login request")
            return jsonify({"error": "Email and password are required"}), 400
            
        email = data['email']
        password = data['password']

        # Add developer user if it doesn't exist
        if "developer@vecaid.ai" not in users_db:
            users_db["developer@vecaid.ai"] = {
                "password": generate_password_hash("devpass"),
                "is_subscribed": True,
                "stripe_customer_id": None,
                "stripe_subscription_id": None
            }

        if email not in users_db:
            logger.warning(f"Login attempt with non-existent email: {email}")
            return jsonify({"error": "Invalid email or password"}), 401

        user = users_db[email]
        if not check_password_hash(user["password"], password):
            logger.warning(f"Invalid password attempt for email: {email}")
            return jsonify({"error": "Invalid email or password"}), 401

        # Generate token
        token = jwt.encode(
            {
                "email": email,
                "is_subscribed": user.get("is_subscribed", False),
                "is_developer": email == "developer@vecaid.ai",
                "exp": datetime.utcnow() + relativedelta(days=30)
            },
            app.secret_key,
            algorithm="HS256"
        )
        
        # Create response
        response = make_response(jsonify({
            "token": token,
            "message": "Login successful",
            "user": {
                "email": email,
                "isSubscribed": user.get("is_subscribed", False),
                "isDeveloper": email == "developer@vecaid.ai"
            }
        }))
        
        # Set cookie
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
        
        logger.info(f"Login successful for user: {email}")
        return response
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({"error": "An error occurred during login"}), 500

@app.route("/api/auth/developer", methods=["POST", "OPTIONS"])
@cross_origin(supports_credentials=True)
def developer_auth():
    try:
        if request.method == "OPTIONS":
            return make_response()

        data = request.get_json()
        if not data or 'key' not in data:
            return jsonify({"error": "Developer key is required"}), 400
        
        submitted_key = data['key']
        if submitted_key != DEVELOPER_KEY:
            return jsonify({"error": "Invalid developer key"}), 401
            
        # Generate a special developer token with direct access privileges
        token = jwt.encode(
            {
                "email": "developer@vecaid.ai",
                "is_subscribed": True,
                "is_developer": True,
                "direct_access": True,  # Special flag for direct prediction access
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
                "isDeveloper": True,
                "directAccess": True
            },
            "redirect": "/prediction"  # Tell frontend to redirect directly to prediction page
        }))
        
        # Set cookie with the token
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
                    "isDeveloper": payload["is_developer"],
                    "directAccess": payload.get("direct_access", False)
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
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': 'VECAID Premium Subscription',
                        'description': 'Unlimited AI-Powered Stock Predictions'
                    },
                    'unit_amount': 999,  # $9.99 in cents
                    'recurring': {
                        'interval': 'month'
                    }
                },
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f'http://localhost:5173/subscription?session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'http://localhost:5173/subscription',
            customer_email=email,
        )

        return jsonify({'url': session.url, 'sessionId': session.id})
    except Exception as e:
        logger.error(f"Error creating checkout session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    event = None
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        logger.error(f"Invalid payload: {str(e)}")
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {str(e)}")
        return jsonify({'error': 'Invalid signature'}), 400

    # Handle the checkout.session.completed event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        customer_email = session.get('customer_email')
        
        if customer_email in users_db:
            users_db[customer_email].update({
                'is_subscribed': True,
                'stripe_customer_id': session.get('customer'),
                'stripe_subscription_id': session.get('subscription')
            })
            logger.info(f"Successfully updated subscription status for {customer_email}")
        else:
            logger.warning(f"User {customer_email} not found in database")

    return jsonify({'status': 'success'})

@app.route('/api/subscription/success', methods=['GET'])
@cross_origin(supports_credentials=True)
def subscription_success():
    try:
        session_id = request.args.get("session_id")
        if not session_id:
            return jsonify({"error": "No session ID provided"}), 400
            
        # Retrieve checkout session
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        subscription = stripe.Subscription.retrieve(checkout_session.subscription)
        
        # Update user subscription status
        customer_email = checkout_session.customer_email
        if customer_email in users_db:
            users_db[customer_email].update({
                'is_subscribed': True,
                'stripe_customer_id': checkout_session.customer,
                'stripe_subscription_id': subscription.id
            })
            
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

@app.route('/api/gpu/status', methods=['GET'])
@cross_origin(supports_credentials=True)
def gpu_status():
    """Get current GPU/CUDA status"""
    try:
        memory_info = {}
        if cuda_info['cuda_available']:
            memory_info = {
                'allocated': torch.cuda.memory_allocated(0) / 1024**2,  # MB
                'cached': torch.cuda.memory_reserved(0) / 1024**2,  # MB
                'max': torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            }
        
        status = {
            'cuda_info': cuda_info,
            'memory_info': memory_info,
            'device': str(device),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Error getting GPU status: {str(e)}")
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
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
        
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
