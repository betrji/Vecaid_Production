
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# === GPU CHECK ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# === Data Utilities ===
def load_stock_data(ticker, window=60):
    df = yf.download(ticker, period="1y")
    close = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i])
        y.append(scaled[i])
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

# === PyTorch GRU Model ===
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# === Train GRU Model ===
def train_gru_model(X, y, epochs=20, lr=0.001):
    model = GRUModel(input_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X, y = X.to(device), y.to(device)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    return model

# === Predict with GRU ===
def predict_next_day_close_pytorch(model, X, scaler):
    model.eval()
    with torch.no_grad():
        input_seq = X[-1].unsqueeze(0).to(device)
        prediction = model(input_seq).cpu().numpy()
    return scaler.inverse_transform(prediction.reshape(-1, 1))[0, 0]

# === XGBoost Model ===
def train_xgb_model(X, y):
    model = XGBRegressor(n_estimators=100)
    model.fit(X.cpu().numpy().reshape(X.shape[0], -1), y.cpu().numpy())
    return model

# === Main Prediction Function ===
def predict(ticker="AAPL", window=60):
    X, y, scaler = load_stock_data(ticker, window)
    print(f"Loaded data for {ticker}: {X.shape} samples")
    
    # Train GRU model
    gru_model = train_gru_model(X, y)
    gru_prediction = predict_next_day_close_pytorch(gru_model, X, scaler)
    print(f"GRU Prediction: {gru_prediction}")

    # Train XGBoost model
    xgb_model = train_xgb_model(X, y)
    xgb_pred = xgb_model.predict(X[-1].cpu().numpy().reshape(1, -1))
    xgb_pred_unscaled = scaler.inverse_transform(xgb_pred.reshape(-1, 1))[0, 0]
    print(f"XGBoost Prediction: {xgb_pred_unscaled}")

    return {
        "ticker": ticker,
        "gru_prediction": float(gru_prediction),
        "xgb_prediction": float(xgb_pred_unscaled)
    }

if __name__ == "__main__":
    result = predict("NVDA")
    print(result)
