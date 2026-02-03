import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. Load the dataset
df = pd.read_csv('gasoline_prices.csv')
if df.shape[1] > 1:
    values = df.iloc[:, 1].values
else:
    values = df.iloc[:, 0].values
data = values.astype(np.float32)

# 2. Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# 3. Split into training/testing sets
N = len(data_scaled)
train_size = int(N * 0.9)
test_size = N - train_size
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# 4. Define LSTM model with improved architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Add bidirectional LSTM for better pattern recognition
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=True
        )
        
        # Double the hidden size for bidirectional
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
            
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# 5. Create training sequences with shorter lookback
seq_len = 52  # Approximately one year of weekly data
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_data, seq_len)
X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, seq_len, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

# 6. Create validation set
val_size = int(len(X_train) * 0.1)
X_val = X_train[-val_size:]
y_val = y_train[-val_size:]
X_train = X_train[:-val_size]
y_train = y_train[:-val_size]

# 7. Initialize model with better parameters
model = LSTMModel(input_size=1, hidden_size=128, num_layers=2, dropout=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
loss_fn = nn.MSELoss()

# 8. Train the model with early stopping
n_epochs = 150
batch_size = 32
best_val_loss = float('inf')
patience = 10
patience_counter = 0

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("\nTraining model...")
for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = loss_fn(val_output, y_val).item()
    
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{n_epochs}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# 9. Predictions using window-based approach instead of autoregressive
# This reduces error accumulation
model.eval()
test_inputs = []
test_targets = []

for i in range(test_size - seq_len):
    # Create test windows that include both training and test data
    test_seq = data_scaled[train_size + i - seq_len:train_size + i]
    test_inputs.append(test_seq)
    test_targets.append(test_data[i])

test_inputs = torch.tensor(test_inputs, dtype=torch.float32).reshape(-1, seq_len, 1)
predictions = []

with torch.no_grad():
    batch_size = 64  # Process test data in batches for efficiency
    for i in range(0, len(test_inputs), batch_size):
        batch_input = test_inputs[i:i+batch_size]
        batch_output = model(batch_input).flatten().numpy()
        predictions.extend(batch_output)

# 10. Inverse scale predictions and actuals
predictions = np.array(predictions).reshape(-1, 1)
targets = np.array(test_targets).reshape(-1, 1)

predictions_inv = scaler.inverse_transform(predictions)
targets_inv = scaler.inverse_transform(targets)

# 11. Multi-step forecasting for visualization (predicting 4 weeks ahead)
# Here we'll use the true input data to predict 1-4 steps ahead
multi_step_preds = {1: [], 2: [], 3: [], 4: []}
for i in range(len(test_data) - seq_len - 4):
    input_seq = data_scaled[train_size + i - seq_len:train_size + i]
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).reshape(1, seq_len, 1)
    with torch.no_grad():
        # Step 1 forecast
        step1 = model(input_tensor).item()
        multi_step_preds[1].append(step1)
        
        # Step 2 forecast
        step2_input = np.append(input_seq[1:], step1)
        step2_tensor = torch.tensor(step2_input, dtype=torch.float32).reshape(1, seq_len, 1)
        step2 = model(step2_tensor).item()
        multi_step_preds[2].append(step2)
        
        # Step 3 forecast
        step3_input = np.append(input_seq[2:], [step1, step2])
        step3_tensor = torch.tensor(step3_input, dtype=torch.float32).reshape(1, seq_len, 1)
        step3 = model(step3_tensor).item()
        multi_step_preds[3].append(step3)
        
        # Step 4 forecast
        step4_input = np.append(input_seq[3:], [step1, step2, step3])
        step4_tensor = torch.tensor(step4_input, dtype=torch.float32).reshape(1, seq_len, 1)
        step4 = model(step4_tensor).item()
        multi_step_preds[4].append(step4)

# Convert to original scale
for step in range(1, 5):
    multi_step_preds[step] = scaler.inverse_transform(
        np.array(multi_step_preds[step]).reshape(-1, 1)).flatten()

# Get actual values for the period
actual_for_multi = test_data[seq_len:seq_len+len(multi_step_preds[1])]
actual_for_multi_inv = scaler.inverse_transform(
    actual_for_multi.reshape(-1, 1)).flatten()

# 12. Calculate metrics for step 1 predictions
mae = np.mean(np.abs(predictions_inv.flatten()[:len(targets_inv)] - targets_inv.flatten()))
mse = np.mean((predictions_inv.flatten()[:len(targets_inv)] - targets_inv.flatten()) ** 2)
rmse = np.sqrt(mse)
print(f"\nMAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# 13. Plot both single-step and multi-step forecasts
plt.figure(figsize=(15, 7))

# Single-step forecast
plt.subplot(1, 2, 1)
x_single = range(len(predictions_inv))
plt.plot(x_single, targets_inv, 'k--', label='Actual')
plt.plot(x_single, predictions_inv, 'b-', label='1-Step Prediction')
plt.xlabel('Test Sample')
plt.ylabel('Price ($)')
plt.title('1-Step Gasoline Price Forecast')
plt.legend()

# Multi-step forecast
plt.subplot(1, 2, 2)
x_multi = range(len(multi_step_preds[1]))
plt.plot(x_multi, actual_for_multi_inv, 'k--', label='Actual')
for step in range(1, 5):
    plt.plot(x_multi, multi_step_preds[step], label=f'Step {step}')
plt.xlabel('Test Sample')
plt.ylabel('Price ($)')
plt.title('4-Week Gas Price Forecast vs Actual')
plt.legend()

plt.tight_layout()
plt.savefig("lstm_forecast.png", dpi=300, bbox_inches='tight')
plt.show()