import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Configuration
SEQ_LEN = 12
AA_DIM = 20
INPUT_DIM = SEQ_LEN * AA_DIM

# ---------------------------
# Model 1: Regression Network
# ---------------------------
class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Model 2: RNN Classification
# ---------------------------
class RNNClassifier(nn.Module):
    def __init__(self, input_size=AA_DIM, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, seq, reg_output):
        lstm_out, _ = self.lstm(seq)
        lstm_final = lstm_out[:, -1, :]
        combined = torch.cat((lstm_final, reg_output), dim=1)
        return self.classifier(combined)

# ---------------------------
# Data preparation
# ---------------------------
def prepare_tensor_data(X, y_reg=None, y_cls=None):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_seq = X_tensor.view(-1, SEQ_LEN, AA_DIM)  # reshape for RNN
    data = {"X_seq": X_seq}
    if y_reg is not None:
        data["y_reg"] = torch.tensor(y_reg, dtype=torch.float32).unsqueeze(1)
    if y_cls is not None:
        data["y_cls"] = torch.tensor(y_cls, dtype=torch.float32).unsqueeze(1)
    return data

# Example train split
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, df['IC50'].values, test_size=0.2)
train_data = prepare_tensor_data(X_train, y_reg=y_reg_train)
test_data = prepare_tensor_data(X_test, y_reg=y_reg_test)

# ---------------------------
# Train Model 1: Regression
# ---------------------------
model_reg = RegressionModel()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model_reg.parameters(), lr=1e-3)

for epoch in range(50):
    model_reg.train()
    preds = model_reg(train_data['X_seq'].view(-1, INPUT_DIM))
    loss = loss_fn(preds, train_data['y_reg'])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# ---------------------------
# Train Model 2: RNN Classifier
# ---------------------------
# Transfer learned output from Model 1
reg_output = model_reg(train_data['X_seq'].view(-1, INPUT_DIM)).detach()

# Use your binary antimicrobial labels for classification
# y_cls = df['label'].values  # 1 for active, 0 for inactive

# For illustration, we'll fake labels:
y_cls = (y_reg_train < 1.0).astype(int)  # just for testing
train_data_cls = prepare_tensor_data(X_train, y_reg=reg_output, y_cls=y_cls)

model_cls = RNNClassifier()
loss_fn_cls = nn.BCELoss()
optimizer_cls = optim.Adam(model_cls.parameters(), lr=1e-3)

for epoch in range(50):
    model_cls.train()
    output = model_cls(train_data_cls['X_seq'], train_data_cls['y_reg'])
    loss = loss_fn_cls(output, train_data_cls['y_cls'])
    loss.backward()
    optimizer_cls.step()
    optimizer_cls.zero_grad()
