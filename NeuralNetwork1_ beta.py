import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 1) Load & clean
from pathlib import Path

csv_path = Path(__file__).parent / "GDC - HIST.csv"
df = pd.read_csv(csv_path)
# Parse numeric columns (remove commas, convert to float)
for col in ['Funding', 'Worldwide Gross', 'Worldwide Ticket Sales']:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '')
        .astype(float)
    )

# 2) Rename genre distribution columns to match our expected format
genre_columns = [
    'GDC[Action/Adventure]', 'GDC[Comedy]', 'GDC[Drama]', 
    'GDC[Horror]', 'GDC[Thriller]', 'GDC[Mystery]', 'GDC[SciFi/Fantasy]'
]

# Rename the genre columns to have 'Genre_' prefix for consistency with the rest of the code
for col in genre_columns:
    # Extract the genre name from the column and create new column with 'Genre_' prefix
    genre_name = col.split('[')[1].split(']')[0]
    df[f'Genre_{genre_name}'] = df[col]

# 3) Drop rows with missing inputs or targets
df = df.dropna(subset=['Year', 'Funding', 'Worldwide Gross', 'Worldwide Ticket Sales'])

# 4) Feature / target split
feature_cols = ['Year', 'Funding'] + [c for c in df.columns if c.startswith('Genre_')]
target_cols  = ['Worldwide Gross', 'Worldwide Ticket Sales']

X = df[feature_cols].values.astype(np.float32)
y = df[target_cols].values.astype(np.float32)

# 5) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6) Scale continuous features (leave one‐hots as-is)
scaler = StandardScaler()
X_train[:, :2] = scaler.fit_transform(X_train[:, :2])   # only Year & Funding
X_test[:, :2]  = scaler.transform(X_test[:, :2])

# 7) PyTorch Dataset & DataLoader
class MovieRevenueDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = MovieRevenueDataset(X_train, y_train)
test_ds  = MovieRevenueDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32)

# 8) Model definition
class RevenuePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)   # two outputs
        )
    def forward(self, x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RevenuePredictor(len(feature_cols)).to(device)

# 9) Loss & optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 10) Training loop
num_epochs = 30
for epoch in range(1, num_epochs+1):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    train_loss = total_loss / len(train_loader.dataset)

    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item() * X_batch.size(0)
        val_loss /= len(test_loader.dataset)
        print(f"Epoch {epoch:02d} → train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

# 11) Save model + metadata


# …existing code…

# Remove the old torch.save(...) block here

if __name__ == "__main__":
    from pathlib import Path

    # Define save path
    save_path = Path(__file__).parent / "revenue_model.pt"

    # Build a checkpoint dict, not just state_dict
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target_cols": target_cols
    }, save_path)

    print(f"Saved full checkpoint (model+scaler+meta) to {save_path}")
# …existing code…