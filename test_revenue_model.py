import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    for col in ['Funding', 'Worldwide Gross', 'Worldwide Ticket Sales']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    genre_cols = [c for c in df.columns if c.startswith('GDC[')]
    for col in genre_cols:
        name = col.split('[')[1].rstrip(']')
        df[f'Genre_{name}'] = df[col]
    df = df.dropna(subset=['Year', 'Funding', 'Worldwide Gross', 'Worldwide Ticket Sales'])
    feature_cols = ['Year', 'Funding'] + [c for c in df.columns if c.startswith('Genre_')]
    target_cols = ['Worldwide Gross', 'Worldwide Ticket Sales']
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train[:, :2] = scaler.fit_transform(X_train[:, :2])
    X_test[:, :2] = scaler.transform(X_test[:, :2])
    return X_train, X_test, y_train, y_test, feature_cols, target_cols, scaler

def load_model(checkpoint_path, input_dim):
    torch.serialization.add_safe_globals([np.dtype])
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    scaler = ckpt.get('scaler')
    feature_cols = ckpt.get('feature_cols')
    target_cols = ckpt.get('target_cols')
    class RevenuePredictor(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, len(target_cols))
            )
        def forward(self, x):
            return self.net(x)
    model = RevenuePredictor(input_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler, feature_cols, target_cols

def main():
    base = Path(__file__).parent
    csv_path = base / 'GDC - HIST.csv'
    ckpt_path = base / 'revenue_model.pt'
    X_train, X_test, y_train, y_test, feat_cols, targ_cols, scaler = load_and_preprocess(csv_path)
    print('Features:', feat_cols)
    print('Targets:', targ_cols)
    print('X_test NaNs:', np.isnan(X_test).sum(), 'y_test NaNs:', np.isnan(y_test).sum())
    model, _, _, _ = load_model(ckpt_path, len(feat_cols))
    X_scaled = X_test.copy()
    X_scaled[:, :2] = scaler.transform(X_scaled[:, :2])
    loader = DataLoader(TensorDataset(torch.from_numpy(X_scaled)), batch_size=32)
    with torch.no_grad():
        batch = next(iter(loader))[0]
        preds = model(batch)
    print('Pred NaNs:', torch.isnan(preds).any().item(),
          'Pred Infs:', torch.isinf(preds).any().item())
    print('Sample preds:', preds[:5].numpy())

print(main())