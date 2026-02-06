import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib


class StockDataset(Dataset):
    def __init__(self, csv_file, seq_len = 60, mode = 'train'):
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])

        exclude_cols = ['Date', 'Ticker', 'Target_Return_20d']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        target_col = 'Target_Return_20d'
        df.sort_values(['Ticker', 'Date'], inplace = True)
        self.scaler = StandardScaler()

        df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])
        if mode == 'train':
            joblib.dump(self.scaler, 'scaler.gz')

        self.samples = []
        tickers = df['Ticker'].unique()
        for ticker in tickers:
            group = df[df['Ticker'] == ticker].copy()
            n = len(group)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            if mode == 'train':
                data_subset = group.iloc[:train_end]
            elif mode == 'val':
                data_subset = group.iloc[train_end:val_end]
            elif mode == 'test':
                data_subset = group.iloc[val_end:]
            else:
                raise ValueError("Mode must be 'train', 'val', or 'test'")

            features = data_subset[self.feature_cols].values
            targets = data_subset[target_col].values
            if len(features) < seq_len:
                continue
            for i in range(len(features) - seq_len):
                x = features[i: i + seq_len]
                y = targets[i + seq_len]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype = torch.float32), torch.tensor(y, dtype = torch.float32)


if __name__ == "__main__":
    try:
        dataset = StockDataset("final_training_data.csv", seq_len = 60, mode = 'train')
        loader = DataLoader(dataset, batch_size = 32, shuffle = True)
        data_iter = iter(loader)
        x_batch, y_batch = next(data_iter)

        print("-" * 30)
        print(f"Total Samples: {len(dataset)}")
        print(f"Input Batch Shape: {x_batch.shape}")
        print(f"Target Batch Shape: {y_batch.shape}")
        print("-" * 30)
    except FileNotFoundError:
        print("Error: 'final_training_data.csv' not found. Run the features script first.")
    except Exception as e:
        print(f"An error occurred: {e}")