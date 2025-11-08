import pandas as pd
import numpy as np
import os

def generate_synthetic_transactions(num_rows=10000, output_path=None, random_state=42):
    """
    Generate a synthetic credit card transaction dataset.

    Columns:
    ['transaction_id', 'card_token', 'amount', 'timestamp', 'merchant_id', 'device_id', 'ip', 'label']
    """

    np.random.seed(random_state)

    df = pd.DataFrame({
        'transaction_id': np.arange(1, num_rows + 1),
        'card_token': np.random.randint(100000, 999999, size=num_rows),
        'amount': np.round(np.random.exponential(scale=100, size=num_rows), 2),
        'timestamp': pd.to_datetime(
            np.random.randint(
                pd.Timestamp("2025-01-01").value //10**9,
                pd.Timestamp("2025-10-14").value // 10**9,
                size=num_rows
            ), unit='s'
        ),
        'merchant_id': np.random.randint(1,500,size=num_rows),
        'device_id': np.random.randint(1000,9999,size=num_rows),
        'ip': [f"192.168.{np.random.randint(0,256)}.{np.random.randint(0,256)}" for __ in range(num_rows)],
        #label: 0 = normal, 1 = fraud
        'label': np.random.choice([0,1], size=num_rows, p=[0.98, 0.02])
    }
    )

    # Add slight noise to avoid identical amounts
    df["amount"] += np.random.normal(0, 0.5, size=num_rows)

    # Ensure chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Cast ID-like fields to categorical (optimizes memory + aligns with feature builder)   
    df["card_token"] = df["card_token"].astype("category")
    df["device_id"] = df["device_id"].astype("category")
    df["ip"] = df["ip"].astype("category")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Synthetic data saved to {output_path}")
    return df
