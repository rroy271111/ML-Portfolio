import pandas as pd
import numpy as np

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """ Return a DataFrame with cleaned + derived features.
    Expected columns: ['transaction_id', 'card_token', 'amount', 'timestamp', 'merchant_id', 'device_id', 'ip']
    """

    df = df_raw.copy()
    # Time features
    df['ts'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['ts'].dt.hour
    df['dow'] = df['ts'].dt.dayofweek


    # amount features
    df['amount_log'] = np.log1p(df['amount'])
    df['is_large_amount'] = (df['amount'] > df['amount'].quantile(0.99)).astype(int)

    # Aggregate placeholders (compute offline or via streaming)
    # df['card_tx_24h'] = ...

    # Encode categorical 
    df["card_token"] = df["card_token"].astype(str).astype("category").cat.codes
    df["device_id"] = df["device_id"].astype(str).astype("category").cat.codes
    df["ip"] = df["ip"].astype(str).astype("category").cat.codes
    df['merchant_id'] = df['merchant_id'].astype(int)

    # Select features
    feat_cols = ['amount_log', 'hour', 'dow', 'merchant_id','card_token','device_id','ip', 'is_large_amount']

    return df[feat_cols]


