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
    df['merchant_id'] = df['merchant_id'].astype(int)

    # Select features
    feat_cols = ['amount_log', 'hour', 'dow', 'merchant_id']

    return df[feat_cols]


