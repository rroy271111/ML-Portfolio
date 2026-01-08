import pandera as pa
from pandera import Column, DataFrameSchema, Check

raw_transactions_schema = DataFrameSchema(
    {
        "transaction_id": Column(int, Check.ge(0), nullable=False),
        "card_token": Column(object, nullable=False),
        "amount": Column(float, Check.ge(0)),
        "timestamp": Column(pa.DateTime, nullable=False),
        "merchant_id": Column(int, Check.ge(0)),
        "device_id": Column(object, nullable=False),
        "ip": Column(object, nullable=False),
        "is_fraud": Column(int, Check.isin([0, 1]), nullable=True),
    },
    strict=True,
    coerce=True,
)
