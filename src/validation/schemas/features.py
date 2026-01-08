import pandera as pa
from pandera import Column, DataFrameSchema, Check

feature_schema = DataFrameSchema(
    {
        "amount_log": Column(float, Check.ge(0)),
        "hour": Column(int, Check.between(0, 23)),
        "dow": Column(int, Check.between(0, 6)),
        "merchant_id": Column(int),
        "card_token": Column(int),
        "device_id": Column(int),
        "ip": Column(int),
        "is_large_amount": Column(int, Check.isin([0, 1])),
    },
    strict=True,
    coerce=True,
)
