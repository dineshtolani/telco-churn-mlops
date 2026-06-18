from feast import Field, FeatureView, FileSource
from feast.types import Float32, Int32, String

customer_source = FileSource(
    path="data/raw/customers_jan.csv",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

customer_features = FeatureView(
    name="customer_features",
    entities=["customer_id"],
    ttl=None,
    schema=[
        Field(name="gender", dtype=String),
        Field(name="SeniorCitizen", dtype=Int32),
        Field(name="Partner", dtype=String),
        Field(name="Dependents", dtype=String),
        Field(name="tenure", dtype=Int32),
        Field(name="PhoneService", dtype=String),
        Field(name="MultipleLines", dtype=String),
        Field(name="InternetService", dtype=String),
        Field(name="OnlineSecurity", dtype=String),
        Field(name="OnlineBackup", dtype=String),
        Field(name="DeviceProtection", dtype=String),
        Field(name="TechSupport", dtype=String),
        Field(name="StreamingTV", dtype=String),
        Field(name="StreamingMovies", dtype=String),
        Field(name="Contract", dtype=String),
        Field(name="PaperlessBilling", dtype=String),
        Field(name="PaymentMethod", dtype=String),
        Field(name="MonthlyCharges", dtype=Float32),
        Field(name="TotalCharges", dtype=Float32),
        Field(name="Churn", dtype=String),
    ],
    source=customer_source,
)
