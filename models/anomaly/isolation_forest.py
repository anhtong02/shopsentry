from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("anomaly_detection")


df = pd.read_parquet("feature_repo/feature_repo/data/offline_features")

feature_cols = [
    "events_per_minute",
    "unique_pages_visited",
    "avg_time_between_events",
    "cart_to_purchase_ratio",
    "session_duration_seconds",
    "event_type_diversity",
    "has_payment",
    "signup_to_purchase_speed",
    "page_revisit_ratio"
]
df["label"] = df["agent_type"].apply(lambda x: 1 if x in ["bot", "fraud"] else 0)

X = df[feature_cols]

iso_forest = IsolationForest(contamination=0.2, random_state=53)


iso_forest.fit(X)

df["prediction"] = iso_forest.predict(X)
df["prediction"] = df["prediction"].apply(lambda x: 1 if x==-1 else 0)

# print(confusion_matrix(df["label"], df["prediction"]))
# print(classification_report(df["label"], df["prediction"]))

report_dict = classification_report(df["label"], df["prediction"], output_dict=True)

bots = df[(df["agent_type"] == "bot")]
print(f"Total bots: {len(bots)}")
print(f"Caught: {(bots['prediction'] == 1).sum()}")
print(f"Missed: {(bots['prediction'] == 0).sum()}")

fraud = df[(df["agent_type"] == "fraud")]
print(f"Total fraud: {len(fraud)}")
print(f"Caught: {(fraud['prediction'] == 1).sum()}")
print(f"Missed: {(fraud['prediction'] == 0).sum()}")

with mlflow.start_run(run_name="isolation_forest_baseline"):
    mlflow.log_param("model", "IsolationForest")
    mlflow.log_param("contamination", 0.2)
    mlflow.log_param("n_features", len(feature_cols))
    # log your precision, recall, f1
    mlflow.log_metric("precision_anomaly", report_dict["1"]["precision"])
    mlflow.log_metric("recall_anomaly", report_dict["1"]["recall"])
    mlflow.log_metric("f1_anomaly", report_dict["1"]["f1-score"])
    mlflow.log_metric("bot_recall", (bots['prediction'] == 1).sum() / len(bots))
    mlflow.log_metric("fraud_recall", (fraud['prediction'] == 1).sum() / len(fraud))