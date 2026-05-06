import xgboost as xgb
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from models.data_loader import load_training_data

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("anomaly_detection")

df, feature_cols = load_training_data()
X = df[feature_cols]
y_true = df["label"]  # for evaluation only

# Same split as IF and the true-label XGBoost run
X_train, X_test, y_true_train, y_true_test = train_test_split(
    X, y_true, test_size=0.35, stratify=y_true, random_state=1
)

# Step 1: IF generates pseudo-labels on training data only
iso = IsolationForest(contamination=0.2, random_state=53)
iso.fit(X_train)
pseudo_train = pd.Series(iso.predict(X_train)).apply(lambda x: 1 if x == -1 else 0).values

# Step 2: Recompute scale_pos_weight from pseudo-labels
# (you don't know the real ratio in production — you only know what IF flagged)
neg = (pseudo_train == 0).sum()
pos = (pseudo_train == 1).sum()
scale = neg / pos if pos else 1.0

# Step 3: Train XGBoost on pseudo-labels
xg = xgb.XGBClassifier(n_estimators=70, max_depth=3, learning_rate=0.1, scale_pos_weight=scale)
xg.fit(X_train, pseudo_train)

# Step 4: Evaluate against TRUE labels on the test set
preds = xg.predict(X_test)
report = classification_report(y_true_test, preds, output_dict=True)
print(classification_report(y_true_test, preds))

# Sanity check: how good were the pseudo-labels themselves?
pseudo_quality = classification_report(y_true_train, pseudo_train, output_dict=True)
print(f"\nPseudo-label quality on train: F1={pseudo_quality['1']['f1-score']:.3f}")

# Per-agent breakdown
test_df = df.loc[X_test.index].copy()
test_df["prediction"] = preds
bots = test_df[test_df["agent_type"] == "bot"]
fraud = test_df[test_df["agent_type"] == "fraud"]
print(f"Bots caught: {(bots['prediction']==1).sum()} / {len(bots)}")
print(f"Fraud caught: {(fraud['prediction']==1).sum()} / {len(fraud)}")

with mlflow.start_run(run_name="xgboost_pseudo_labels"):
    mlflow.log_param("model", "XGBoost_PseudoLabels")
    mlflow.log_param("pseudo_label_source", "IsolationForest_contam_0.2")
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("scale_pos_weight", scale)
    mlflow.log_metric("precision_anomaly", report["1"]["precision"])
    mlflow.log_metric("recall_anomaly", report["1"]["recall"])
    mlflow.log_metric("f1_anomaly", report["1"]["f1-score"])
    mlflow.log_metric("pseudo_label_f1", pseudo_quality["1"]["f1-score"])
    mlflow.log_metric("bot_recall", (bots['prediction']==1).sum() / len(bots) if len(bots) else 0)
    mlflow.log_metric("fraud_recall", (fraud['prediction']==1).sum() / len(fraud) if len(fraud) else 0)