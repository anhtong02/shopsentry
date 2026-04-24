from models.data_loader import load_training_data
from sklearn.preprocessing import StandardScaler
import mlflow
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("anomaly_detection")
#load data
df, feature_cols = load_training_data()

normal_df = df[df["label"] == 0]
anomalies_df = df[df["label"] == 1]

train_normal, test_normal = train_test_split(normal_df, test_size = 0.3, random_state=42)

test_df = pd.concat([test_normal, anomalies_df]).copy()

# scale features 
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_normal[feature_cols])

model = keras.Sequential([
    Dense(5, activation='relu', input_shape=(9,)), #encoder
    Dense(2, activation='relu'), #bottleneck
    Dense(5, activation='relu'), #decoder
    Dense(9, activation='sigmoid') #output layer

])
model.compile(optimizer='adam', loss ='mse')
model.fit(train_scaled, train_scaled, epochs=120, batch_size=32, verbose=1)
# 5. Reconstruct all data
train_reconstructed = model.predict(train_scaled)


# 6. Compute reconstruction error per row
train_errors = np.mean((train_scaled - train_reconstructed) ** 2, axis=1)

# 7. Set threshold — 85th percentile
threshold = np.percentile(train_errors, 95) # Accept a 5% False Positive rate on training data

test_scaled = scaler.transform(test_df[feature_cols])

# 8. Reconstruct the UNSEEN test set
test_reconstructed = model.predict(test_scaled)
test_errors = np.mean((test_scaled - test_reconstructed) ** 2, axis=1)

test_df["prediction"] = (test_errors > threshold).astype(int)
# 9. Evaluate
#print(confusion_matrix(df["label"], df["prediction"]))
report_dict = classification_report(test_df["label"], test_df["prediction"], output_dict=True)

bots = test_df[test_df["agent_type"] == "bot"]
print(f"Total unseen bots: {len(bots)}")
print(f"Caught: {(bots['prediction'] == 1).sum()}")

fraud = test_df[test_df["agent_type"] == "fraud"]
print(f"Total unseen fraud: {len(fraud)}")
print(f"Caught: {(fraud['prediction'] == 1).sum()}")
print(classification_report(test_df["label"], test_df["prediction"]))
# with mlflow.start_run(run_name="autoencoder_vanilla"):
#     mlflow.log_param("model", "IsolationForest")  # model name, epochs, batch_size, architecture, threshold
#     mlflow.log_param("n_features", len(feature_cols))
#     mlflow.log_metric("precision_anomaly", report_dict["1"]["precision"])
#     mlflow.log_metric("recall_anomaly", report_dict["1"]["recall"])
#     mlflow.log_metric("f1_anomaly", report_dict["1"]["f1-score"])
#     mlflow.log_metric("bot_recall", (bots['prediction'] == 1).sum() / len(bots))
#     mlflow.log_metric("fraud_recall", (fraud['prediction'] == 1).sum() / len(fraud))