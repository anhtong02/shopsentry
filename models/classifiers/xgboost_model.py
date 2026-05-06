import xgboost as xgb
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.metrics import confusion_matrix, classification_report
from models.data_loader import load_training_data


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("anomaly_detection")
#load data
df, feature_cols = load_training_data()
X = df[feature_cols]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, stratify=y, random_state=1)

xg = xgb.XGBClassifier(n_estimators=70, max_depth=3, learning_rate=0.1, scale_pos_weight=5.4)

xg.fit(X_train, y_train)

predictions = xg.predict(X_test)

print(classification_report(y_test, predictions))

import pandas as pd
importance = pd.DataFrame({"feature": feature_cols, "importance": xg.feature_importances_})
print(importance.sort_values("importance", ascending=False))