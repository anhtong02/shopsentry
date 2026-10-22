from models.data_loader import load_training_data
from models.pseudo_labelers.autoencoder_labeler import AutoencoderLabeler
from models.classifiers.xgboost_classifier import XGBoostClassifier

df, feature_cols = load_training_data()

for n in [50, 100, 200, 500]:
    f1s = []
    for seed in range(5):
        labeled = AutoencoderLabeler(verified_sample=n, random_state=seed).fit_predict(df, feature_cols)
        m = XGBoostClassifier().fit_evaluate(labeled, feature_cols)
        f1s.append(m["f1_anomaly"])
    print(f"{n:>4} verified | F1 min={min(f1s):.2f} avg={sum(f1s)/5:.2f} max={max(f1s):.2f}")