from models.data_loader import load_training_data
from models.holdout import split_train_holdout
from models.pseudo_labelers.autoencoder_labeler import AutoencoderLabeler
from models.classifiers.xgboost_classifier import XGBoostClassifier

df, feature_cols = load_training_data()
train_df, _ = split_train_holdout(df)   # verified sample comes from the training split only

for n in [50, 100, 200, 500]:
    f1s = []
    for seed in range(5):
        ae = AutoencoderLabeler(verified_sample=n, random_state=seed)
        ae.fit(train_df, feature_cols)
        labeled = ae.predict(df)
        m = XGBoostClassifier().fit_evaluate(labeled, feature_cols)
        f1s.append(m["f1_anomaly"])
    print(f"{n:>4} verified | F1 min={min(f1s):.2f} avg={sum(f1s)/5:.2f} max={max(f1s):.2f}")