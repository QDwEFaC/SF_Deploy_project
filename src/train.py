"""
Создание и обучение модели
Сохранение модели в joblib
"""
import pandas as pd
import joblib
import json

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.config import model_path, data_csv_path, feature_columns_path


def main():

    df = pd.read_csv(data_csv_path())

    TARGET_NAME = 'default.payment.next.month'

    y = df[TARGET_NAME]
    X = df.drop([TARGET_NAME, 'ID'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    out_model = model_path()
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_model)

    meta = {
        "feature_columns": list(X.columns),
        "target": TARGET_NAME,
        "positive_class": int(clf.classes_[1]) if len(clf.classes_) == 2 else 1,
    }
    feat_path = feature_columns_path()
    feat_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f'Model saved at {model_path()}')
    print(f'Feature metadata saved at {feature_columns_path()}')
    print(f'Score on test: {score:.3f}')


if __name__ == "__main__":
    main()