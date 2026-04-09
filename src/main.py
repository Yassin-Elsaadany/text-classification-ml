import json
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    SSCD1_PATH, SSCD2_PATH,
    TEXT_COLUMN, LABEL_COLUMN,
    TEST_SIZE, RANDOM_STATE,
    MAX_FEATURES, USE_TFIDF,
    OUTPUT_METRICS
)
from data_loader import load_and_prepare_dataset
from vectorize import get_vectorizer
from train import train_logistic_regression, train_naive_bayes
from evaluate import evaluate_model

def run_dataset(dataset_name, dataset_path):
    df = load_and_prepare_dataset(dataset_path, TEXT_COLUMN, LABEL_COLUMN)

    print(f"\n===== {dataset_name} =====")
    print("Shape:", df.shape)
    print("Class distribution:")
    print(df[LABEL_COLUMN].value_counts())

    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    vectorizer = get_vectorizer(use_tfidf=USE_TFIDF, max_features=MAX_FEATURES)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    lr_model = train_logistic_regression(X_train_vec, y_train)
    nb_model = train_naive_bayes(X_train_vec, y_train)

    lr_acc, lr_report = evaluate_model(lr_model, X_test_vec, y_test)
    nb_acc, nb_report = evaluate_model(nb_model, X_test_vec, y_test)

    return [
        {
            "dataset": dataset_name,
            "model": "Logistic Regression",
            "accuracy": lr_acc,
            "report": lr_report
        },
        {
            "dataset": dataset_name,
            "model": "Naive Bayes",
            "accuracy": nb_acc,
            "report": nb_report
        }
    ]

def main():
    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)

    results = []
    results.extend(run_dataset("SSCD1", SSCD1_PATH))
    results.extend(run_dataset("SSCD2", SSCD2_PATH))

    results_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)
    results_df.to_csv(OUTPUT_METRICS / "model_ranking.csv", index=False)

    with open(OUTPUT_METRICS / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== FINAL RANKING ===")
    print(results_df[["dataset", "model", "accuracy"]])

if __name__ == "__main__":
    main()
