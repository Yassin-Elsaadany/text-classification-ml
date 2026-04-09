import pandas as pd
from preprocess import clean_text

def load_csv_safely(path, text_col="message", label_col="label"):
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # skip header
    for i, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line:
            continue

        # split only on the first comma
        parts = line.split(",", 1)

        if len(parts) != 2:
            print(f"Skipping malformed line {i}: {line}")
            continue

        label = parts[0].strip()
        message = parts[1].strip()

        rows.append({label_col: label, text_col: message})

    return pd.DataFrame(rows)

def load_and_prepare_dataset(path, text_col="message", label_col="label"):
    try:
        df = pd.read_csv(path)
    except Exception:
        print(f"[INFO] Standard CSV reading failed for {path}. Using safe loader...")
        df = load_csv_safely(path, text_col=text_col, label_col=label_col)

    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    return df
