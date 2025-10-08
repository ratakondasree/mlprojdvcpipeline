import yaml, pathlib, json
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report

params = yaml.safe_load(open("params.yaml"))
proc_dir = pathlib.Path(params["data"]["processed_dir"])
model_dir = pathlib.Path(params["paths"]["model_dir"])
metrics_dir = pathlib.Path(params["paths"]["metrics_dir"])
metrics_dir.mkdir(parents=True, exist_ok=True)

test = pd.read_csv(proc_dir / "test.csv")
X_test = test.drop(columns=["species"])
y_test = test["species"]

model = load(model_dir / "model.joblib")
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# DVC will track this file as metrics
metrics = {"test_accuracy": float(acc)}
with open(metrics_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Evaluation:", metrics)
