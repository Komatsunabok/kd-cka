# This script summarizes the test metrics of teacher models stored in a specified directory.
import os
import json
import pandas as pd

base_dir = "../save/teachers/models"
summary = []


for model_name in os.listdir(base_dir):
    model_path = os.path.join(base_dir, model_name)
    if os.path.isdir(model_path):
        entry = {"name": model_name}  # モデル名をベースに初期化

        # parameters.json 読み込み
        params_path = os.path.join(model_path, "parameters.json")
        if os.path.isfile(params_path):
            with open(params_path, "r") as f:
                try:
                    params = json.load(f)
                    entry["model"] = params.get("model")
                    entry["dataset"] = params.get("dataset")
                except json.JSONDecodeError:
                    print(f"[Warning] {model_name} のパラメータJSONが読み込めませんでした。")
                    continue

        # test_best_metrics.json 読み込み
        metrics_path = os.path.join(model_path, "test_best_metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r") as f:
                try:
                    metrics = json.load(f)
                    entry["test_loss"] = metrics.get("test_loss")
                    entry["test_acc"] = metrics.get("test_acc")
                    entry["test_acc_top5"] = metrics.get("test_acc_top5")
                    entry["epoch"] = metrics.get("epoch")
                except json.JSONDecodeError:
                    print(f"[Warning] {model_name} のメトリクスJSONが読み込めませんでした。")

        summary.append(entry)

# DataFrameにまとめて確認
df = pd.DataFrame(summary)
print(df)

# 必要ならCSV保存
# df.to_csv("teacher_models_summary.csv", index=False)
