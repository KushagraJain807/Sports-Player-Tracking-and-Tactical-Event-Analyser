
from ultralytics import YOLO
from pathlib import Path
import json

# Paths
MODEL_PATH = "models/best.pt"
DATASET_DIR = Path(r"D:\Football_Datasets_V1")
DATA_YAML = DATASET_DIR / "data.yaml"

# Load the existing trained checkpoint
model = YOLO(MODEL_PATH)

# Evaluate on the independent labeled test split
metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=8,
    device="cpu",  # Change to 0 if CUDA GPU is available
    plots=True,
    project="evaluation_results",
    name="football_test",
    exist_ok=True,
    verbose=True
)

precision = float(metrics.box.mp)
recall = float(metrics.box.mr)
map50 = float(metrics.box.map50)
map5095 = float(metrics.box.map)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0 else 0
)

results = {
    "dataset": "Roboflow Football Players Detection V1",
    "split": "test",
    "precision": precision,
    "recall": recall,
    "f1_from_mean_precision_recall": f1,
    "mAP50": map50,
    "mAP50_95": map5095,
    "test_images_expected": 13
}

output_path = Path("evaluation_results") / "test_metrics.json"
output_path.parent.mkdir(exist_ok=True)

output_path.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8"
)

print("\n========== TEST RESULTS ==========")
for key, value in results.items():
    print(f"{key}: {value}")

print("\nResults saved to:", output_path)
print("Plots saved in evaluation_results/football_test/")
