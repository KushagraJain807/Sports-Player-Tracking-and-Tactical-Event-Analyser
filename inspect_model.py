
from ultralytics import YOLO
import json
from pathlib import Path

model = YOLO("models/best.pt")

print("\nMODEL TASK:", model.task)
print("\nMODEL CLASSES:", model.names)
print("\nMODEL ARCHITECTURE:")
print(model.model)

print("\nMODEL CONFIGURATION:")
print(getattr(model.model, "yaml", None))

ckpt = model.ckpt or {}
print("\nCHECKPOINT KEYS:", list(ckpt.keys()))

print("\nTRAINING ARGUMENTS:")
args = ckpt.get("train_args", {})
for key, value in sorted(args.items()):
    print(f"{key}: {value}")

print("\nEPOCH:", ckpt.get("epoch"))
print("\nTRAINING METRICS:", ckpt.get("train_metrics"))
print("\nTRAINING RESULTS:", ckpt.get("train_results"))

output = {
    "task": model.task,
    "classes": model.names,
    "architecture": str(model.model),
    "model_yaml": str(getattr(model.model, "yaml", None)),
    "training_args": args,
    "epoch": ckpt.get("epoch"),
    "training_metrics": ckpt.get("train_metrics"),
}

Path("model_verification.json").write_text(
    json.dumps(output, indent=2, default=str),
    encoding="utf-8"
)

print("\nSaved model_verification.json")
