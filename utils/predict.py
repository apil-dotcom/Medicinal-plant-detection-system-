from ultralytics import YOLO
import json
import os

# ------------------ BASE DIRECTORY ------------------
BASE_DIR = r"D:\8semProject\8semProject"

# ------------------ LOAD YOLO MODEL ------------------
model = YOLO(
    os.path.join(
        BASE_DIR,
        "YOLOClassification32classes2ndTry-20260131T174214Z-3-001",
        "YOLOClassification32classes2ndTry",
        "runs",
        "classify",
        "medicinal_plants",
        "weights",
        "best.pt"
    )
)

# ------------------ LOAD JSON FILES ------------------
with open(os.path.join(BASE_DIR, "model", "class_name.json"), "r", encoding="utf-8") as f:
    class_name_map = json.load(f)

with open(os.path.join(BASE_DIR, "model", "plants.json"), "r", encoding="utf-8") as f:
    plant_benefits = json.load(f)


# ------------------ PREDICTION FUNCTION ------------------
def predict_image(image_path):
    """
    Predict the plant in an image and return:
    - plant name
    - confidence %
    - list of benefits
    """

    results = model.predict(image_path)

    # Check if YOLO returned probabilities
    if results[0].probs is None:
        return "No plant detected", 0, ["No benefits found"]

    # Get top prediction
    cls_id = results[0].probs.top1
    confidence = round(results[0].probs.top1conf.item() * 100, 2)

    # Map class ID to plant name
    class_str = model.names[cls_id].strip()  # remove extra spaces
    plant_name = class_name_map.get(class_str, "Unknown")

    # Get benefits
    plant_info = plant_benefits.get(plant_name, {})
    benefits = plant_info.get("benefits", ["No benefits found"])

    return plant_name, confidence, benefits
