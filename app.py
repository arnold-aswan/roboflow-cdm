
from flask import Flask, request, jsonify
from flask_cors import CORS
from roboflow import Roboflow
from dotenv import load_dotenv
# from PIL import Image
import os
import cv2
# import tempfile
import numpy as np
import random
# import io
import base64
from werkzeug.utils import secure_filename


# Load .env
load_dotenv()

# Env vars
API_KEY = os.getenv("ROBOFLOW_API_KEY")
PROJECT_ID = os.getenv("ROBOFLOW_PROJECT")
MODEL_VERSION = os.getenv("ROBOFLOW_VERSION")

# Flask app
app = Flask(__name__)
CORS(app)

# ensure folders exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# Roboflow setup
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project(PROJECT_ID)
model = project.version(MODEL_VERSION).model

# Confidence threshold (set to 1%)
CONFIDENCE_THRESHOLD = 0.01


@app.get("/health")
def health():
    return {"status": "ok"}

# Generate consistent random colors for each class
CLASS_COLORS = {}
def get_class_color(class_name):
    if class_name not in CLASS_COLORS:
        CLASS_COLORS[class_name] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return CLASS_COLORS[class_name]

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_path)

    # Run prediction (include masks for segmentation)
    preds = model.predict(temp_path, confidence=1).json()

    img = cv2.imread(temp_path)
    h, w, _ = img.shape
    total_pixels = h * w

    detections = []
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    class_masks = {}
    
    # Minimum percentage each detection should get (adjustable)
    MIN_PERCENTAGE = 0.05  # 0.05% of total image
    min_pixels = int((MIN_PERCENTAGE * total_pixels) / 100)
    
    # Sort predictions by confidence (highest first)
    sorted_preds = sorted(preds["predictions"], key=lambda x: x["confidence"], reverse=True)

    for pred in sorted_preds:
        cls = pred["class"]
        conf = pred["confidence"]

        if "points" in pred:  # segmentation result
            # Create original mask
            mask = np.zeros((h, w), dtype=np.uint8)
            points = np.array([[p["x"], p["y"]] for p in pred["points"]], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
            
            original_area = np.count_nonzero(mask)

            # Calculate unique area (non-overlapping with previous detections)
            unique_mask = cv2.bitwise_and(mask, cv2.bitwise_not(combined_mask))
            unique_area = np.count_nonzero(unique_mask)
            
            # If unique area is too small, try to guarantee minimum area
            if unique_area < min_pixels and original_area > 0:
                # Strategy 1: Try eroded version of original mask
                kernel_size = max(3, int(min(pred["width"], pred["height"]) / 50))
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                eroded_mask = cv2.erode(mask, kernel, iterations=1)
                
                if np.count_nonzero(eroded_mask) > unique_area:
                    unique_mask = eroded_mask
                    unique_area = np.count_nonzero(unique_mask)
                
                # Strategy 2: If still too small, create a small center region
                if unique_area < min_pixels:
                    center_y, center_x = int(pred["y"]), int(pred["x"])
                    radius = max(15, int(min(pred["width"], pred["height"]) * 0.15))
                    
                    center_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(center_mask, (center_x, center_y), radius, 255, -1)
                    
                    # Only use pixels within original detection boundary
                    center_mask = cv2.bitwise_and(center_mask, mask)
                    center_area = np.count_nonzero(center_mask)
                    
                    if center_area > unique_area:
                        unique_mask = center_mask
                        unique_area = center_area

            # Update combined mask with the final unique mask
            combined_mask = cv2.bitwise_or(combined_mask, unique_mask)
            
            # Store class mask
            if cls not in class_masks:
                class_masks[cls] = np.zeros((h, w), dtype=np.uint8)
            class_masks[cls] = cv2.bitwise_or(class_masks[cls], unique_mask)

            # Calculate overlap percentage
            overlap_pct = round(((original_area - unique_area) / original_area * 100), 2) if original_area > 0 else 0

            # Draw bounding box
            x0, y0, x1, y1 = int(pred["x"] - pred["width"] / 2), int(pred["y"] - pred["height"] / 2), \
                             int(pred["x"] + pred["width"] / 2), int(pred["y"] + pred["height"] / 2)
            
            # Generate consistent color based on class name
            color_seed = hash(cls) % 1000000
            np.random.seed(color_seed)
            color = tuple(np.random.randint(50, 255, 3).tolist())
            np.random.seed()  # Reset seed
            
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            # Label with class + confidence
            label = f"{cls} {conf:.2f}"
            cv2.putText(img, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            detections.append({
                "bbox": [x0, y0, x1, y1],
                "class": cls,
                "confidence": conf,
                "area_pixels": int(unique_area),
                "original_area_pixels": int(original_area),
                "overlap_percentage": overlap_pct
            })

    # Compute final percentages
    class_pixels = {cls: int(np.count_nonzero(mask)) for cls, mask in class_masks.items()}
    percentages = {cls: round((px / total_pixels) * 100, 2) for cls, px in class_pixels.items()}
    total_percentage = sum(percentages.values())
    total_coverage_pixels = sum(class_pixels.values())

    # Annotate image with percentages
    y_offset = 30
    for cls, pct in sorted(percentages.items(), key=lambda x: x[1], reverse=True):
        text = f"{cls}: {pct}%"
        cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30

    # Add total percentage and coverage info
    cv2.putText(img, f"Total: {total_percentage:.2f}%", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    y_offset += 30
    cv2.putText(img, f"Covered: {total_coverage_pixels:,} px", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

    output_path = os.path.join(UPLOAD_FOLDER, "annotated_" + filename)
    cv2.imwrite(output_path, img)

    # Save annotated image
    output_path = os.path.join(UPLOAD_FOLDER, "annotated_" + filename)
    cv2.imwrite(output_path, img)

    # Clean up temporary file
    try:
        os.remove(temp_path)
    except:
        pass

    # Convert image to base64
    with open(output_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    results = {
        "results": {
            "detections": detections,
            "percentages": percentages,
            "total_percentage": round(total_percentage, 2),
            "coverage_pixels": total_coverage_pixels,
            "total_pixels": total_pixels,
            "image_dimensions": {"width": w, "height": h},
            "detection_count": len(detections),
            "min_area_guarantee": f"{MIN_PERCENTAGE}%"
        },
        "annotated_image": img_base64
    }

    return jsonify(results)

def run_model(image):
    # Run inference
    results = model.predict(image)

    detections = []
    for r in results:
        for box, mask, cls, conf in zip(r.boxes.xyxy, r.masks.data, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())
            mask_np = mask.cpu().numpy().astype("uint8")

            detections.append({
                "class": model.names[int(cls)],
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "mask": mask_np
            })

    return detections

def compute_percentages(detections, img_h, img_w):
    total_area = img_h * img_w
    class_areas = {}

    for det in detections:
        mask_area = int(det["mask"].sum())  # count white pixels in mask
        cls = det["class"]
        class_areas[cls] = class_areas.get(cls, 0) + mask_area

    percentages = {
        cls: round((area / total_area) * 100, 2)
        for cls, area in class_areas.items()
    }

    return percentages

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

