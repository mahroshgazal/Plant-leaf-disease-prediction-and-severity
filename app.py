import os
import io
import json
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import cv2

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── PlantVillage class names ────────────────────────────────────────────────
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

DISEASE_INFO = {
    "Apple_scab": {"severity": "High", "treatment": "Apply fungicide (captan or mancozeb) early season. Remove infected leaves.", "color": "#e74c3c"},
    "Black_rot": {"severity": "High", "treatment": "Prune infected wood. Apply copper-based fungicides.", "color": "#c0392b"},
    "Cedar_apple_rust": {"severity": "Medium", "treatment": "Remove nearby juniper/cedar trees. Use myclobutanil fungicide.", "color": "#e67e22"},
    "healthy": {"severity": "None", "treatment": "No treatment needed. Plant appears healthy!", "color": "#27ae60"},
    "Powdery_mildew": {"severity": "Medium", "treatment": "Apply sulfur or potassium bicarbonate. Improve air circulation.", "color": "#f39c12"},
    "Cercospora_leaf_spot": {"severity": "Medium", "treatment": "Apply chlorothalonil. Rotate crops annually.", "color": "#e67e22"},
    "Common_rust": {"severity": "Medium", "treatment": "Apply triazole fungicide. Plant resistant varieties.", "color": "#e74c3c"},
    "Northern_Leaf_Blight": {"severity": "High", "treatment": "Use resistant hybrids. Apply foliar fungicides at early infection.", "color": "#c0392b"},
    "Esca_(Black_Measles)": {"severity": "High", "treatment": "No cure; remove infected vines. Protect pruning wounds.", "color": "#8e44ad"},
    "Leaf_blight": {"severity": "High", "treatment": "Apply mancozeb or copper fungicide. Avoid overhead irrigation.", "color": "#e74c3c"},
    "Haunglongbing": {"severity": "Critical", "treatment": "No cure. Remove infected trees. Control psyllid vectors.", "color": "#c0392b"},
    "Bacterial_spot": {"severity": "High", "treatment": "Apply copper bactericide. Use disease-free seeds.", "color": "#e74c3c"},
    "Early_blight": {"severity": "Medium", "treatment": "Apply chlorothalonil or mancozeb. Remove lower infected leaves.", "color": "#e67e22"},
    "Late_blight": {"severity": "Critical", "treatment": "Apply metalaxyl immediately. Destroy infected plants.", "color": "#c0392b"},
    "Leaf_Mold": {"severity": "Medium", "treatment": "Improve ventilation. Apply copper-based fungicide.", "color": "#f39c12"},
    "Septoria_leaf_spot": {"severity": "Medium", "treatment": "Remove infected leaves. Apply mancozeb or chlorothalonil.", "color": "#e67e22"},
    "Spider_mites": {"severity": "Medium", "treatment": "Apply miticide (abamectin). Increase humidity.", "color": "#f39c12"},
    "Target_Spot": {"severity": "Medium", "treatment": "Apply azoxystrobin fungicide. Rotate crops.", "color": "#e67e22"},
    "Yellow_Leaf_Curl_Virus": {"severity": "Critical", "treatment": "No cure. Control whitefly vectors. Remove infected plants.", "color": "#c0392b"},
    "mosaic_virus": {"severity": "High", "treatment": "No cure. Use virus-free seeds. Control aphid vectors.", "color": "#8e44ad"},
    "Leaf_scorch": {"severity": "Medium", "treatment": "Remove infected leaves. Apply fungicide in spring.", "color": "#e67e22"},
}

# ─── Model loader ────────────────────────────────────────────────────────────
MODEL = None


def load_model():
    global MODEL
    model_path = os.path.join(os.path.dirname(
        __file__), 'model', 'leaf_disease_model.pth')

    try:
        import torch
        import torchvision.models as models

        if os.path.exists(model_path):
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            MODEL = model
            print("✅ Loaded trained model from", model_path)
        else:
            print("⚠️  No trained model found at models/plant_disease_model.pth")
            print("    Using visual analysis fallback (for demo purposes).")
    except ImportError:
        print("⚠️  PyTorch not installed. Using visual analysis fallback.")


def predict_with_model(img_array):
    """Predict using the trained PyTorch model."""
    import torch
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    tensor = transform(img_array).unsqueeze(0)
    with torch.no_grad():
        output = MODEL(tensor)
        probs = torch.softmax(output, dim=1)[0]
        top5 = torch.topk(probs, 5)

    results = []
    for prob, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        results.append({
            "class": CLASS_NAMES[idx],
            "confidence": round(prob * 100, 2)
        })
    print("MODEL OUTPUT:", results)
    return results


def predict_visual_fallback(img_array):
    """
    Rule-based visual analysis when no trained model is available.
    Analyzes color distribution to estimate leaf health.
    """
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Green channel (healthy leaf)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    # Yellow/brown (disease spots)
    lower_yellow = np.array([10, 60, 60])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size

    # Dark spots (necrosis)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 50])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    dark_ratio = np.sum(dark_mask > 0) / dark_mask.size

    candidates = []

    # Is it even a leaf?
    if green_ratio < 0.05 and yellow_ratio < 0.05:
        return [{
            "class": "Unknown",
            "confidence": 95.0,
            "note": "⚠️ Image may not contain a leaf. Please upload a clear leaf image."
        }]

    if green_ratio > 0.45 and yellow_ratio < 0.1 and dark_ratio < 0.05:
        candidates.append(("healthy", 85 + green_ratio * 10))

    if yellow_ratio > 0.15:
        candidates.append(("Early_blight", min(70 + yellow_ratio * 50, 92)))

    if dark_ratio > 0.1 and yellow_ratio > 0.1:
        candidates.append(("Late_blight", min(65 + dark_ratio * 60, 90)))

    if yellow_ratio > 0.2 and green_ratio < 0.3:
        candidates.append(("Leaf_Mold", min(60 + yellow_ratio * 40, 88)))

    if dark_ratio > 0.15 and yellow_ratio < 0.1:
        candidates.append(("Black_rot", min(62 + dark_ratio * 50, 87)))

    if not candidates:
        if green_ratio > yellow_ratio:
            candidates.append(("healthy", 60))
        else:
            candidates.append(("Early_blight", 55))

    candidates.sort(key=lambda x: x[1], reverse=True)
    total = sum(c[1] for c in candidates[:3])

    results = []
    for name, score in candidates[:4]:
        # Find best matching class name
        matched_class = next(
            (c for c in CLASS_NAMES if name.lower() in c.lower()),
            f"Tomato___{name}" if name != "Unknown" else "Unknown"
        )
        results.append({
            "class": matched_class,
            "confidence": round((score / total) * 100 if total > 0 else score, 2)
        })
    return results


def get_disease_info(class_name):
    """Look up disease info from class name."""
    parts = class_name.split("___")
    plant = parts[0].replace("_", " ").title() if parts else "Unknown"
    disease_raw = parts[1] if len(parts) > 1 else "Unknown"
    disease = disease_raw.replace("_", " ").replace("  ", " ")

    info = {"severity": "Unknown",
            "treatment": "Consult a plant pathologist.", "color": "#7f8c8d"}
    for key, val in DISEASE_INFO.items():
        if key.lower() in disease_raw.lower():
            info = val
            break

    return {
        "plant": plant,
        "disease": disease,
        "severity": info["severity"],
        "treatment": info["treatment"],
        "color": info["color"]
    }

# ─── Routes ──────────────────────────────────────────────────────────────────


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(img)

        # Resize for display
        display_img = img.resize((300, 300))
        buffered = io.BytesIO()
        display_img.save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        # Predict
        if MODEL is not None:
            predictions = predict_with_model(img_array)
            model_type = "Neural Network (ResNet-50)"
        else:
            predictions = predict_visual_fallback(img_array)
            model_type = "Visual Analysis (Demo Mode)"

        # Enrich with disease info
        enriched = []
        for p in predictions:
            info = get_disease_info(p["class"])
            enriched.append({**p, **info})

        top = enriched[0]  # best prediction

        return jsonify({
            "prediction": {
                "plant": top["plant"],
                "disease": top["disease"],
                "confidence": top["confidence"],
                "type": top["severity"],   # using severity as type
                "is_healthy": "healthy" in top["disease"].lower(),
                "class": top["class"],
                "advice": top["treatment"]
            },
            "alternatives": enriched[1:4],  # next predictions
            "demo_mode": MODEL is None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "classes": len(CLASS_NAMES)
    })


if __name__ == '__main__':
    load_model()
    print("\n🌿 Leaf Disease Detector running at http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
