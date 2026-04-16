#!/usr/bin/env python3
"""
BPS DTSEN Captcha Solver - Pure Python (No External AI API)
============================================================
Solves 6-char alphanumeric captchas from BPS DTSEN API using:
1. Image preprocessing (magenta noise removal + binarization)
2. Character segmentation (fixed grid)
3. CNN classifier (PyTorch) with ensemble fallback (sklearn SVM+KNN+RF)

Training data: D:\\dtsen\\captcha\\images\\ + captcha_results.json

Usage:
    # Scrape & solve N captchas (live from API)
    python3 captcha_solver.py --count 20

    # Solve a specific image file
    python3 captcha_solver.py --image captcha/images/captcha_0000.jpg

    # Train/retrain the model
    python3 captcha_solver.py --train

    # Evaluate on existing training data
    python3 captcha_solver.py --eval

    # Scrape only (no solving), then solve later
    python3 captcha_solver.py --count 50 --fetch-only
    python3 captcha_solver.py --solve-only
"""

import os
import sys
import json
import time
import pickle
import base64
import argparse
import requests
import numpy as np
from pathlib import Path
from datetime import datetime

CAPTCHA_API_URL = "https://dtsen-form-api.web.bps.go.id/api/reload-captcha"
SCRIPT_DIR = Path(__file__).parent.absolute()
BACKEND_DIR = SCRIPT_DIR.parent
CAPTCHA_DIR = BACKEND_DIR
IMAGES_DIR = CAPTCHA_DIR / "images"
RESULTS_FILE = CAPTCHA_DIR / "captcha_results.json"
MODEL_FILE = CAPTCHA_DIR / "captcha_solver_model.pkl"

# Valid characters (from BPS captcha analysis - no ambiguous chars like 0/O, 1/l/I)
VALID_CHARS = "2346789abcdefghjmnpqrtuxyz"
CHAR_TO_IDX = {c: i for i, c in enumerate(VALID_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VALID_CHARS)}

DEFAULT_DELAY = 1.5

# --- IMAGE PREPROCESSING ---

def preprocess_captcha(img_bgr):
    """Remove magenta noise lines, sharpen, binarize captcha image.
    
    Pipeline:
    1. HSV color space - detect & remove magenta noise lines via inpainting
    2. Grayscale + sharpen
    3. Otsu threshold (binary inverse - text=white, bg=black)
    4. Morphological close (connect fragmented character parts)
    5. Morphological open (remove small noise artifacts)
    """
    import cv2
    
    # Remove magenta noise lines
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mag_mask = cv2.inRange(hsv, np.array([140, 40, 80]), np.array([175, 255, 255]))
    cleaned = cv2.inpaint(img_bgr, mag_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    # Grayscale + sharpen
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    kernel_sharpen = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
    
    # Otsu binarization (BINARY_INV: text=255/white, bg=0/black)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological close (connect character fragments)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    return binary


def segment_chars(binary, n_chars=6, img_width=360):
    """Segment binary image into individual character crops.
    
    Uses fixed grid segmentation (360px / 6 chars = 60px each)
    with vertical crop optimization (tight bounding box per char).
    """
    char_w = img_width // n_chars
    padding = 5
    chars = []
    
    for i in range(n_chars):
        x1 = i * char_w + padding
        x2 = (i + 1) * char_w - padding
        crop = binary[:, x1:x2]
        
        # Find vertical bounds (tight crop)
        row_sum = crop.sum(axis=1)
        nonzero_rows = np.where(row_sum > 0)[0]
        if len(nonzero_rows) > 0:
            y_top = max(0, nonzero_rows[0] - 2)
            y_bottom = min(crop.shape[0], nonzero_rows[-1] + 3)
            char_crop = crop[y_top:y_bottom, :]
        else:
            char_crop = crop
        
        chars.append(char_crop)
    
    return chars


# --- FEATURE EXTRACTION ---

def extract_features(img, size=28):
    """Extract combined feature vector from a character image.
    
    Features:
    - Raw pixels (28x28 = 784)
    - HOG descriptors (324)
    - Horizontal projection profile (28)
    - Vertical projection profile (28)
    - Zone density (4x4 = 16)
    - Structural features (aspect ratio, fill ratio)
    Total: ~1182 dimensions
    """
    import cv2
    
    resized = cv2.resize(img, (size, size))
    
    # Raw pixels
    flat = resized.flatten().astype(np.float32) / 255.0
    
    # HOG features
    hog = cv2.HOGDescriptor((size, size), (14, 14), (7, 7), (7, 7), 9)
    hog_feat = hog.compute(resized).flatten()
    
    # Projection profiles
    h_proj = resized.sum(axis=1).astype(np.float32) / (size * 255)
    v_proj = resized.sum(axis=0).astype(np.float32) / (size * 255)
    
    # Zone density (4x4 grid)
    zone_size = size // 4
    zones = []
    for r in range(4):
        for c in range(4):
            zone = resized[r*zone_size:(r+1)*zone_size, c*zone_size:(c+1)*zone_size]
            zones.append(zone.mean() / 255.0)
    
    # Structural features
    row_sum = resized.sum(axis=1)
    nonzero_rows = np.where(row_sum > 0)[0]
    col_sum = resized.sum(axis=0)
    nonzero_cols = np.where(col_sum > 0)[0]
    if len(nonzero_rows) > 0 and len(nonzero_cols) > 0:
        aspect = len(nonzero_cols) / len(nonzero_rows)
        fill = resized[nonzero_rows[0]:nonzero_rows[-1]+1, 
                       nonzero_cols[0]:nonzero_cols[-1]+1].mean() / 255
    else:
        aspect = 1.0
        fill = 0.0
    
    return np.concatenate([flat, hog_feat, h_proj, v_proj, np.array(zones), [aspect, fill]])


# --- DATA AUGMENTATION ---

def augment_char_images(img_list, target_per_class=150, size=28):
    """Generate augmented character images for training.
    
    Augmentation types:
    - Translation (shift ±3px)
    - Rotation (±12 degrees)
    - Scaling (0.8x - 1.2x)
    - Erosion / Dilation (morphological)
    - Additive noise (0-50 intensity)
    - Perspective warp (±3px corner displacement)
    """
    import cv2
    
    result = []
    for img in img_list:
        result.append(cv2.resize(img, (size, size)))
    
    cur = len(result)
    if cur == 0:
        return result
    
    for _ in range(target_per_class * 3):
        if len(result) >= target_per_class:
            break
        img = result[np.random.randint(0, cur)].copy()
        aug_type = np.random.randint(0, 7)
        
        if aug_type == 0:  # Translation
            M = np.float32([[1, 0, np.random.randint(-3, 4)],
                            [0, 1, np.random.randint(-3, 4)]])
            img = cv2.warpAffine(img, M, (size, size), borderValue=0)
        elif aug_type == 1:  # Rotation
            M = cv2.getRotationMatrix2D((size/2, size/2), np.random.uniform(-12, 12), 1.0)
            img = cv2.warpAffine(img, M, (size, size), borderValue=0)
        elif aug_type == 2:  # Scale
            s = np.random.uniform(0.8, 1.2)
            M = cv2.getRotationMatrix2D((size/2, size/2), 0, s)
            img = cv2.warpAffine(img, M, (size, size), borderValue=0)
        elif aug_type == 3:  # Erosion
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img = cv2.erode(img, k, iterations=1)
        elif aug_type == 4:  # Dilation
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img = cv2.dilate(img, k, iterations=1)
        elif aug_type == 5:  # Noise
            noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
            img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        else:  # Perspective warp
            pts1 = np.float32([[0,0],[size,0],[0,size],[size,size]])
            pts2 = pts1 + np.random.randint(-3, 4, (4, 2)).astype(np.float32)
            M = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(img, M, (size, size), borderValue=0)
        
        result.append(img)
    
    return result[:target_per_class]


# --- CNN MODEL (PyTorch) ---

def build_cnn(num_classes=27):
    """Build a small CNN for character classification.
    
    Architecture:
    - 2 Conv blocks (Conv + BN + ReLU + MaxPool)
    - 2 FC layers with dropout
    - Output: 27 classes (valid chars)
    """
    import torch
    import torch.nn as nn
    
    class CaptchaCNN(nn.Module):
        def __init__(self, num_classes=27):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: 1x28x28 -> 32x13x13
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.1),
                # Block 2: 32x13x13 -> 64x6x6
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.2),
                # Block 3: 64x6x6 -> 128x3x3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 3 * 3, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    return CaptchaCNN(num_classes)


def train_cnn(char_images_dict, epochs=50, batch_size=64, lr=0.001):
    """Train CNN on augmented character data.
    
    Args:
        char_images_dict: {char: [binary_images]}
    
    Returns:
        trained model
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import cv2
    
    SIZE = 28
    
    # Augment and prepare data
    print("Augmenting training data...")
    X = []
    y = []
    for ch, imgs in char_images_dict.items():
        aug = augment_char_images(imgs, target_per_class=200, size=SIZE)
        for img in aug:
            resized = cv2.resize(img, (SIZE, SIZE))
            X.append(resized.astype(np.float32) / 255.0)
            y.append(CHAR_TO_IDX[ch])
    
    X = np.array(X, dtype=np.float32)[:, np.newaxis, :, :]  # (N, 1, 28, 28)
    y = np.array(y, dtype=np.int64)
    
    print(f"Training data: {X.shape[0]} samples, {len(char_images_dict)} classes")
    
    # To tensors
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    # Build model
    device = torch.device('cpu')
    model = build_cnn(num_classes=len(VALID_CHARS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # Train
    print(f"Training CNN ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
        
        scheduler.step()
        acc = correct / total * 100
        avg_loss = total_loss / total
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={acc:.1f}%")
    
    print(f"Training complete! Final accuracy: {acc:.1f}%")
    return model


def predict_cnn(model, char_crops):
    """Predict characters using CNN model."""
    import torch
    import cv2
    
    SIZE = 28
    device = torch.device('cpu')
    model.eval()
    
    result = ""
    confidences = []
    
    with torch.no_grad():
        for crop in char_crops:
            resized = cv2.resize(crop, (SIZE, SIZE)).astype(np.float32) / 255.0
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(device)
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(1)
            result += IDX_TO_CHAR[pred.item()]
            confidences.append(conf.item())
    
    return result, confidences


# --- ENSEMBLE SKLEARN MODEL ---

def train_ensemble(char_images_dict):
    """Train sklearn ensemble (SVM + KNN + RandomForest) on combined features."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    import cv2
    
    SIZE = 28
    
    # Augment
    print("Augmenting for ensemble...")
    aug_data = {ch: augment_char_images(imgs, 150, SIZE) for ch, imgs in char_images_dict.items()}
    
    # Extract features
    X = []
    y = []
    for ch, imgs in aug_data.items():
        for img in imgs:
            X.append(extract_features(img, SIZE))
            y.append(ch)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Ensemble
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    ensemble = VotingClassifier([('svm', svm), ('knn', knn), ('rf', rf)], voting='soft')
    
    print("Training ensemble (SVM + KNN + RF)...")
    ensemble.fit(X_scaled, y)
    print("Ensemble trained!")
    
    return scaler, ensemble


def predict_ensemble(scaler, ensemble, char_crops):
    """Predict using sklearn ensemble."""
    result = ""
    for crop in char_crops:
        feat = extract_features(crop).reshape(1, -1)
        feat_s = scaler.transform(feat)
        pred = ensemble.predict(feat_s)[0]
        result += pred
    return result


# --- FULL PIPELINE ---

def solve_captcha(img_bgr, model_data=None):
    """Solve a captcha image using trained model.
    
    Args:
        img_bgr: BGR image (numpy array from cv2.imread)
        model_data: dict with 'cnn_model', 'scaler', 'ensemble' etc.
    
    Returns:
        (predicted_text, confidence_or_None)
    """
    binary = preprocess_captcha(img_bgr)
    char_crops = segment_chars(binary)
    
    if model_data is None:
        return None, None
    
    # Try CNN first (faster + better)
    if model_data.get('cnn_model') is not None:
        text, confs = predict_cnn(model_data['cnn_model'], char_crops)
        avg_conf = np.mean(confs)
        
        # If confidence is low, try ensemble as fallback
        if avg_conf < 0.7 and model_data.get('ensemble') is not None:
            ens_text = predict_ensemble(model_data['scaler'], model_data['ensemble'], char_crops)
            # If both agree, high confidence
            if text == ens_text:
                return text, avg_conf
            # If disagree, use the one with more "common" chars
            # Or prefer CNN if confidence > 0.5
            if avg_conf >= 0.5:
                return text, avg_conf
            return ens_text, 0.5
        
        return text, avg_conf
    
    # Fallback to ensemble only
    if model_data.get('ensemble') is not None:
        text = predict_ensemble(model_data['scaler'], model_data['ensemble'], char_crops)
        return text, None
    
    return None, None


def load_model():
    """Load trained model from disk."""
    if not MODEL_FILE.exists():
        return None
    
    import torch
    
    model_data = pickle.loads(MODEL_FILE.read_bytes())
    
    # Rebuild CNN from state dict
    if model_data.get('cnn_state_dict') is not None:
        cnn = build_cnn(num_classes=len(VALID_CHARS))
        cnn.load_state_dict(model_data['cnn_state_dict'])
        cnn.eval()
        model_data['cnn_model'] = cnn
    else:
        model_data['cnn_model'] = None
    
    return model_data


def save_model(cnn_model=None, scaler=None, ensemble=None, char_distribution=None):
    """Save model to disk."""
    import torch
    
    model_data = {
        'cnn_state_dict': cnn_model.state_dict() if cnn_model is not None else None,
        'scaler': scaler,
        'ensemble': ensemble,
        'char_distribution': char_distribution or {},
        'valid_chars': VALID_CHARS,
        'version': '2.0',
    }
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {MODEL_FILE}")


# --- DATA LOADING ---

def load_training_data():
    """Load captcha images + labels from captcha_results.json."""
    import cv2
    
    if not RESULTS_FILE.exists():
        print(f"[ERROR] {RESULTS_FILE} not found!")
        print("Run with --count N --fetch-only first to collect training data.")
        sys.exit(1)
    
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    
    char_images = {}
    for c in data['captchas']:
        if not c.get('solved_text'):
            continue
        img = cv2.imread(c['image_path'])
        if img is None:
            continue
        
        label = c['solved_text']
        binary = preprocess_captcha(img)
        char_crops = segment_chars(binary)
        
        for i, ch in enumerate(label):
            if i < len(char_crops) and ch in CHAR_TO_IDX:
                if ch not in char_images:
                    char_images[ch] = []
                char_images[ch].append(char_crops[i])
    
    print(f"Loaded {len(data['captchas'])} captchas, {sum(len(v) for v in char_images.values())} char samples, "
          f"{len(char_images)} unique chars")
    return char_images


def load_results():
    if RESULTS_FILE.exists():
        try:
            return json.loads(RESULTS_FILE.read_text())
        except:
            pass
    return {"metadata": {}, "captchas": []}


def save_results(results):
    results["metadata"]["last_updated"] = datetime.now().isoformat()
    results["metadata"]["total_captchas"] = len(results["captchas"])
    results["metadata"]["solved_count"] = sum(1 for c in results["captchas"] if c.get("solved_text"))
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))


# --- API FETCH ---

def fetch_captcha(session):
    """Fetch a new captcha from BPS DTSEN API."""
    try:
        resp = session.get(CAPTCHA_API_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        captcha_key = data["captcha"]["key"]
        img_b64 = data["captcha"]["img"]
        raw_b64 = img_b64.split(",", 1)[1] if "," in img_b64 else img_b64
        img_bytes = base64.b64decode(raw_b64)
        return captcha_key, img_bytes
    except Exception as e:
        print(f"  [ERROR] Fetch gagal: {e}")
        return None, None


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="BPS DTSEN Captcha Solver - Pure Python")
    parser.add_argument("--count", type=int, default=20, help="Jumlah captcha di-scrape")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay antar request (detik)")
    parser.add_argument("--train", action="store_true", help="Train/retrain model dari data yang ada")
    parser.add_argument("--eval", action="store_true", help="Evaluasi model pada data training")
    parser.add_argument("--image", help="Solve satu file gambar captcha")
    parser.add_argument("--fetch-only", action="store_true", help="Hanya fetch captcha, jangan solve")
    parser.add_argument("--solve-only", action="store_true", help="Solve captcha yang sudah di-fetch")
    parser.add_argument("--epochs", type=int, default=50, help="CNN training epochs")
    parser.add_argument("--no-cnn", action="store_true", help="Skip CNN, hanya pakai ensemble sklearn")
    args = parser.parse_args()
    
    # Ensure directories exist
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- TRAIN MODE ---
    if args.train:
        print("\n" + "=" * 60)
        print("  TRAINING CAPTCHA SOLVER")
        print("=" * 60 + "\n")
        
        char_images = load_training_data()
        
        if len(char_images) < 5:
            print("[ERROR] Data training terlalu sedikit! Minimal 5 karakter berbeda.")
            print("Jalankan --count 50 --fetch-only lalu solve manual untuk mengumpulkan data.")
            sys.exit(1)
        
        cnn_model = None
        scaler = None
        ensemble = None
        
        # Train CNN
        if not args.no_cnn:
            try:
                import torch
                print("\n--- Training CNN (PyTorch) ---")
                cnn_model = train_cnn(char_images, epochs=args.epochs)
                print("CNN trained!")
            except ImportError:
                print("[WARN] PyTorch not installed, skipping CNN. Install: pip install torch")
            except Exception as e:
                print(f"[WARN] CNN training failed: {e}")
        
        # Train ensemble
        print("\n--- Training Ensemble (sklearn) ---")
        scaler, ensemble = train_ensemble(char_images)
        
        # Save
        char_dist = {ch: len(imgs) for ch, imgs in char_images.items()}
        save_model(cnn_model=cnn_model, scaler=scaler, ensemble=ensemble, char_distribution=char_dist)
        
        # Evaluate
        print("\n--- Evaluation on Training Data ---")
        model_data = load_model()
        correct = 0
        total = 0
        for c in json.loads(RESULTS_FILE.read_text())["captchas"]:
            if not c.get('solved_text'):
                continue
            import cv2
            img = cv2.imread(c['image_path'])
            if img is None:
                continue
            label = c['solved_text']
            pred, conf = solve_captcha(img, model_data)
            ok = pred == label
            if ok:
                correct += 1
            total += 1
            print(f"  {'OK' if ok else 'MISS'}: expected={label}, got={pred}" + 
                  (f" (conf={conf:.2f})" if conf else ""))
        
        print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)" if total > 0 else "No data to eval")
        return
    
    # --- EVAL MODE ---
    if args.eval:
        model_data = load_model()
        if model_data is None:
            print("[ERROR] Model belum di-train! Jalankan --train dulu.")
            sys.exit(1)
        
        import cv2
        print("\n--- Evaluating on Training Data ---")
        data = json.loads(RESULTS_FILE.read_text())
        correct = 0
        char_correct = 0
        char_total = 0
        total = 0
        
        for c in data["captchas"]:
            if not c.get('solved_text'):
                continue
            img = cv2.imread(c['image_path'])
            if img is None:
                continue
            label = c['solved_text']
            pred, conf = solve_captcha(img, model_data)
            ok = pred == label
            if ok:
                correct += 1
            total += 1
            
            for i in range(min(len(pred), len(label))):
                char_total += 1
                if pred[i] == label[i]:
                    char_correct += 1
            
            print(f"  {'OK' if ok else 'MISS'}: {label} -> {pred}" + 
                  (f" (conf={conf:.2f})" if conf else ""))
        
        print(f"\nFull captcha: {correct}/{total} ({correct/total*100:.1f}%)" if total > 0 else "No data")
        print(f"Char-level: {char_correct}/{char_total} ({char_correct/char_total*100:.1f}%)" if char_total > 0 else "")
        return
    
    # --- SINGLE IMAGE MODE ---
    if args.image:
        import cv2
        model_data = load_model()
        if model_data is None:
            print("[ERROR] Model belum di-train! Jalankan --train dulu.")
            sys.exit(1)
        
        img = cv2.imread(args.image)
        if img is None:
            print(f"[ERROR] Tidak bisa baca file: {args.image}")
            sys.exit(1)
        
        pred, conf = solve_captcha(img, model_data)
        print(f"Result: {pred}" + (f" (confidence={conf:.2f})" if conf else ""))
        return
    
    # --- SOLVE ONLY MODE ---
    if args.solve_only:
        model_data = load_model()
        if model_data is None:
            print("[ERROR] Model belum di-train! Jalankan --train dulu.")
            sys.exit(1)
        
        import cv2
        results = load_results()
        unsolved = [c for c in results["captchas"] if not c.get("solved_text")]
        
        if not unsolved:
            print("Semua captcha sudah di-solve!")
            return
        
        print(f"Solving {len(unsolved)} captcha...")
        solved = 0
        for c in unsolved:
            img = cv2.imread(c['image_path'])
            if img is None:
                continue
            pred, conf = solve_captcha(img, model_data)
            if pred:
                c['solved_text'] = pred
                c['solver'] = 'cnn_ensemble_local'
                c['confidence'] = conf
                solved += 1
                print(f"  [{c['image_filename']}] -> {pred}" + 
                      (f" (conf={conf:.2f})" if conf else ""))
            time.sleep(0.1)
        
        save_results(results)
        print(f"\nDone! {solved}/{len(unsolved)} solved.")
        return
    
    # --- FETCH + SOLVE MODE ---
    import cv2
    
    model_data = None
    if not args.fetch_only:
        model_data = load_model()
        if model_data is None:
            print("[WARN] Model belum di-train! Jalankan --train dulu.")
            print("Lanjutkan dengan --fetch-only untuk mengumpulkan data.")
            sys.exit(1)
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://dtsen-form.web.bps.go.id",
        "Referer": "https://dtsen-form.web.bps.go.id/"
    })
    
    results = load_results()
    start_idx = len(results["captchas"])
    
    print(f"\n{'=' * 60}")
    print(f"  BPS DTSEN Captcha Solver")
    print(f"  Mode: {'fetch-only' if args.fetch_only else 'fetch + solve (LOCAL ML)'}")
    print(f"  Target: {args.count} captchas (delay {args.delay}s)")
    print(f"{'=' * 60}\n")
    
    success = failed = 0
    
    for i in range(args.count):
        idx = start_idx + i
        print(f"[{idx+1}/{start_idx+args.count}] Fetching...", end=" ", flush=True)
        
        captcha_key, img_bytes = fetch_captcha(session)
        if not captcha_key:
            failed += 1
            print("GAGAL")
            time.sleep(args.delay * 2)
            continue
        
        # Save image
        filename = f"captcha_{idx:04d}.jpg"
        filepath = IMAGES_DIR / filename
        filepath.write_bytes(img_bytes)
        
        solved_text = None
        confidence = None
        
        if not args.fetch_only and model_data:
            print("solving...", end=" ", flush=True)
            img_bgr = cv2.imread(str(filepath))
            solved_text, confidence = solve_captcha(img_bgr, model_data)
            
            if solved_text:
                success += 1
                conf_str = f" (conf={confidence:.2f})" if confidence else ""
                print(f"-> [{solved_text}]{conf_str}")
            else:
                failed += 1
                print("-> GAGAL")
        else:
            print(f"OK -> {filename}")
        
        results["captchas"].append({
            "index": idx,
            "captcha_key": captcha_key,
            "solved_text": solved_text,
            "image_path": str(filepath),
            "image_filename": filename,
            "timestamp": datetime.now().isoformat(),
            "solver": "cnn_ensemble_local" if solved_text else None,
            "confidence": confidence,
        })
        
        if (i + 1) % 5 == 0:
            save_results(results)
        
        if i < args.count - 1:
            time.sleep(args.delay)
    
    save_results(results)
    
    print(f"\n{'=' * 60}")
    print(f"  SELESAI!")
    if not args.fetch_only:
        print(f"  Berhasil solve: {success}/{args.count}")
        print(f"  Gagal: {failed}/{args.count}")
    print(f"  Total tersimpan: {len(results['captchas'])}")
    print(f"  JSON: {RESULTS_FILE}")
    print(f"  Images: {IMAGES_DIR}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
