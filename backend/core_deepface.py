"""
Core image analysis.
- Prefers a custom multitask Keras model (if available)
- Falls back to DeepFace for emotion/race/age/gender
"""
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image
import numpy as np
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    import torch  # type: ignore
    _CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    torch = None
    _CUDA_AVAILABLE = False

try:
    from facenet_pytorch import MTCNN  # type: ignore
except Exception:
    MTCNN = None

_CUSTOM_MODEL = None
_CUSTOM_META: Dict[str, Any] = None
_MTCNN_DETECTOR = None


def _get_mtcnn():
    """Instantiate MTCNN once; prefer CUDA if available."""
    global _MTCNN_DETECTOR
    if _MTCNN_DETECTOR is not None:
        return _MTCNN_DETECTOR
    if MTCNN is None:
        return None
    device = 'cuda' if _CUDA_AVAILABLE else 'cpu'
    try:
        _MTCNN_DETECTOR = MTCNN(keep_all=True, device=device)
    except Exception:
        _MTCNN_DETECTOR = None
    return _MTCNN_DETECTOR


def _load_custom_model() -> Tuple[Any, Dict[str, Any]]:
    """Load cached custom Keras model if present."""
    global _CUSTOM_MODEL, _CUSTOM_META
    if _CUSTOM_MODEL is not None and _CUSTOM_META is not None:
        return _CUSTOM_MODEL, _CUSTOM_META

    model_path = Path(os.getenv("CUSTOM_MODEL_PATH", "models/multitask_model.keras"))
    meta_path = Path(os.getenv("CUSTOM_MODEL_META", "models/multitask_meta.json"))
    if not model_path.exists() or not meta_path.exists():
        return None, None

    try:
        import tensorflow as tf

        _CUSTOM_MODEL = tf.keras.models.load_model(model_path)
        with meta_path.open("r", encoding="utf-8") as f:
            _CUSTOM_META = json.load(f)
        return _CUSTOM_MODEL, _CUSTOM_META
    except Exception as e:
        print(f"Custom model load error: {e}")
        return None, None


def _extract_largest_face(img_array: np.ndarray):
    """Try GPU-accelerated MTCNN first, fall back to DeepFace extractor."""
    # Try facenet-pytorch MTCNN
    detector = _get_mtcnn()
    if detector is not None:
        try:
            boxes, _ = detector.detect(img_array)
            if boxes is not None and len(boxes) > 0:
                areas = [(i, (box[2] - box[0]) * (box[3] - box[1])) for i, box in enumerate(boxes)]
                idx = max(areas, key=lambda x: x[1])[0]
                x1, y1, x2, y2 = boxes[idx]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, img_array.shape[1]-1), min(y2, img_array.shape[0]-1)
                face_crop = img_array[y1:y2, x1:x2]
                return face_crop, {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1}
        except Exception:
            pass

    # Fallback to DeepFace
    try:
        from deepface import DeepFace

        faces = DeepFace.extract_faces(
            img_array,
            detector_backend='opencv',
            enforce_detection=True
        )
        if isinstance(faces, list) and faces:
            best = None
            best_area = -1
            for f in faces:
                area = f.get('facial_area') or {}
                w = area.get('w', 0)
                h = area.get('h', 0)
                a = w * h
                if a > best_area:
                    best_area = a
                    best = f
            if best and 'face' in best:
                return best['face'], best.get('facial_area')
    except Exception:
        return None, None
    return None, None


def _quality_checks(img_array: np.ndarray) -> Dict[str, Any]:
    quality = {"low_quality": False, "reason": ""}
    if cv2 is not None:
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if variance < 50.0:
                quality["low_quality"] = True
                quality["reason"] = "Image is too blurry; use a sharper photo."
        except Exception:
            pass
    return quality


def _age_band(age: int) -> str:
    band = 'Adult'
    if age < 2:
        band = 'Infant'
    elif age < 13:
        band = 'Child'
    elif age < 18:
        band = 'Teen'
    elif age > 60:
        band = 'Senior'
    return band


def _analyze_custom(image_bytes: bytes) -> Dict[str, Any]:
    model, meta = _load_custom_model()
    if not model or not meta:
        return None

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    w, h = img.size
    max_side = max(w, h)
    if max_side > 640:
        scale = 640.0 / max_side
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    img_array = np.array(img)

    quality = _quality_checks(img_array)
    face_crop, facial_area = _extract_largest_face(img_array)
    if facial_area and (facial_area.get('w', 0) < 60 or facial_area.get('h', 0) < 60):
        quality["low_quality"] = True
        extra = "Detected face too small; upload a closer photo."
        quality["reason"] = quality["reason"] + (" " if quality["reason"] else "") + extra

    target = face_crop if face_crop is not None else img_array
    target_size = meta.get('image_size', 224)
    if cv2 is not None:
        target = cv2.resize(target, (target_size, target_size))
    else:
        target = np.array(Image.fromarray(target).resize((target_size, target_size)))
    target = target.astype(np.float32) / 255.0
    batch = np.expand_dims(target, axis=0)

    preds = model(batch, training=False)
    age_val = float(preds['age'][0][0])
    gender_prob = float(preds['gender'][0][0])
    race_probs = preds['race'][0].numpy() if hasattr(preds['race'][0], 'numpy') else preds['race'][0]
    emo_probs = preds['emotion'][0].numpy() if hasattr(preds['emotion'][0], 'numpy') else preds['emotion'][0]

    race_labels = meta.get('race_labels') or []
    emo_labels = meta.get('emotion_labels') or []

    race_idx = int(np.argmax(race_probs)) if len(race_probs) else 0
    emo_idx = int(np.argmax(emo_probs)) if len(emo_probs) else 0

    race_dict = {race_labels[i]: float(p * 100.0) for i, p in enumerate(race_probs)} if race_labels else {}
    emo_dict = {emo_labels[i]: float(p * 100.0) for i, p in enumerate(emo_probs)} if emo_labels else {}

    low_conf = False
    if race_dict:
        low_conf = low_conf or max(race_dict.values()) < 40.0
    if emo_dict:
        low_conf = low_conf or max(emo_dict.values()) < 40.0

    return {
        'age': int(round(age_val)),
        'age_band': _age_band(int(round(age_val))),
        'gender': 'Woman' if gender_prob >= 0.5 else 'Man',
        'dominant_emotion': emo_labels[emo_idx] if emo_labels else 'neutral',
        'emotion': emo_dict,
        'dominant_race': race_labels[race_idx] if race_labels else 'unknown',
        'race': race_dict,
        'low_quality': quality['low_quality'],
        'quality_reason': quality['reason'],
        'low_confidence': low_conf,
        'facial_area': facial_area,
    }


def _analyze_deepface(image_bytes: bytes) -> Dict[str, Any]:
    from deepface import DeepFace

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    w, h = img.size
    max_side = max(w, h)
    if max_side > 640:
        scale = 640.0 / max_side
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    img_array = np.array(img)

    quality = _quality_checks(img_array)

    face_crop, facial_area = _extract_largest_face(img_array)
    if facial_area and (facial_area.get('w', 0) < 60 or facial_area.get('h', 0) < 60):
        quality["low_quality"] = True
        extra = "Detected face too small; upload a closer photo."
        quality["reason"] = quality["reason"] + (" " if quality["reason"] else "") + extra

    target = face_crop if face_crop is not None else img_array
    try:
        result = DeepFace.analyze(
            target,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=False,
            silent=True
        )
    except Exception:
        result = DeepFace.analyze(
            target,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )

    if isinstance(result, list):
        result = result[0]

    age = int(result.get('age', 30))
    emotion = {k: float(v) for k, v in result.get('emotion', {}).items()}
    race = {k: float(v) for k, v in result.get('race', {}).items()}
    emo_max = max(emotion.values()) if emotion else 0.0
    race_max = max(race.values()) if race else 0.0
    low_conf = (emo_max < 40.0) or (race_max < 40.0)

    return {
        'age': age,
        'age_band': _age_band(age),
        'gender': result.get('dominant_gender', 'Unknown'),
        'dominant_emotion': result.get('dominant_emotion', 'neutral'),
        'emotion': emotion,
        'dominant_race': result.get('dominant_race', 'unknown'),
        'race': race,
        'low_quality': quality['low_quality'],
        'quality_reason': quality['reason'],
        'low_confidence': low_conf,
        'facial_area': facial_area
    }


def analyze_image(image_bytes: bytes, use_mock: bool = False) -> Dict[str, Any]:
    """Analyze an image using a custom model when available, otherwise DeepFace."""
    try:
        if os.getenv("USE_CUSTOM_MODEL", "1") == "1":
            custom = _analyze_custom(image_bytes)
            if custom:
                return custom
    except Exception as e:
        print(f"Custom model error: {e}")

    try:
        return _analyze_deepface(image_bytes)
    except Exception as e:
        print(f"DeepFace error: {e}")
        return {
            'error': str(e),
            'age': 30,
            'gender': 'Unknown',
            'dominant_emotion': 'neutral',
            'dominant_race': 'unknown'
        }
