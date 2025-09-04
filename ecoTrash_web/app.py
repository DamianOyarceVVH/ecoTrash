import os
import io
import re
import json
import base64
import logging
import unicodedata
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

# Preferir tflite_runtime si existe, sino tensorflow
Interpreter = None
try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except Exception:
    try:
        import tensorflow as tf  # type: ignore
        Interpreter = tf.lite.Interpreter
    except Exception:
        raise SystemExit("Instala 'tensorflow' o 'tflite-runtime' para usar TFLite.")

# OpenCV opcional para /capture
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from flask import Flask, request, jsonify, render_template, Response

# -------------------------
# Configuración paths / app
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
STATIC_FOTOS = os.path.join(STATIC_DIR, "fotos")
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")

os.makedirs(STATIC_FOTOS, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

MODEL_PATH_CANDIDATES = [
    os.path.join(MODELOS_DIR, "model.tflite"),
    os.path.join(BASE_DIR, "model.tflite"),
]
LABELS_PATH_CANDIDATES = [
    os.path.join(MODELOS_DIR, "labels.txt"),
    os.path.join(BASE_DIR, "labels.txt"),
]

LAST_FILENAME = "foto.jpg"
LAST_PATH = os.path.join(STATIC_FOTOS, LAST_FILENAME)

# Threshold para considerar UNKNOWN (si confidence < threshold -> "No reciclable")
UNKNOWN_THRESHOLD = float(os.getenv("UNKNOWN_THRESHOLD", "0.80"))

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ecoTrash")

# -------------------------
# Utilidades
# -------------------------
def normalize_label(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s

def pick_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def downscale(img: Image.Image, max_w: int = 1280, max_h: int = 720) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    if w <= max_w and h <= max_h:
        return img
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return img

def atomic_save_jpeg(pil_img: Image.Image, final_path: str, quality: int = 90) -> None:
    tmp_path = final_path + ".tmp"
    pil_img.save(tmp_path, format="JPEG", quality=quality)
    os.replace(tmp_path, final_path)

def windows_relpath(path: str) -> str:
    rel = os.path.relpath(path, BASE_DIR)
    return rel.replace("/", "\\")

def is_image_mimetype(mtype: str) -> bool:
    return mtype and mtype.lower().startswith("image/")

# -------------------------
# Cargar labels y modelo
# -------------------------
LABELS_PATH = pick_first_existing(LABELS_PATH_CANDIDATES)
labels_original, labels_norm = [], []
if LABELS_PATH:
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                l = line.strip()
                if not l:
                    continue
                parts = l.split()
                if parts and parts[0].isdigit():
                    l = " ".join(parts[1:])
                labels_original.append(l)
                labels_norm.append(normalize_label(l))
        log.info("Etiquetas cargadas (%d): %s", len(labels_original), labels_original)
    except Exception:
        log.exception("Error cargando labels.txt")
else:
    log.warning("No se encontró labels.txt; se usará índice de clases numérico.")

MODEL_PATH = pick_first_existing(MODEL_PATH_CANDIDATES)
if not MODEL_PATH:
    raise SystemExit("No se encontró model.tflite. Coloca model.tflite en ./modelos/ o ./model.tflite")

log.info("Cargando modelo TFLite desde: %s", MODEL_PATH)
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

log.info("Input details: %s", input_details[0]["shape"])
log.info("Output details: %s", output_details[0]["shape"])

# Contenedores y nombres de despliegue
contenedores = {
    "plastico": ("Plástico", "#3B82F6"),
    "papel": ("Papel", "#FACC15"),
    "metal": ("Metal", "#EF4444"),
    "organico": ("Orgánico", "#22C55E"),
    "no_reciclable": ("No reciclable", "#6B7280"),
}
display_name_map = {
    "plastico": "Botella de plástico",
    "papel": "Caja de leche",
    "metal": "Lata de bebida",
    "organico": "Tomate",
    "no_reciclable": "No reciclable"
}

# -------------------------
# Pre / post procesado del modelo
# -------------------------
def to_model_input(img: Image.Image, input_meta: dict) -> np.ndarray:
    img = ensure_rgb(img)
    shape = input_meta["shape"]
    if len(shape) == 4:
        h, w = int(shape[1]), int(shape[2])
    elif len(shape) == 3:
        h, w = int(shape[0]), int(shape[1])
    else:
        raise ValueError(f"Input shape no soportada: {shape}")
    img = img.resize((w, h), Image.LANCZOS)
    arr = np.asarray(img)
    dtype = np.dtype(input_meta["dtype"])
    if np.issubdtype(dtype, np.floating):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(dtype)
    arr = np.expand_dims(arr, axis=0)
    return arr

def extract_scores(raw_output: np.ndarray, output_meta: dict, labels_count: int) -> np.ndarray:
    out_qscale, out_qzero = output_meta.get("quantization", (0.0, 0))
    arr = np.array(raw_output)
    if out_qscale and out_qscale != 0:
        arr = (arr.astype(np.float32) - out_qzero) * out_qscale
    else:
        arr = arr.astype(np.float32)
    if arr.ndim == 1:
        scores = arr
    elif arr.ndim == 2 and arr.shape[0] == 1:
        scores = arr[0]
    else:
        class_axis = None
        for ax in range(arr.ndim - 1, -1, -1):
            if arr.shape[ax] in (labels_count, labels_count + 1):
                class_axis = ax
                break
        if class_axis is None:
            scores = arr.flatten()
        else:
            if class_axis != arr.ndim - 1:
                arr = np.moveaxis(arr, class_axis, -1)
            # promedio sobre ejes espaciales para robustez
            scores = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    # si parecen logits -> softmax
    if scores.max() > 1.01 or scores.min() < 0:
        ex = np.exp(scores - np.max(scores))
        scores = ex / np.clip(ex.sum(), 1e-8, None)
    if scores.shape[0] < labels_count:
        pad = np.zeros(labels_count, dtype=np.float32)
        pad[: scores.shape[0]] = scores
        scores = pad
    else:
        scores = scores[:labels_count]
    return np.clip(scores.astype(np.float32), 0.0, 1.0)

# -------------------------
# Pipeline: guardar -> inferir -> responder
# -------------------------
def run_pipeline(pil_img: Image.Image, save_as_last: bool = True) -> dict:
    pil_img = ensure_rgb(pil_img)
    pil_img = downscale(pil_img)

    if save_as_last:
        final_path = LAST_PATH
    else:
        now = datetime.now()
        final_path = os.path.join(STATIC_FOTOS, f"{now.strftime('%Y%m%d_%H%M%S')}.jpg")

    atomic_save_jpeg(pil_img, final_path, quality=90)

    # inferencia
    model_input = to_model_input(pil_img, input_details[0])
    try:
        interpreter.set_tensor(input_details[0]["index"], model_input)
    except Exception:
        model_input = model_input.astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], model_input)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]["index"])

    labels_count = max(1, len(labels_original)) if labels_original else 4
    scores = extract_scores(raw_output, output_details[0], labels_count)
    idx = int(np.argmax(scores))
    conf = float(scores[idx])

    if labels_original and idx < len(labels_original):
        raw_label = labels_original[idx]
        label_key = labels_norm[idx]
    else:
        raw_label = f"clase_{idx}"
        label_key = normalize_label(raw_label)

    is_unknown = False
    if conf < UNKNOWN_THRESHOLD:
        log.info("Confianza %.4f < threshold %.4f -> tratado como NO RECONOCIDO", conf, UNKNOWN_THRESHOLD)
        label_key = "no_reciclable"
        raw_label = "No reciclable"
        is_unknown = True

    display_name = display_name_map.get(label_key, raw_label)
    contenedor, color = contenedores.get(label_key, ("--", "#9CA3AF"))

    payload = {
        "material": display_name,
        "contenedor": contenedor,
        "confidence": round(conf, 4),
        "image_path": windows_relpath(final_path),
        "color": color,
        "is_unknown": is_unknown
    }
    log.info("Resultado: %s", json.dumps(payload, ensure_ascii=False))
    return payload

# -------------------------
# Rutas HTTP
# -------------------------
@app.route("/")
def index():
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(template_path):
        return render_template("index.html")
    return Response("<h2>ecoTrash</h2><p>Coloca templates/index.html</p>", mimetype="text/html")

@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload():
    try:
        pil_img = None
        data_url = request.form.get("imageBase64")
        if data_url:
            if "," not in data_url:
                return jsonify({"error": "imageBase64 malformado"}), 400
            header, encoded = data_url.split(",", 1)
            if not header.lower().startswith("data:image/"):
                return jsonify({"error": "imageBase64 no es imagen"}), 400
            missing = len(encoded) % 4
            if missing:
                encoded += "=" * (4 - missing)
            try:
                binary = io.BytesIO(base64.b64decode(encoded))
            except Exception as e:
                log.exception("Error decodificando base64")
                return jsonify({"error": "imageBase64 inválido", "detail": str(e)}), 400
            pil_img = Image.open(binary)
        if pil_img is None:
            file = request.files.get("image") or request.files.get("photo")
            if not file:
                return jsonify({"error": "No se recibió 'imageBase64' ni archivo 'image/photo'"}), 400
            if not is_image_mimetype(file.mimetype or "") and not re.search(r"\.(jpe?g|png|bmp|webp|tif?f)$", file.filename or "", re.I):
                return jsonify({"error": "Archivo no es imagen válida"}), 400
            buf = io.BytesIO(file.read())
            buf.seek(0)
            pil_img = Image.open(buf)

        # verificar/reabrir imagen para evitar problemas PIL
        try:
            pil_img.verify()
            # reabrir
            if 'binary' in locals():
                pil_img = Image.open(io.BytesIO(binary.getvalue()))
            else:
                buf.seek(0)
                pil_img = Image.open(buf)
            pil_img.load()
        except Exception:
            # fallback: reabrir stream
            try:
                if 'file' in locals():
                    file.stream.seek(0)
                    buf2 = io.BytesIO(file.stream.read())
                    buf2.seek(0)
                    pil_img = Image.open(buf2)
                    pil_img.load()
                else:
                    return jsonify({"error": "No se pudo verificar la imagen enviada"}), 400
            except Exception as e:
                log.exception("Error reabriendo imagen")
                return jsonify({"error": "Imagen inválida", "detail": str(e)}), 400

        result = run_pipeline(pil_img, save_as_last=True)
        return jsonify(result), 200

    except Exception as e:
        log.exception("Error en /upload")
        return jsonify({"error": "Error al procesar la imagen", "detail": str(e)}), 500

@app.route("/capture", methods=["GET","POST"])
def capture():
    """Captura desde la webcam del servidor (solo si OpenCV está instalado)."""
    if cv2 is None:
        return jsonify({"error": "OpenCV no instalado. Instala opencv-python si quieres /capture."}), 500
    try:
        w = int(request.args.get("w", 640))
        h = int(request.args.get("h", 480))
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        try:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ok, frame = cam.read()
            if not ok:
                return jsonify({"error": "No se pudo leer de la cámara"}), 500
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            result = run_pipeline(pil_img, save_as_last=True)
            return jsonify(result), 200
        finally:
            cam.release()
    except Exception as e:
        log.exception("Error en /capture")
        return jsonify({"error": "Error al capturar la imagen", "detail": str(e)}), 500

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ecoTrash - servidor simplificado (sin Raspberry/TCP)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    log.info("Iniciando servidor en %s:%d (UNKNOWN_THRESHOLD=%s)", args.host, args.port, UNKNOWN_THRESHOLD)
    app.run(host=args.host, port=args.port, debug=args.debug)
