import os
import io
import re
import uuid
import json
import time
import socket
import base64
import logging
import threading
import unicodedata
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

# Preferir tflite_runtime si existe
Interpreter = None
try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except Exception:
    try:
        import tensorflow as tf  # type: ignore
        Interpreter = tf.lite.Interpreter
    except Exception:
        raise SystemExit("Instala 'tensorflow' o 'tflite-runtime' (ver README).")

# OpenCV (para fallback capture)
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from flask import Flask, request, jsonify, render_template, Response
from flask_socketio import SocketIO

# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
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

# Guardar siempre con este nombre (se reemplaza)
LAST_FILENAME = "foto.jpg"
LAST_PATH = os.path.join(STATIC_FOTOS, LAST_FILENAME)

# límites / resoluciones
MAX_W, MAX_H = 1280, 720
CAPTURE_W, CAPTURE_H = 640, 480

# Threshold para considerar UNKNOWN (configurable por variable de entorno)
UNKNOWN_THRESHOLD = float(os.getenv("UNKNOWN_THRESHOLD", "0.98"))

# TCP listener para Raspberry
TCP_LISTENER_PORT = 6000

# SocketIO: forzamos threading en Windows (no usar eventlet aquí)
async_mode = "threading"
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ecoTrash")

# Lock para acceso concurrente al intérprete
infer_lock = threading.Lock()

# Contador de clientes conectados
connected_clients = set()
clients_lock = threading.Lock()

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

def downscale(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
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
# Cargar modelo y labels
# -------------------------
LABELS_PATH = pick_first_existing(LABELS_PATH_CANDIDATES)
labels_original, labels_norm = [], []
if LABELS_PATH:
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if parts and parts[0].isdigit():
                    line = " ".join(parts[1:])
                labels_original.append(line)
                labels_norm.append(normalize_label(line))
        log.info("Etiquetas cargadas (%d).", len(labels_original))
    except Exception:
        log.exception("Error cargando labels.txt")
else:
    log.warning("No se encontró labels.txt; continuará sin etiquetas humanas.")

MODEL_PATH = pick_first_existing(MODEL_PATH_CANDIDATES)
if not MODEL_PATH:
    raise SystemExit("No se encontró model.tflite. Coloca model.tflite en ./modelos/ o ./model.tflite")

log.info("Cargando modelo TFLite desde: %s", MODEL_PATH)
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Contenedores (mapa simple). añadimos 'no_reciclable'
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
# Pre / post procesado modelo
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
            # Promediamos sobre spatial/otros ejes para obtener vector de clases robusto
            scores = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    # if logits scale -> softmax -> probabilities
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
# Pipeline: guardar -> inferir -> emitir
# -------------------------
def run_pipeline_and_emit(pil_img: Image.Image, save_as_last: bool = True) -> dict:
    pil_img = ensure_rgb(pil_img)
    pil_img = downscale(pil_img, MAX_W, MAX_H)

    if save_as_last:
        final_path = LAST_PATH
    else:
        now = datetime.now()
        final_path = os.path.join(STATIC_FOTOS, f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg")

    atomic_save_jpeg(pil_img, final_path, quality=90)

    # Inferencia (con lock)
    with infer_lock:
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

    # Si la confianza es menor que el umbral, tratamos como "no_reciclable"
    is_unknown = False
    if conf < UNKNOWN_THRESHOLD:
        log.info("Confianza %.4f < threshold %.4f -> marcado como NO RECONOCIDO", conf, UNKNOWN_THRESHOLD)
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

    # Emitir resultado por SocketIO (intenta emitir)
    try:
        socketio.emit("inference", payload)
    except Exception:
        log.exception("Error al emitir por SocketIO (emit inference)")
    log.info("Inferencia emitida: %s", json.dumps(payload, ensure_ascii=False))
    return payload

# -------------------------
# Rutas HTTP
# -------------------------
@app.route("/")
def index():
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(template_path):
        return render_template("index.html")
    return Response(DEFAULT_INDEX_HTML, mimetype="text/html")

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
        try:
            pil_img.verify()
            try:
                if 'binary' in locals():
                    pil_img = Image.open(io.BytesIO(binary.getvalue()))
                else:
                    buf.seek(0)
                    pil_img = Image.open(buf)
                pil_img.load()
            except Exception:
                pass
        except Exception:
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

        result = run_pipeline_and_emit(pil_img, save_as_last=True)
        return jsonify(result), 200

    except Exception as e:
        log.exception("Error en /upload")
        return jsonify({"error": "Error al procesar la imagen", "detail": str(e)}), 500

@app.route("/capture", methods=["GET", "POST"])
def capture():
    if cv2 is None:
        return jsonify({"error": "OpenCV no instalado. Instala opencv-python"}), 500
    try:
        w = int(request.args.get("w", CAPTURE_W))
        h = int(request.args.get("h", CAPTURE_H))
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        try:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ok, frame = cam.read()
            if not ok:
                return jsonify({"error": "No se pudo leer de la cámara"}), 500
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            result = run_pipeline_and_emit(pil_img, save_as_last=True)
            return jsonify(result), 200
        finally:
            cam.release()
    except Exception as e:
        log.exception("Error en /capture")
        return jsonify({"error": "Error al capturar la imagen", "detail": str(e)}), 500

# -------------------------
# SocketIO handlers
# -------------------------
@socketio.on("connect")
def on_connect():
    with clients_lock:
        connected_clients.add(threading.get_ident())
    log.info("Cliente web conectado. Conexiones aprox: %d", len(connected_clients))

@socketio.on("disconnect")
def on_disconnect():
    with clients_lock:
        try:
            connected_clients.remove(threading.get_ident())
        except Exception:
            pass
    log.info("Cliente web desconectado. Conexiones aprox: %d", len(connected_clients))

# -------------------------
# TCP listener: Raspberry -> servidor
# -------------------------
def handle_tcp_connection(conn, addr):
    try:
        data = conn.recv(2048)
        if not data:
            conn.sendall(b"EMPTY")
            return
        text = data.decode(errors="ignore").strip()
        log.info("TCP desde %s: %s", addr, text)
        if text.upper().startswith("FOTO"):
            client_count = 0
            with clients_lock:
                client_count = max(0, len(connected_clients))
            if client_count > 0:
                try:
                    socketio.emit("trigger_capture", {"reason": "raspberry"})
                    conn.sendall(b"TRIGGER_SENT")
                    return
                except Exception:
                    log.exception("No se pudo emitir trigger_capture")
            if cv2 is None:
                conn.sendall(json.dumps({"error": "No hay cliente y OpenCV no instalado en servidor"}).encode())
                return
            try:
                cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
                ok, frame = cam.read()
                cam.release()
                if not ok:
                    conn.sendall(json.dumps({"error": "No se pudo capturar desde la cámara del servidor"}).encode())
                    return
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
                result = run_pipeline_and_emit(pil_img, save_as_last=True)
                conn.sendall(json.dumps(result, ensure_ascii=False).encode('utf-8'))
                return
            except Exception as e:
                log.exception("Error en fallback capture")
                conn.sendall(json.dumps({"error": "Error interno", "detail": str(e)}).encode())
                return
        else:
            conn.sendall(b"UNKNOWN_COMMAND")
    except Exception:
        log.exception("Error manejando conexión TCP")
        try:
            conn.sendall(b"ERROR")
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

def tcp_listener(host="0.0.0.0", port=TCP_LISTENER_PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(5)
    log.info("TCP listener activo en %s:%d", host, port)
    while True:
        try:
            conn, addr = s.accept()
            t = threading.Thread(target=handle_tcp_connection, args=(conn, addr), daemon=True)
            t.start()
        except Exception:
            log.exception("Error en accept del TCP listener")

# HTML fallback (mínimo)
DEFAULT_INDEX_HTML = r"""<!doctype html><html lang="es"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>ecoTrash</title></head><body><h2>ecoTrash - Servidor</h2><p>Coloca templates/index.html si quieres UI completa.</p></body></html>"""

# -------------------------
# Main
# -------------------------
def find_ssl_files(explicit_cert: Optional[str], explicit_key: Optional[str]):
    candidates = [
        (explicit_cert, explicit_key) if explicit_cert and explicit_key else (None, None),
        (os.path.join(BASE_DIR, "cert.pem"), os.path.join(BASE_DIR, "key.pem")),
        (os.path.join(BASE_DIR, "cert.key"), os.path.join(BASE_DIR, "perm.key")),
    ]
    for cert, key in candidates:
        if cert and key and os.path.exists(cert) and os.path.exists(key):
            return (cert, key)
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ecoTrash - servidor (HTTP/WS/TCP)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--tcp-port", type=int, default=TCP_LISTENER_PORT)
    parser.add_argument("--ssl", action="store_true")
    parser.add_argument("--cert")
    parser.add_argument("--key")
    args = parser.parse_args()

    t = threading.Thread(target=tcp_listener, kwargs={"host":"0.0.0.0","port":args.tcp_port}, daemon=True)
    t.start()

    ssl_context = None
    if args.ssl:
        pair = find_ssl_files(args.cert, args.key)
        if not pair:
            log.error("No se encontraron certificados TLS (pasaste --ssl).")
            raise SystemExit(1)
        ssl_context = pair
        log.info("TLS habilitado (modo explícito): %s", pair)
    else:
        auto_pair = find_ssl_files(None, None)
        if auto_pair:
            ssl_context = auto_pair
            log.info("TLS habilitado automáticamente (se detectaron certificados): %s", auto_pair)
        else:
            log.info("No se encontraron certificados TLS; servidor iniciará en HTTP.")

    log.info("Iniciando Flask/SocketIO en %s:%d (async_mode=%s) -- UNKNOWN_THRESHOLD=%s", args.host, args.port, async_mode, UNKNOWN_THRESHOLD)
    socketio.run(app, host=args.host, port=args.port, ssl_context=ssl_context)

if __name__ == "__main__":
    main()
