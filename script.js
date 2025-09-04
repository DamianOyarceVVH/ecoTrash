// script.js - Para usar junto a index.html en GitHub Pages
// Coloca aquí la carpeta `model/` (model.json + weights.bin) exportada desde Teachable Machine

const MODEL_PATH = "model/model.json"; // ruta relativa en el repo GH Pages
const VIDEO_FACING = { video: { facingMode: "environment" } };
const UNKNOWN_THRESHOLD = 0.98; // si la prob. máxima < threshold => "No reciclable"

const classes = ["plastico", "papel", "metal", "organico"]; // asegúrate que coincida con tu modelo

const materialMap = {
  "plastico": "Botella de plástico",
  "plástico": "Botella de plástico",
  "papel": "Caja de leche",
  "metal": "Lata de bebida",
  "organico": "Tomate",
  "orgánico": "Tomate",
  "no_reciclable": "No reciclable",
  "no reciclable": "No reciclable"
};

const rewardMessages = {
  "Botella de plástico": "¡Has ganado 200$!",
  "Caja de leche": "¡Has ganado 100$!",
  "Lata de bebida": "¡Has ganado 300$!",
  "Tomate": "¡Has ganado 50$!"
};

let model = null;
let inputSize = { w: 224, h: 224 }; // fallback; después intentamos leer del modelo
let video, snap, foto, statusEl;

function $(id){ return document.getElementById(id); }

async function loadModel() {
  statusEl.innerText = "Cargando modelo...";
  try {
    // Intentamos cargar como GraphModel, si falla intentamos LayersModel
    try {
      model = await tf.loadGraphModel(MODEL_PATH);
      console.log("Modelo cargado (GraphModel)");
    } catch (e) {
      model = await tf.loadLayersModel(MODEL_PATH);
      console.log("Modelo cargado (LayersModel)");
    }

    // Intentar inferir tamaño de entrada
    try {
      const inputInfo = model.inputs && model.inputs[0] && model.inputs[0].shape;
      if (inputInfo && inputInfo.length >= 3) {
        // forma puede ser [1, h, w, 3] o [h, w, 3]
        if (inputInfo.length === 4) {
          inputSize.h = inputInfo[1] || inputSize.h;
          inputSize.w = inputInfo[2] || inputSize.w;
        } else if (inputInfo.length === 3) {
          inputSize.h = inputInfo[0] || inputSize.h;
          inputSize.w = inputInfo[1] || inputSize.w;
        }
      }
    } catch (e) {
      console.warn("No se pudo leer size del modelo, usando 224x224 por defecto");
    }

    statusEl.innerText = "Modelo cargado";
  } catch (err) {
    console.error("Error cargando modelo:", err);
    statusEl.innerText = "Error cargando modelo. Revisa consola.";
    alert("No se pudo cargar el modelo. Asegúrate de que 'model/model.json' exista y sea accesible.");
  }
}

async function startCamera(){
  try {
    const stream = await navigator.mediaDevices.getUserMedia(VIDEO_FACING);
    video.srcObject = stream;
    statusEl.innerText = "Cámara lista";
  } catch (e) {
    console.error('Error getUserMedia:', e);
    statusEl.innerText = "Error cámara: " + (e && e.message ? e.message : e);
    alert('Error al acceder a la cámara: ' + e + '\nEn GitHub Pages la página está en HTTPS y la cámara debe permitir acceso.');
  }
}

function normalizeText(s) {
  if (!s) return "";
  return s.normalize('NFD').replace(/[\u0300-\u036f]/g, "").toLowerCase();
}

function mapMaterialName(raw) {
  if (!raw) return "--";
  const key = normalizeText(raw).replace(/\s+/g,'_');
  if (materialMap[key]) return materialMap[key];
  const first = normalizeText(raw).split(/\s+/)[0];
  return materialMap[first] || raw;
}

function formatConfidence(v) {
  if (v === null || v === undefined) return "--%";
  let num = Number(v);
  if (isNaN(num)) return String(v);
  if (num <= 1.0) {
    return (Math.round(num * 10000) / 100) + "%";
  }
  return (Math.round(num * 10) / 10) + "%";
}

async function predictFromCanvas(canvas) {
  if (!model) {
    console.warn("Modelo no cargado");
    return null;
  }

  let t = tf.browser.fromPixels(canvas).toFloat();
  // resize to model input
  t = tf.image.resizeBilinear(t, [inputSize.h, inputSize.w]);
  // normalize: Teachable Machine typically uses 0..1
  t = t.div(255.0);
  t = t.expandDims(0); // [1,h,w,3]

  // Predict
  let out = model.predict(t);
  // model.predict puede devolver tensor o array de tensores
  if (Array.isArray(out)) out = out[0];
  const vals = await out.data();
  tf.dispose(t);
  if (out && typeof out.dispose === "function") out.dispose();

  // process results: si la salida no está en 0..1 aplicamos softmax
  let arr = Array.from(vals);
  const maxVal = Math.max(...arr);
  const minVal = Math.min(...arr);
  let probs = arr;
  if (maxVal > 1.01 || minVal < 0) {
    // aplicar softmax
    const exps = arr.map(v => Math.exp(v - maxVal));
    const s = exps.reduce((a,b)=>a+b,0);
    probs = exps.map(e => e / s);
  }

  return probs;
}

function applyInferenceToUIResult(probs) {
  if (!probs) return;
  // Asegurar longitud
  let L = classes.length;
  if (probs.length < L) {
    // rellenar con ceros
    while (probs.length < L) probs.push(0);
  }

  const maxIndex = probs.indexOf(Math.max(...probs));
  const maxProb = probs[maxIndex];
  let label = classes[maxIndex] || ("clase_" + maxIndex);

  // si la confianza es baja -> no reciclable
  let isUnknown = (maxProb < UNKNOWN_THRESHOLD);

  // Nombre a mostrar en pantalla
  let displayName = isUnknown ? "No reciclable" : mapMaterialName(label);

  // Determinar contenedor: si es desconocido -> "No reciclable" explícito
  let contenedor;
  if (isUnknown) {
    contenedor = "No reciclable";
  } else {
    // mapear contenedores según la etiqueta (solo cuando hay suficiente confianza)
    if (label.startsWith("plast")) contenedor = "Plástico";
    else if (label.startsWith("papel")) contenedor = "Papel";
    else if (label.startsWith("metal")) contenedor = "Metal";
    else if (label.startsWith("organ")) contenedor = "Orgánico";
    else contenedor = "No reciclable";
  }

  // Colores (sencillo). Para unknown usamos gris consistente.
  const color = (function(key, unknown){
    if (unknown) return "#9CA3AF"; // gris (tailwind gray-400)
    if (key.startsWith("plast")) return "#3B82F6";
    if (key.startsWith("papel")) return "#FACC15";
    if (key.startsWith("metal")) return "#EF4444";
    if (key.startsWith("organ")) return "#22C55E";
    return "#9CA3AF";
  })(label, isUnknown);

  // Aplicar a UI
  $("materialIdentificado").innerText = displayName;
  $("confidence").innerText = "Confianza: " + formatConfidence(maxProb);
  $("contenedorAsignado").innerText = contenedor;

  // Color del indicador: background + border para mayor contraste
  const contColorEl = $("contenedorColor");
  contColorEl.style.backgroundColor = color;
  // elegir un borde ligeramente más oscuro/contraste para que se vea sobre fondo oscuro
  contColorEl.style.borderColor = color === "#9CA3AF" ? "#6B7280" : color;

  if (isUnknown || displayName === "No reciclable") {
    $("feedbackMessage").innerText = "Objeto no identificado. Enviado a desechos no reciclables.";
  } else {
    const msg = rewardMessages[displayName] || "Clasificación completada.";
    $("feedbackMessage").innerText = msg;
  }
}

function takePhotoDataURLAndPredict() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  foto.src = canvas.toDataURL("image/jpeg", 0.9);
  return predictFromCanvas(canvas).then(probs => {
    applyInferenceToUIResult(probs);
  }).catch(err=>{
    console.error("Error en predict:", err);
    statusEl.innerText = "Error en predicción";
  });
}

async function init() {
  video = $("video");
  snap = $("snap");
  foto = $("videoStream");
  statusEl = $("status");

  await loadModel();
  await startCamera();

  snap.addEventListener("click", async () => {
    if (video.paused || video.ended) return;
    statusEl.innerText = "Tomando foto...";
    await takePhotoDataURLAndPredict();
    statusEl.innerText = "Listo";
  });
}

// iniciar
window.addEventListener("DOMContentLoaded", init);
