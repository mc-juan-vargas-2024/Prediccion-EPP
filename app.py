import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import io
import base64

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Detección EPP",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --amarillo: #F5C518;
    --oscuro:   #0A0C0F;
    --panel:    #12161C;
    --borde:    #1E2530;
    --texto:    #C8D0DC;
    --peligro:  #FF3B30;
    --exito:    #30D158;
}

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: var(--oscuro);
    color: var(--texto);
}

.ppe-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 28px 0 20px 0;
    border-bottom: 2px solid var(--amarillo);
    margin-bottom: 28px;
}
.ppe-badge {
    background: var(--amarillo);
    color: #000;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 2px;
    letter-spacing: 2px;
}
.ppe-titulo {
    font-size: 2.6rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
    margin: 0;
}
.ppe-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #556070;
    letter-spacing: 1px;
    margin-top: 4px;
}

.info-panel {
    background: var(--panel);
    border: 1px solid var(--borde);
    border-left: 3px solid var(--amarillo);
    padding: 16px 20px;
    border-radius: 4px;
    margin-bottom: 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #8899AA;
}
.estado-ok   { border-left-color: var(--exito);   }
.estado-warn { border-left-color: var(--peligro); }

.stButton > button {
    background: var(--amarillo) !important;
    color: #000 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 10px 28px !important;
    letter-spacing: 1px !important;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85 !important; }

[data-testid="stFileUploader"] {
    background: var(--panel) !important;
    border: 2px dashed var(--borde) !important;
    border-radius: 6px !important;
    padding: 20px !important;
}

[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--borde) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--amarillo) !important; }

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--borde);
    border-radius: 4px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #556070 !important; font-family: 'IBM Plex Mono', monospace; font-size:11px; }
[data-testid="stMetricValue"] { color: var(--amarillo) !important; font-size: 2rem !important; }

[data-testid="stProgress"] > div > div { background: var(--amarillo) !important; }

.video-container {
    width: 100%;
    border-radius: 6px;
    overflow: hidden;
    background: #000;
    border: 1px solid var(--borde);
    margin-bottom: 12px;
}
.video-container video {
    width: 100%;
    max-height: 480px;
    display: block;
}

.frame-preview {
    width: 100%;
    border-radius: 4px;
    border: 1px solid var(--amarillo);
    display: block;
    margin-bottom: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--panel) !important;
    border-bottom: 2px solid var(--borde) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #556070 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 10px 24px !important;
    border: none !important;
    letter-spacing: 1px;
}
.stTabs [aria-selected="true"] {
    color: var(--amarillo) !important;
    border-bottom: 2px solid var(--amarillo) !important;
    background: transparent !important;
}

/* Camara snapshot */
.cam-frame {
    width: 100%;
    border-radius: 6px;
    border: 2px solid var(--amarillo);
    display: block;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Encabezado principal ──────────────────────────────────────────────────────
st.markdown("""
<div class="ppe-header">
    <div>
        <div class="ppe-badge">SISTEMA DE SEGURIDAD</div>
        <div class="ppe-titulo">⚠ Detección de EPP</div>
        <div class="ppe-sub">Equipo de Protección Personal · Visión por Computador · YOLOv8</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Carga automática del modelo ───────────────────────────────────────────────
# DESPUÉS — busca automáticamente .pt o .onnx
def buscar_modelo():
    base = Path(__file__).parent
    for nombre in ["best.onnx", "best.pt", "model.onnx", "model.pt"]:
        ruta = base / nombre
        if ruta.exists():
            return ruta
    return None

MODEL_PATH = buscar_modelo()

@st.cache_resource
def cargar_modelo():
    try:
        from ultralytics import YOLO
        if MODEL_PATH is None:
            return None, "No se encontró best.onnx ni best.pt en la carpeta del proyecto"
        mdl = YOLO(str(MODEL_PATH))
        return mdl, None
    except ImportError:
        return None, "ultralytics no instalado. Ejecuta: pip install ultralytics"
    except Exception as e:
        return None, str(e)
modelo, error_modelo = cargar_modelo()

# ── Barra lateral ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ Configuración")

    if modelo:
        st.markdown(
            '<div class="info-panel estado-ok">✅ Modelo cargado</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-panel estado-warn">❌ Modelo no encontrado<br>'
            '<span style="font-size:11px;color:#556070">Coloca best.onnx en la misma carpeta que app.py</span></div>',
            unsafe_allow_html=True,
        )
        if error_modelo:
            st.error(error_modelo)

    st.markdown("---")
    st.markdown("### Parámetros de detección")
    umbral_conf  = st.slider("Umbral de confianza", 0.10, 0.95, 0.40, 0.05)
    umbral_iou   = st.slider("Umbral IoU (NMS)",    0.10, 0.95, 0.45, 0.05)
    grosor_linea = st.slider("Grosor de caja (px)", 1, 6, 2)

    st.markdown("---")
    st.markdown("### Acerca de")
    st.markdown("""
<div class="info-panel">
Detecta EPP en imágenes, videos<br>
y cámara en tiempo real.<br><br>
🪖 Casco &nbsp;·&nbsp; 🦺 Chaleco<br>
🧤 Guantes &nbsp;·&nbsp; 👓 Gafas<br>
👢 Botas &nbsp;·&nbsp; 😷 Mascarilla
</div>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def frame_a_base64(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil   = Image.fromarray(frame_rgb)
    buf       = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()

def video_a_base64(ruta_video):
    with open(ruta_video, "rb") as f:
        return base64.b64encode(f.read()).decode()

def html_video_tag(b64_data, mime="video/mp4"):
    return f"""
<div class="video-container">
  <video controls preload="metadata">
    <source src="data:{mime};base64,{b64_data}" type="{mime}">
  </video>
</div>
"""

def dibujar_resultados(frame_bgr, resultado, grosor=2):
    anotado = frame_bgr.copy()
    cajas   = resultado.boxes
    if cajas is None:
        return anotado, 0
    nombres = resultado.names
    cuenta  = 0
    paleta  = [
        (245, 197, 24), (48, 209, 88), (255, 59, 48),
        (10, 132, 255), (255, 149, 0), (175, 82, 222),
    ]
    for caja in cajas:
        conf = float(caja.conf[0])
        cls  = int(caja.cls[0])
        x1, y1, x2, y2 = map(int, caja.xyxy[0])
        etiqueta = f"{nombres[cls]} {conf:.2f}"
        color    = paleta[cls % len(paleta)]
        cv2.rectangle(anotado, (x1, y1), (x2, y2), color, grosor)
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(anotado, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(anotado, etiqueta, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cuenta += 1
    return anotado, cuenta

# ── Pipeline IMAGEN ───────────────────────────────────────────────────────────
def procesar_imagen(bytes_archivo, mdl, conf, iou, grosor):
    img_pil    = Image.open(io.BytesIO(bytes_archivo)).convert("RGB")
    frame      = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    resultados = mdl(frame, conf=conf, iou=iou, verbose=False)
    anotado, cnt = dibujar_resultados(frame, resultados[0], grosor)
    return cv2.cvtColor(anotado, cv2.COLOR_BGR2RGB), cnt, resultados[0]

# ── Pipeline VIDEO ────────────────────────────────────────────────────────────
def procesar_video(bytes_archivo, mdl, conf, iou, grosor,
                   barra, estado_txt, preview_slot):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(bytes_archivo)
        ruta_entrada = tmp_in.name
    ruta_salida = ruta_entrada.replace(".mp4", "_salida.mp4")

    cap   = cv2.VideoCapture(ruta_entrada)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    escritor = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto))

    PREVIEW_CADA = max(1, int(fps // 4))
    idx = 0
    total_dets = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resultados   = mdl(frame, conf=conf, iou=iou, verbose=False)
        anotado, cnt = dibujar_resultados(frame, resultados[0], grosor)
        escritor.write(anotado)
        total_dets += cnt
        idx        += 1

        barra.progress(min(idx / max(total, 1), 1.0))
        estado_txt.markdown(
            f'<div class="info-panel">🎞 Fotograma {idx}/{total} · '
            f'Detecciones: <b>{cnt}</b> · Total: <b>{total_dets}</b></div>',
            unsafe_allow_html=True,
        )
        if idx % PREVIEW_CADA == 0 or idx == 1:
            b64 = frame_a_base64(anotado)
            preview_slot.markdown(
                f'<img src="data:image/jpeg;base64,{b64}" class="frame-preview">',
                unsafe_allow_html=True,
            )

    cap.release()
    escritor.release()
    os.unlink(ruta_entrada)
    return ruta_salida, total_dets, idx

# ── TABS principales ──────────────────────────────────────────────────────────
tab_imagen, tab_video, tab_camara = st.tabs([
    "🖼️  Imagen",
    "🎬  Video",
    "📷  Cámara en vivo",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGEN
# ════════════════════════════════════════════════════════════════════════════
with tab_imagen:
    st.markdown("### 📂 Subir imagen")
    archivo_img = st.file_uploader(
        "Arrastra o selecciona una imagen",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
        key="uploader_img",
    )

    if archivo_img:
        bytes_img = archivo_img.read()
        col_orig, col_pred = st.columns(2, gap="large")
        with col_orig:
            st.markdown("#### Original")
            st.image(Image.open(io.BytesIO(bytes_img)), use_container_width=True)
        with col_pred:
            st.markdown("#### Predicción")
            if not modelo:
                st.markdown('<div class="info-panel estado-warn">⚠ Modelo no encontrado</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Ejecutando inferencia…"):
                    anotado_rgb, cnt, _ = procesar_imagen(bytes_img, modelo, umbral_conf, umbral_iou, grosor_linea)
                st.image(anotado_rgb, use_container_width=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Detecciones", cnt)
                m2.metric("Confianza",   f"{umbral_conf:.0%}")
                m3.metric("Resolución",  f"{anotado_rgb.shape[1]}×{anotado_rgb.shape[0]}")
                buf = io.BytesIO()
                Image.fromarray(anotado_rgb).save(buf, format="PNG")
                buf.seek(0)
                st.download_button(
                    "⬇ Descargar imagen con predicción",
                    data=buf,
                    file_name=f"epp_{Path(archivo_img.name).stem}_pred.png",
                    mime="image/png",
                    use_container_width=True,
                )
    else:
        st.markdown("""
<div style="background:#12161C;border:1px solid #1E2530;border-radius:8px;
    padding:50px 40px;text-align:center;margin-top:10px;">
    <div style="font-size:56px;margin-bottom:12px;">🖼️</div>
    <div style="font-size:1.3rem;font-weight:700;color:#fff;">Sube una imagen para comenzar</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#556070;margin-top:8px;">
        Formatos: JPG · PNG · BMP · WEBP
    </div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO
# ════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown("### 📂 Subir video")
    archivo_vid = st.file_uploader(
        "Arrastra o selecciona un video",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
        key="uploader_vid",
    )

    if archivo_vid:
        bytes_vid = archivo_vid.read()
        ext       = Path(archivo_vid.name).suffix.lower()
        clave_pred = f"pred_{archivo_vid.name}_{len(bytes_vid)}"
        clave_meta = f"meta_{archivo_vid.name}_{len(bytes_vid)}"

        col_orig, col_pred = st.columns(2, gap="large")

        with col_orig:
            st.markdown("#### Video original")
            tam_mb = len(bytes_vid) / 1024 / 1024
            st.markdown(
                f'<div class="info-panel estado-ok" style="text-align:center;padding:40px 20px;">'
                f'<div style="font-size:40px;margin-bottom:10px;">✅</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:#fff;">{archivo_vid.name}</div>'
                f'<div style="font-size:11px;color:#556070;margin-top:4px;">{tam_mb:.1f} MB</div></div>',
                unsafe_allow_html=True,
            )

        with col_pred:
            st.markdown("#### Predicción")
            if not modelo:
                st.markdown('<div class="info-panel estado-warn">⚠ Modelo no encontrado</div>', unsafe_allow_html=True)
            elif clave_pred in st.session_state:
                meta = st.session_state[clave_meta]
                st.markdown(
                    f'<div class="info-panel estado-ok" style="text-align:center;padding:24px;">'
                    f'<div style="font-size:32px;">✅</div>'
                    f'<div style="font-weight:700;color:#fff;margin-top:8px;">Completado</div>'
                    f'<div style="font-size:11px;color:#556070;">{meta["total_dets"]} detecciones · {meta["total_frames"]} frames</div></div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "⬇ Descargar video con predicción",
                    data=base64.b64decode(st.session_state[clave_pred]),
                    file_name=f"epp_{Path(archivo_vid.name).stem}_pred.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
            else:
                preview_slot = st.empty()
                barra_slot   = st.empty()
                estado_slot  = st.empty()
                if st.button("▶ Ejecutar detección en video", use_container_width=True):
                    preview_slot.markdown('<div class="info-panel" style="text-align:center;padding:40px;">⏳ Iniciando…</div>', unsafe_allow_html=True)
                    barra_prog = barra_slot.progress(0.0)
                    ruta_salida, total_dets, total_frames = procesar_video(
                        bytes_vid, modelo, umbral_conf, umbral_iou, grosor_linea,
                        barra_prog, estado_slot, preview_slot,
                    )
                    barra_slot.empty(); estado_slot.empty()
                    st.session_state[clave_pred] = video_a_base64(ruta_salida)
                    st.session_state[clave_meta] = {"total_dets": total_dets, "total_frames": total_frames}
                    os.unlink(ruta_salida)
                    st.rerun()
    else:
        st.markdown("""
<div style="background:#12161C;border:1px solid #1E2530;border-radius:8px;
    padding:50px 40px;text-align:center;margin-top:10px;">
    <div style="font-size:56px;margin-bottom:12px;">🎬</div>
    <div style="font-size:1.3rem;font-weight:700;color:#fff;">Sube un video para comenzar</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#556070;margin-top:8px;">
        Formatos: MP4 · AVI · MOV · MKV
    </div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — CÁMARA EN VIVO (detección real frame a frame con YOLO)
# ════════════════════════════════════════════════════════════════════════════
with tab_camara:
    st.markdown("### 📷 Detección en tiempo real")
    st.markdown("""
<div class="info-panel" style="font-size:12px;line-height:1.7;">
    El modelo analiza cada frame de la cámara en tiempo real.<br>
    <b>Tip:</b> Permite el acceso a la cámara cuando el navegador lo solicite.
</div>
""", unsafe_allow_html=True)

    if not modelo:
        st.markdown(
            '<div class="info-panel estado-warn">⚠ Modelo no encontrado. '
            'Coloca <b>best.onnx</b> en la carpeta del proyecto.</div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
            import av

            # ── Configuración STUN/TURN para conexión WebRTC ──────────────
            RTC_CONFIG = RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            })

            # ── Processor: cada frame pasa por aquí ───────────────────────
            class DetectorEPP(VideoProcessorBase):

                def __init__(self):
                    self.conf       = umbral_conf
                    self.iou        = umbral_iou
                    self.grosor     = grosor_linea
                    self.detecciones = 0
                    self.nombres_det = []

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    # Convertir frame a numpy BGR
                    img_bgr = frame.to_ndarray(format="bgr24")

                    # Inferencia YOLO
                    resultados = modelo(
                        img_bgr,
                        conf=self.conf,
                        iou=self.iou,
                        verbose=False,
                    )
                    resultado = resultados[0]

                    # Dibujar cajas
                    anotado, cnt = dibujar_resultados(img_bgr, resultado, self.grosor)

                    # Guardar stats para mostrar en sidebar
                    self.detecciones = cnt
                    if resultado.boxes and len(resultado.boxes) > 0:
                        nombres = resultado.names
                        self.nombres_det = [
                            f"{nombres[int(c.cls[0])]} {float(c.conf[0]):.0%}"
                            for c in resultado.boxes
                        ]
                    else:
                        self.nombres_det = []

                    # Superponer contador en el frame
                    h, w = anotado.shape[:2]
                    overlay_txt = f"EPP: {cnt} detecciones"
                    cv2.rectangle(anotado, (0, 0), (w, 36), (10, 12, 15), -1)
                    cv2.putText(
                        anotado, overlay_txt,
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (245, 197, 24), 2, cv2.LINE_AA,
                    )

                    # Devolver frame anotado
                    return av.VideoFrame.from_ndarray(anotado, format="bgr24")

            # ── Panel lateral: stats en vivo ──────────────────────────────
            col_cam, col_info = st.columns([3, 1], gap="medium")

            with col_cam:
                ctx = webrtc_streamer(
                    key="epp-live",
                    video_processor_factory=DetectorEPP,
                    rtc_configuration=RTC_CONFIG,
                    media_stream_constraints={
                        "video": {
                            "width":  {"ideal": 640},
                            "height": {"ideal": 480},
                            "frameRate": {"ideal": 24},
                        },
                        "audio": False,
                    },
                    async_processing=True,
                    translations={
                        "start": "▶ Iniciar cámara",
                        "stop":  "⏹ Detener",
                        "select_device": "Seleccionar cámara",
                    },
                )

            with col_info:
                st.markdown("#### Estado")

                if ctx.state.playing:
                    st.markdown(
                        '<div class="info-panel estado-ok" style="text-align:center;padding:14px;">'
                        '<b style="color:#30D158;font-size:13px;">● EN VIVO</b></div>',
                        unsafe_allow_html=True,
                    )

                    # Refresca stats del processor cada 0.5s
                    import time
                    stats_placeholder = st.empty()
                    objetos_placeholder = st.empty()

                    while ctx.state.playing:
                        if ctx.video_processor:
                            proc = ctx.video_processor
                            cnt  = proc.detecciones
                            noms = proc.nombres_det

                            # Métrica principal
                            stats_placeholder.markdown(
                                f'<div class="info-panel" style="text-align:center;padding:16px;">'
                                f'<div style="font-family:\'IBM Plex Mono\',monospace;'
                                f'font-size:10px;color:#556070;letter-spacing:1px;">OBJETOS</div>'
                                f'<div style="font-size:2.4rem;color:#F5C518;font-weight:700;">{cnt}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            # Lista de objetos
                            if noms:
                                items_html = "".join([
                                    f'<div style="padding:7px 12px;margin-bottom:6px;'
                                    f'background:#0A0C0F;border-left:3px solid #F5C518;'
                                    f'font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                                    f'color:#C8D0DC;border-radius:2px;">'
                                    f'{n}</div>'
                                    for n in noms
                                ])
                                objetos_placeholder.markdown(
                                    f'<div style="margin-top:8px;">'
                                    f'<div style="font-family:\'IBM Plex Mono\',monospace;'
                                    f'font-size:10px;color:#556070;letter-spacing:1px;'
                                    f'margin-bottom:6px;">DETECTADOS</div>'
                                    f'{items_html}</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                objetos_placeholder.markdown(
                                    '<div class="info-panel" style="font-size:11px;'
                                    'color:#556070;text-align:center;">Sin detecciones</div>',
                                    unsafe_allow_html=True,
                                )

                        time.sleep(0.5)

                else:
                    st.markdown(
                        '<div class="info-panel" style="text-align:center;padding:14px;">'
                        '<span style="color:#556070;font-size:12px;">— INACTIVO</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("""
<div class="info-panel" style="font-size:11px;line-height:1.8;margin-top:8px;">
🪖 Casco<br>
🦺 Chaleco<br>
🧤 Guantes<br>
👓 Gafas<br>
👢 Botas<br>
😷 Mascarilla
</div>
""", unsafe_allow_html=True)

        except ImportError:
            st.error("❌ Falta instalar streamlit-webrtc")
            st.code("pip install streamlit-webrtc", language="bash")
            st.markdown("""
<div class="info-panel estado-warn" style="font-size:12px;line-height:1.8;">
Después de instalar, reinicia Streamlit:<br>
<code>streamlit run app.py</code>
</div>
""", unsafe_allow_html=True)
    st.markdown("### 📷 Cámara en vivo")
    st.markdown("""
<div class="info-panel" style="font-size:12px;line-height:1.7;">
    Captura una foto con tu cámara y el modelo analizará la imagen automáticamente.<br>
    <b>Tip:</b> Asegúrate de dar permiso de cámara al navegador cuando lo solicite.
</div>
""", unsafe_allow_html=True)

    if not modelo:
        st.markdown('<div class="info-panel estado-warn">⚠ Modelo no encontrado. Coloca <b>best.onnx</b> en la carpeta del proyecto.</div>', unsafe_allow_html=True)
    else:
        # st.camera_input es el widget nativo de Streamlit para acceder a la cámara
        foto = st.camera_input(
            label="Captura una foto",
            label_visibility="collapsed",
            key="camara_input",
        )

        if foto:
            bytes_foto = foto.read()
            col_orig, col_pred = st.columns(2, gap="large")

            with col_orig:
                st.markdown("#### Foto capturada")
                st.image(Image.open(io.BytesIO(bytes_foto)), use_container_width=True)

            with col_pred:
                st.markdown("#### Detección EPP")
                with st.spinner("Analizando imagen…"):
                    anotado_rgb, cnt, resultado = procesar_imagen(
                        bytes_foto, modelo, umbral_conf, umbral_iou, grosor_linea
                    )
                st.image(anotado_rgb, use_container_width=True)

                # Métricas
                m1, m2, m3 = st.columns(3)
                m1.metric("Objetos detectados", cnt)
                m2.metric("Confianza mínima",   f"{umbral_conf:.0%}")
                m3.metric("Resolución",          f"{anotado_rgb.shape[1]}×{anotado_rgb.shape[0]}")

                # Detalle por clase
                if resultado.boxes and len(resultado.boxes) > 0:
                    st.markdown("#### Detalle por detección")
                    nombres = resultado.names
                    for i, caja in enumerate(resultado.boxes):
                        cls  = int(caja.cls[0])
                        conf = float(caja.conf[0])
                        nombre_clase = nombres[cls]
                        st.markdown(
                            f'<div class="info-panel" style="padding:10px 16px;margin-bottom:8px;">'
                            f'<b style="color:#fff;">#{i+1} {nombre_clase}</b> '
                            f'<span style="color:#F5C518;float:right;">confianza: {conf:.2%}</span></div>',
                            unsafe_allow_html=True,
                        )

                # Descarga
                buf = io.BytesIO()
                Image.fromarray(anotado_rgb).save(buf, format="PNG")
                buf.seek(0)
                st.download_button(
                    "⬇ Descargar foto con detecciones",
                    data=buf,
                    file_name="epp_camara_pred.png",
                    mime="image/png",
                    use_container_width=True,
                )
        else:
            st.markdown("""
<div style="background:#12161C;border:1px solid #1E2530;border-radius:8px;
    padding:40px;text-align:center;margin-top:10px;">
    <div style="font-size:56px;margin-bottom:12px;">📷</div>
    <div style="font-size:1.3rem;font-weight:700;color:#fff;">Presiona el botón para activar la cámara</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#556070;margin-top:8px;">
        La detección se ejecutará automáticamente al capturar la foto
    </div>
</div>""", unsafe_allow_html=True)
