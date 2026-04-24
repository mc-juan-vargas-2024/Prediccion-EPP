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
st.markdown("""
<div class="info-panel" style="font-size:12px;line-height:1.7;">
    <b>Umbral de confianza:</b> nivel mínimo de certeza para mostrar una detección. 
    Valores altos reducen falsos positivos; bajos detectan más con menor seguridad.<br><br>
    <b>Umbral IoU (NMS):</b> controla el solapamiento permitido entre cajas. 
    Valores bajos eliminan más duplicados; altos permiten cajas más próximas.<br><br>
    <b>Grosor de caja:</b> tamaño en px del borde dibujado sobre cada detección.
</div>
""", unsafe_allow_html=True)

# ── Carga automática del modelo ───────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "best.onnx"

@st.cache_resource
def cargar_modelo():
    try:
        from ultralytics import YOLO
        if not MODEL_PATH.exists():
            return None, f"No se encontró el archivo: {MODEL_PATH}"
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
            '<span style="font-size:11px;color:#556070">Coloca best.pt en la misma carpeta que app.py</span></div>',
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
Sube una imagen o video para detectar EPP.<br>
El modelo detectará elementos como:<br><br>
🪖 Casco &nbsp;·&nbsp; 🦺 Chaleco<br>
🧤 Guantes &nbsp;·&nbsp; 👓 Gafas<br>
👢 Botas &nbsp;·&nbsp; 😷 Mascarilla
</div>
""", unsafe_allow_html=True)

# ── Subida de archivo ─────────────────────────────────────────────────────────
st.markdown("### 📂 Subir imagen o video")
archivo_subido = st.file_uploader(
    "Arrastra y suelta o haz clic para seleccionar",
    type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed",
)

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
    Tu navegador no soporta reproducción de video.
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

# ── Pipeline VIDEO con preview en tiempo real ─────────────────────────────────
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

    idx        = 0
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
            f'Detecciones en frame: <b>{cnt}</b> · '
            f'Total acumulado: <b>{total_dets}</b></div>',
            unsafe_allow_html=True,
        )

        if idx % PREVIEW_CADA == 0 or idx == 1:
            b64 = frame_a_base64(anotado)
            preview_slot.markdown(
                f'<img src="data:image/jpeg;base64,{b64}" '
                f'class="frame-preview" alt="frame {idx}">',
                unsafe_allow_html=True,
            )

    cap.release()
    escritor.release()
    os.unlink(ruta_entrada)
    return ruta_salida, total_dets, idx

# ── Área principal ────────────────────────────────────────────────────────────
if archivo_subido:
    bytes_archivo = archivo_subido.read()
    ext      = Path(archivo_subido.name).suffix.lower()
    es_video = ext in {".mp4", ".avi", ".mov", ".mkv"}

    col_orig, col_pred = st.columns(2, gap="large")

    # ════════════════════════════════════════════════════════════
    # IMAGEN
    # ════════════════════════════════════════════════════════════
    if not es_video:
        with col_orig:
            st.markdown("#### Original")
            st.image(Image.open(io.BytesIO(bytes_archivo)), use_container_width=True)

        with col_pred:
            st.markdown("#### Predicción")
            if not modelo:
                st.markdown(
                    '<div class="info-panel estado-warn">⚠ Coloca <b>best.pt</b> en la misma carpeta que app.py</div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Ejecutando inferencia …"):
                    anotado_rgb, cnt, _ = procesar_imagen(
                        bytes_archivo, modelo, umbral_conf, umbral_iou, grosor_linea
                    )
                st.image(anotado_rgb, use_container_width=True)

                st.markdown("#### Resultados")
                m1, m2, m3 = st.columns(3)
                m1.metric("Detecciones",  cnt)
                m2.metric("Confianza",    f"{umbral_conf:.0%}")
                m3.metric("Resolución",   f"{anotado_rgb.shape[1]}×{anotado_rgb.shape[0]}")

                buf = io.BytesIO()
                Image.fromarray(anotado_rgb).save(buf, format="PNG")
                buf.seek(0)
                st.download_button(
                    label="⬇ Descargar imagen con predicción",
                    data=buf,
                    file_name=f"epp_{Path(archivo_subido.name).stem}_pred.png",
                    mime="image/png",
                    use_container_width=True,
                )

    # ════════════════════════════════════════════════════════════
    # VIDEO
    # ════════════════════════════════════════════════════════════
    else:
        mime_map  = {".mp4": "video/mp4", ".avi": "video/avi",
                     ".mov": "video/quicktime", ".mkv": "video/x-matroska"}
        mime_orig = mime_map.get(ext, "video/mp4")

        # Claves únicas por archivo en session_state
        clave_orig = f"orig_{archivo_subido.name}_{len(bytes_archivo)}"
        clave_pred = f"pred_{archivo_subido.name}_{len(bytes_archivo)}"
        clave_meta = f"meta_{archivo_subido.name}_{len(bytes_archivo)}"

        # Codificar el original solo una vez y persistirlo
        if clave_orig not in st.session_state:
            st.session_state[clave_orig] = base64.b64encode(bytes_archivo).decode()

        # ── Columna izquierda: mensaje video cargado ────────────────────
        with col_orig:
            st.markdown("#### Video original")
            nombre = archivo_subido.name
            tam_mb = len(bytes_archivo) / 1024 / 1024
            st.markdown(
                f"""<div class="info-panel estado-ok" style="text-align:center;padding:40px 20px;">
                <div style="font-size:48px;margin-bottom:12px;">✅</div>
                <div style="font-size:1.2rem;font-weight:700;color:#fff;margin-bottom:8px;">Video cargado correctamente</div>
                <div style="font-size:11px;color:#556070;">{nombre} &nbsp;·&nbsp; {tam_mb:.1f} MB</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # ── Columna derecha: predicción ───────────────────────────────────
        with col_pred:
            st.markdown("#### Predicción")

            if not modelo:
                st.markdown(
                    '<div class="info-panel estado-warn">⚠ Coloca <b>best.pt</b> '
                    'en la misma carpeta que app.py</div>',
                    unsafe_allow_html=True,
                )

            elif clave_pred in st.session_state:
                # Ya procesado: solo botón de descarga
                meta = st.session_state[clave_meta]
                st.markdown(
                    f'''<div class="info-panel estado-ok" style="text-align:center;padding:24px 20px;">
                    <div style="font-size:36px;margin-bottom:8px;">✅</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#fff;margin-bottom:4px;">Predicción completada</div>
                    <div style="font-size:11px;color:#556070;">{meta["total_dets"]} detecciones · {meta["total_frames"]} fotogramas</div>
                    </div>''',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    label="⬇ Descargar video con predicción",
                    data=base64.b64decode(st.session_state[clave_pred]),
                    file_name=f"epp_{Path(archivo_subido.name).stem}_pred.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

            else:
                # Aún no procesado: botón + slots de progreso
                preview_slot = st.empty()
                barra_slot   = st.empty()
                estado_slot  = st.empty()

                if st.button("▶ Ejecutar detección en video", use_container_width=True):
                    preview_slot.markdown(
                        '<div class="info-panel" style="text-align:center;padding:40px;">'
                        '⏳ Iniciando procesamiento…</div>',
                        unsafe_allow_html=True,
                    )
                    barra_prog = barra_slot.progress(0.0)

                    ruta_salida, total_dets, total_frames = procesar_video(
                        bytes_archivo, modelo,
                        umbral_conf, umbral_iou, grosor_linea,
                        barra_prog, estado_slot, preview_slot,
                    )

                    barra_slot.empty()
                    estado_slot.empty()

                    # Persistir resultado en session_state
                    st.session_state[clave_pred] = video_a_base64(ruta_salida)
                    st.session_state[clave_meta] = {
                        "total_dets": total_dets,
                        "total_frames": total_frames,
                    }
                    os.unlink(ruta_salida)

                    # Rerender limpio: ambos videos lado a lado con controles
                    st.rerun()

else:
    # ── Banner de bienvenida ──────────────────────────────────────────────────
    st.markdown("""
<div style="
    background:#12161C;border:1px solid #1E2530;border-radius:8px;
    padding:60px 40px;text-align:center;margin-top:20px;">
    <div style="font-size:72px;margin-bottom:16px;">🦺</div>
    <div style="font-size:1.6rem;font-weight:700;color:#fff;margin-bottom:8px;">
        Detección de Equipo de Protección Personal
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#556070;margin-bottom:32px;">
        Asegúrate de tener <b style="color:#F5C518">best.pt</b> en la misma carpeta · luego sube una imagen o video
    </div>
    <div style="display:flex;justify-content:center;gap:24px;flex-wrap:wrap;">
        <span style="background:#1E2530;padding:10px 20px;border-radius:4px;font-size:14px;">🪖 Cascos</span>
        <span style="background:#1E2530;padding:10px 20px;border-radius:4px;font-size:14px;">🦺 Chalecos</span>
        <span style="background:#1E2530;padding:10px 20px;border-radius:4px;font-size:14px;">🧤 Guantes</span>
        <span style="background:#1E2530;padding:10px 20px;border-radius:4px;font-size:14px;">👓 Gafas</span>
        <span style="background:#1E2530;padding:10px 20px;border-radius:4px;font-size:14px;">😷 Mascarillas</span>
        <span style="background:#1E2530;padding:10px 20px;border-radius:4px;font-size:14px;">👢 Botas</span>
    </div>
</div>
""", unsafe_allow_html=True)