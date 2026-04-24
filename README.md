# 🦺 EPP Detector — Streamlit App

**Autor:** Juan José Vargas  
**Modelo:** YOLOv8s entrenado con dataset PPE Factory (Roboflow)  
**Framework:** Streamlit + Ultralytics

🔗 **App en producción:** [Abrir aplicación](https://prediccion-epp-c3a4ni9hru98cekaqrgoz5.streamlit.app/)

---

## 📌 Descripción

Aplicación web interactiva para la **detección de Equipos de Protección Personal (EPP)** en imágenes y videos, utilizando un modelo YOLOv8s entrenado con el dataset `ppe-factory` de Roboflow sobre una GPU Tesla T4.

La app permite:
- Subir imágenes o videos para análisis en tiempo real
- Visualizar detecciones con bounding boxes por clase
- Ajustar el umbral de confianza con un slider interactivo
- Consultar métricas del modelo (mAP, precisión, recall)
- Exportar resultados anotados

---

## 🗂️ Estructura del Proyecto

```
epp-detector/
│
├── app.py                        # Aplicación principal Streamlit
├── EPP.ipynb                     # Notebook de entrenamiento y evaluación
├── requirements.txt              # Dependencias del proyecto
├── README.md                     # Este archivo
│
├── models/
│   └── best.pt                   # Pesos del mejor modelo entrenado (YOLOv8s)
│
├── data/
│   └── ppe-factory-1/
│       ├── train/                # 5002 imágenes de entrenamiento
│       ├── valid/                # 371 imágenes de validación
│       ├── test/                 # 143 imágenes de prueba
│       └── data.yaml             # Configuración de clases y rutas
│
└── runs/
    └── ppe_detector/
        └── weights/
            ├── best.pt           # Mejores pesos (mAP50 más alto)
            └── best.onnx         # Modelo exportado a ONNX
```

---

## ⚙️ Instalación local

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/epp-detector.git
cd epp-detector
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la app

```bash
streamlit run app.py
```

Disponible en `http://localhost:8501`

---

## 📦 Dependencias (`requirements.txt`)

```
streamlit>=1.32.0
ultralytics>=8.0.0
roboflow>=1.1.0
opencv-python-headless>=4.9.0
Pillow>=10.0.0
numpy>=1.24.0
PyYAML>=6.0
torch>=2.0.0
```

> ⚠️ En Streamlit Cloud usar siempre `opencv-python-headless` en lugar de `opencv-python`.

---

## 🧠 Modelo

| Parámetro        | Valor                        |
|------------------|------------------------------|
| Arquitectura     | YOLOv8s                      |
| Dataset          | PPE Factory v1 (Roboflow)    |
| Épocas           | 50                           |
| Tamaño de imagen | 416 × 416 px                 |
| Batch size       | 16                           |
| GPU              | Tesla T4                     |
| Formato export   | PyTorch `.pt` / ONNX         |

### 📈 Métricas finales (validación)

| Clase      | Precisión | Recall | mAP50 |
|------------|-----------|--------|-------|
| boots      | 0.957     | 0.763  | 0.887 |
| earmuffs   | 0.746     | 0.616  | 0.673 |
| glasses    | 0.842     | 0.608  | 0.750 |
| gloves     | 0.411     | 0.510  | 0.510 |
| helmet     | 0.908     | 0.882  | 0.917 |
| person     | 0.871     | 0.782  | 0.887 |
| vest       | 0.907     | 0.809  | 0.892 |
| **all**    | **0.806** | **0.710** | **0.788** |

---

## 🚀 Uso de la App

### Detección en imagen
1. Selecciona la pestaña **"Imagen"** en el menú lateral
2. Sube un archivo `.jpg`, `.jpeg` o `.png`
3. Ajusta el umbral de confianza con el slider
4. Haz clic en **"Detectar"** para ver el resultado anotado

### Detección en video
1. Selecciona la pestaña **"Video"**
2. Sube un archivo `.mp4` o `.avi`
3. El modelo procesará cada frame y mostrará las detecciones

### Métricas del modelo
La pestaña **"Métricas"** muestra el rendimiento del modelo por clase: mAP50, mAP50-95, Precisión y Recall.

---

## 📓 Notebook de Entrenamiento

El archivo `EPP.ipynb` contiene el pipeline completo:

1. Instalación de dependencias (`ultralytics`, `roboflow`)
2. Descarga del dataset desde Roboflow (v1, formato YOLOv8)
3. Exploración y conteo de archivos por split
4. Lectura de clases desde `data.yaml`
5. Carga del modelo base YOLOv8s (transfer learning)
6. Entrenamiento — 50 épocas, imgsz=416, batch=16
7. Carga del mejor modelo (`best.pt`)
8. Evaluación con métricas mAP50 / mAP50-95
9. Inferencia en imágenes individuales y en batch
10. Descarga y predicción sobre video de ejemplo
11. Exportación del modelo a ONNX

> El notebook fue desarrollado en **Google Colab** con GPU T4. Para ejecutarlo localmente ajusta las rutas `/content/...` a rutas locales.

---

## 🛠️ Código base — `app.py`

```python
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="EPP Detector", page_icon="🦺", layout="wide")
st.title("🦺 Detección de EPP con YOLOv8")
st.markdown("**Autor:** Juan José Vargas")

# Carga el modelo entrenado
model = YOLO("models/best.pt")

# Sidebar con configuración
st.sidebar.header("⚙️ Configuración")
confidence = st.sidebar.slider("Umbral de confianza", 0.1, 1.0, 0.25)

# Carga de imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen original", use_column_width=True)

    if st.button("🔍 Detectar EPP"):
        results = model.predict(np.array(image), conf=confidence)
        annotated = results[0].plot()
        st.image(annotated, caption="Resultado de detección", use_column_width=True)

        # Resumen de detecciones
        boxes = results[0].boxes
        st.success(f"✅ Se detectaron **{len(boxes)}** objetos.")
```

---

## 👤 Autor

**Juan José Vargas**  
Proyecto de detección de Equipos de Protección Personal usando visión por computadora y deep learning con YOLOv8.


# Prediccion-EPP
# Prediccion-EPP
