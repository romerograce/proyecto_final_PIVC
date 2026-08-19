"""
app.py
------
Interfaz web para la demo en vivo del proyecto: sube una foto o sacala
con la cámara, y muestra el resultado de las 3 redes de clasificación
de edad (CNN propia, EfficientNet-B0, SSR-Net) lado a lado.

Uso:
    streamlit run app.py

Se abre automáticamente en el navegador (normalmente localhost:8501).
Ideal para mostrar en vivo durante la defensa: subís o sacás una foto
y ves las 3 predicciones comparadas en tiempo real.

Requiere (además de lo que ya usan los otros scripts):
    pip install streamlit
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.detectar_edades import cargar_modelos as cargar_cnn, detectar_edades
from src.detectar_edades_effi import cargar_modelos as cargar_effi, detectar_edades_effi
from src.detectar_edades_ssr import cargar_modelos as cargar_ssr, detectar_edades_ssr


st.set_page_config(
    page_title="Detección de personas y edad — FEXPOCRUZ",
    layout="wide"
)


# Cachea los modelos para no recargarlos en cada foto -- se cargan
# una sola vez cuando arranca la app.
@st.cache_resource(show_spinner="Cargando modelos (una sola vez)...")
def cargar_todos_los_modelos():
    modelos_cnn = cargar_cnn()
    modelos_effi = cargar_effi()
    modelos_ssr = cargar_ssr()
    return modelos_cnn, modelos_effi, modelos_ssr


def bgr_a_rgb_pil(imagen_bgr):
    return Image.fromarray(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB))


def guardar_temporal(imagen_pil):
    ruta = Path("temp_captura.jpg")
    imagen_pil.save(ruta)
    return ruta


st.title("Conteo y clasificación de edad — Pabellones FEXPOCRUZ")
st.caption(
    "Detección de personas (YOLO + CrowdHuman) y estimación de rango "
    "etario, comparando 3 arquitecturas entrenadas sobre UTKFace."
)

modelos_cnn, modelos_effi, modelos_ssr = cargar_todos_los_modelos()

st.divider()

origen = st.radio(
    "Elegí el origen de la imagen:",
    ["Subir foto", "Usar cámara"],
    horizontal=True
)

imagen_subida = None

if origen == "Subir foto":
    archivo = st.file_uploader(
        "Subí una imagen con una o más personas",
        type=["jpg", "jpeg", "png"]
    )
    if archivo is not None:
        imagen_subida = Image.open(archivo).convert("RGB")

else:
    captura = st.camera_input("Sacá una foto")
    if captura is not None:
        imagen_subida = Image.open(captura).convert("RGB")

if imagen_subida is not None:
    ruta_temporal = guardar_temporal(imagen_subida)

    with st.spinner("Procesando con las 3 redes..."):
        t0 = time.time()
        img_cnn, det_cnn = detectar_edades(ruta_temporal, modelos=modelos_cnn)
        t_cnn = time.time() - t0

        t0 = time.time()
        img_effi, det_effi = detectar_edades_effi(ruta_temporal, modelos=modelos_effi)
        t_effi = time.time() - t0

        t0 = time.time()
        img_ssr, det_ssr = detectar_edades_ssr(ruta_temporal, modelos=modelos_ssr)
        t_ssr = time.time() - t0

    st.divider()
    st.subheader(f"Personas detectadas: {len(det_cnn)}")
    st.caption(
        "CNN propia y SSR-Net dieron los mejores resultados en pruebas reales: "
        "la CNN con mayor accuracy numérica, SSR-Net más robusta al descartar "
        "rostros poco claros en vez de forzar una predicción."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🏆 CNN propia")
        st.image(bgr_a_rgb_pil(img_cnn), use_container_width=True)
        st.caption(f"{t_cnn:.2f}s | 421,828 parámetros | 68.3% acc. val")
        st.caption("Mejor accuracy medida en validación")
        for d in det_cnn:
            st.write(f"- {d['edad']} ({d['confianza']:.1%})")

    with col2:
        st.markdown("### EfficientNet-B0")
        st.image(bgr_a_rgb_pil(img_effi), use_container_width=True)
        st.caption(f"{t_effi:.2f}s | 4,054,737 parámetros | 67.8% acc. val")
        for d in det_effi:
            st.write(f"- {d['edad']} ({d['confianza']:.1%})")

    with col3:
        st.markdown("### ✅ SSR-Net")
        st.image(bgr_a_rgb_pil(img_ssr), use_container_width=True)
        st.caption(f"{t_ssr:.2f}s | 88,297 parámetros | 9.55 años MAE val")
        st.caption("Más robusta: descarta rostros poco claros en vez de adivinar")
        for d in det_ssr:
            edad_txt = f"{d['edad_estimada']:.0f} años" if d["edad_estimada"] else "N/D"
            st.write(f"- {edad_txt} → {d['edad']}")

    ruta_temporal.unlink(missing_ok=True)

else:
    st.info("Subí una foto o sacá una con la cámara para ver la demo en vivo.")
