# CrowdLens: Detección de Personas y Estimación de Edad mediante Aprendizaje Profundo

Detección de personas en escenas de alta densidad (YOLO + CrowdHuman) y
estimación de rango etario, comparando tres arquitecturas de clasificación
entrenadas sobre UTKFace.

**Universidad Católica Boliviana "San Pablo" — Sede Santa Cruz**

Maestría en Ciencia de Datos e Inteligencia Artificial Aplicada

MCI-509 · Procesamiento de Imágenes y Visión Computacional

## Resumen

En diferentes ferias comerciales del país no se cuenta hoy en día con
información sobre cómo se distribuyen sus visitantes dentro del recinto:
el único dato disponible es el control de acceso general, que no indica
qué pabellones/stands concentran más flujo ni en qué horarios. Este
proyecto aborda la detección y el conteo de personas en escenas de alta
densidad y oclusión —la condición típica de un pabellón ferial en hora
pico— usando el video de las cámaras de vigilancia ya instaladas como
fuente de datos, y explora una capa adicional de estimación de rango
etario sobre las cabezas detectadas para enriquecer el conteo con una
caracterización demográfica básica del flujo.

## Arquitectura del pipeline

1. **Detección de cabezas** — YOLO11n, *fine-tuning* sobre CrowdHuman
   (caja `hbox`), una sola clase (`head`). Entrenado 30 épocas,
   `imgsz`=640, `batch`=8, sobre la partición train01, con early
   stopping (patience=5).
2. **Refinamiento facial** — MediaPipe/BlazeFace, aplicado únicamente
   dentro de la región ya acotada por YOLO, para reducir ruido de fondo
   antes de la clasificación de edad.
3. **Estimación de edad** — sobre cada rostro refinado, se clasifica en
   uno de 4 rangos etarios: `01_15`, `16_35`, `36_55`, `56_mas`. Se
   entrenaron y compararon **tres arquitecturas** sobre UTKFace:
   - **CNN propia** — diseñada desde cero (4 bloques convolucionales,
     32/64/128/256 canales, pooling global, dropout). 421,828 parámetros.
   - **EfficientNet-B0** — transfer learning desde ImageNet, últimos 2
     bloques descongelados. 4,054,737 parámetros.
   - **SSR-Net** — regresión coarse-to-fine en 3 etapas, edad continua
     convertida a rango con umbrales fijos. 88,297 parámetros.

## Resultados

### Detector de cabezas (CrowdHuman, 30 épocas, 345,423 instancias efectivas)

| Métrica | Valor |
|---|---|
| mAP50 | 67.3% |
| mAP50-95 | 39.9% |
| Precisión | 82.6% |
| Recall | 60.9% |

La precisión es sensiblemente mayor que el recall: el modelo es
conservador, prioriza no marcar falsos positivos por sobre encontrar
todas las cabezas presentes — preferible para aforo (subcontar es más
seguro que sobrecontar). El umbral de confianza con mejor balance
precisión/recall es 0.311 (F1=0.70), más permisivo que el `conf=0.35`
usado por defecto en los scripts de inferencia.

### Clasificación de edad (UTKFace, 4 clases, validación)

| Red | Parámetros | Accuracy | MAE (años) | Tiempo inf. |
|---|---|---|---|---|
| **CNN propia** | 421,828 | **68.3%** | — | 1.26 s |
| EfficientNet-B0 | 4,054,737 | 67.8% | — | 1.16 s |
| SSR-Net | **88,297** | — | 9.55 | **0.46 s** |

La CNN propia obtuvo la mejor accuracy pese a tener ~10x menos
parámetros que EfficientNet-B0 — el dominio de la tarea (rostros
pequeños, baja resolución) difiere del dominio de ImageNet, y el
*fine-tuning* de EfficientNet-B0 corrió menos épocas (10 vs 20).
SSR-Net es la más liviana y rápida en inferencia (2.7x más que la CNN
propia), y descarta explícitamente rostros de baja calidad en vez de
forzar una predicción — comportamiento más robusto, verificado en
pruebas manuales, que se replicó luego en la CNN propia agregando el
mismo criterio de descarte.

## Estructura del repositorio

```
.
├── app.py                          # demo web (Streamlit)
├── requirements.txt
├── crowdhuman.yaml
├── src/
│   ├── preparar_crowdhuman.py      # ODGT -> formato YOLO (hbox)
│   ├── preparar_utkface.py         # organiza UTKFace en 4 rangos
│   ├── entrenar_detector.py        # entrena YOLO sobre CrowdHuman
│   ├── entrenar_clasificador.py    # entrena la CNN propia
│   ├── entrenar_clasificador_effi.py   # entrena EfficientNet-B0
│   ├── entrenar_clasificador_ssr.py    # entrena SSR-Net
│   ├── detectar_edades.py          # inferencia: YOLO + CNN propia
│   ├── detectar_edades_effi.py     # inferencia: YOLO + MediaPipe + EfficientNet
│   └── detectar_edades_ssr.py      # inferencia: YOLO + MediaPipe + SSR-Net
└── models/
    ├── detector/best_head.pt
    ├── clasificador/best_age.pth
    ├── clasificador_effi/best_age.pth
    ├── clasificador_ssr/best_age.pth
    └── mediapipe/blaze_face_short_range.tflite
```

## Datos

| Dataset | Uso | Tamaño usado | Licencia |
|---|---|---|---|
| [CrowdHuman](https://www.crowdhuman.org/) | Detección de cabezas | 15,000 train (train01) / 4,370 val | Académica / no comercial |
| [UTKFace](https://susanqq.github.io/UTKFace/) | Clasificación de edad | 16,872 train / 3,615 val | Investigación |

Los datasets **no se incluyen en este repositorio** por su tamaño. Los
`.odgt`/imágenes de CrowdHuman se descargan desde el sitio oficial o
[Hugging Face](https://huggingface.co/datasets/sshao0516/CrowdHuman);
UTKFace desde su página oficial.

**Limitaciones conocidas:** las imágenes de CrowdHuman están tomadas
mayormente a la altura de los ojos, mientras que una cámara de
vigilancia real se ubica en el techo con ángulo picado — desfase de
dominio no evaluado aún cuantitativamente con video propio. El
desbalance de clases en UTKFace (`16_35` concentra 7,921 de 16,872
imágenes de train) se mitigó con pesos por clase en la función de
pérdida.

## Modelos entrenados

Los pesos (`best_head.pt`, `best_age.pth` x3) **están incluidos
directamente en este repositorio**, dentro de `models/`, listos para
usar sin descargas adicionales.

## Instalación

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

Requiere Python 3.11 (algunas dependencias, como OpenCV con soporte
Caffe legacy, no son compatibles con versiones más nuevas al momento de
este proyecto).

## Cómo correr

### Preparar los datos (solo si se va a reentrenar)

```bash
python src/preparar_crowdhuman.py
python src/preparar_utkface.py
```

### Entrenar

```bash
python src/entrenar_detector.py
python src/entrenar_clasificador.py
python src/entrenar_clasificador_effi.py
python src/entrenar_clasificador_ssr.py
```

### Demo web (recomendado para la defensa)

```bash
streamlit run app.py
```

Abre una interfaz en el navegador donde se puede subir una foto o usar la
cámara en vivo, comparando las 3 redes lado a lado.

## Trabajo futuro

- Validar el pipeline con video real grabado en pabellones feriales, para
  medir el desfase de dominio respecto a CrowdHuman en condiciones de
  cámara cenital.
- Incorporar seguimiento (*tracking*) multi-objeto sobre las
  detecciones, para pasar de conteo por fotograma a tiempo de
  permanencia por zona.
- Ampliar el entrenamiento del detector a las tres particiones
  completas de CrowdHuman (train01-03).

## Equipo

- Grace Linda Romero Arancibia
- Guery Sanz Guerrero Selaez

## Referencias

- Shao, S., Zhao, Z., Li, B., Xiao, T., Yu, G., Zhang, X., & Sun, J. (2018).
  *CrowdHuman: A Benchmark for Detecting Human in a Crowd*. arXiv:1805.00123.
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).
  *You Only Look Once: Unified, Real-Time Object Detection*. CVPR 2016.
  arXiv:1506.02640.
- Lugaresi, C., Tang, J., Nash, H., et al. (2019).
  *MediaPipe: A Framework for Building Perception Pipelines*.
  arXiv:1906.08172.
- Zhang, Z., Song, Y., & Qi, H. (2017).
  *Age Progression/Regression by Conditional Adversarial Autoencoder*.
  CVPR 2017. (Dataset UTKFace).
- Tan, M. & Le, Q. (2019).
  *EfficientNet: Rethinking Model Scaling for Convolutional Neural
  Networks*. ICML 2019. arXiv:1905.11946.
- Yang, T.-Y., Huang, Y.-H., Lin, Y.-Y., Hsiu, P.-C., & Chuang, Y.-Y. (2018).
  *SSR-Net: A Compact Soft Stagewise Regression Network for Age
  Estimation*. IJCAI 2018.
