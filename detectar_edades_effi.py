from pathlib import Path

import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from ultralytics import YOLO


# raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# modelos entrenados
DETECTOR_PATH = BASE_DIR / "models" / "detector" / "best_head.pt"
CLASIFICADOR_PATH = BASE_DIR / "models" / "clasificador_effi" / "best_age.pth"

# modelo facial de MediaPipe
FACE_DETECTOR_PATH = (
    BASE_DIR
    / "models"
    / "mediapipe"
    / "blaze_face_short_range.tflite"
)

# configuración
MIN_ROSTRO = 20
MIN_FACE_CONF = 0.50
MIN_EDAD_CONF = 0.55


# selecciona automáticamente CUDA o CPU
def obtener_dispositivo():
    if torch.cuda.is_available():
        print(f"Dispositivo: CUDA | GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    print("Dispositivo: CPU")
    return torch.device("cpu")


# carga YOLO, EfficientNet y MediaPipe una sola vez
def cargar_modelos():
    # verifica que existan todos los modelos
    if not DETECTOR_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el detector YOLO: {DETECTOR_PATH}"
        )

    if not CLASIFICADOR_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró EfficientNet: {CLASIFICADOR_PATH}"
        )

    if not FACE_DETECTOR_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo facial de MediaPipe: {FACE_DETECTOR_PATH}"
        )

    device = obtener_dispositivo()

    # YOLO entrenado con CrowdHuman para detectar cabezas
    detector = YOLO(str(DETECTOR_PATH))

    # carga EfficientNet entrenada con UTKFace
    checkpoint = torch.load(
        CLASIFICADOR_PATH,
        map_location=device
    )

    clases = checkpoint["classes"]

    # reconstruye EfficientNet-B0
    clasificador = efficientnet_b0(weights=None)

    # reemplaza la salida original por nuestras clases
    clasificador.classifier[1] = torch.nn.Linear(
        clasificador.classifier[1].in_features,
        len(clases)
    )

    # carga pesos entrenados
    clasificador.load_state_dict(
        checkpoint["model_state_dict"]
    )

    clasificador = clasificador.to(device)
    clasificador.eval()

    # transformación oficial de EfficientNet-B0
    transformacion = EfficientNet_B0_Weights.DEFAULT.transforms()

    # configura MediaPipe Tasks Face Detector
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    RunningMode = mp.tasks.vision.RunningMode

    opciones = FaceDetectorOptions(
        base_options=BaseOptions(
            model_asset_path=str(FACE_DETECTOR_PATH)
        ),
        running_mode=RunningMode.IMAGE,
        min_detection_confidence=MIN_FACE_CONF
    )

    detector_rostro = FaceDetector.create_from_options(opciones)

    return (
        detector,
        clasificador,
        clases,
        transformacion,
        detector_rostro,
        device
    )


# busca el rostro dentro de la cabeza detectada por YOLO
def extraer_rostro(cabeza, detector_rostro):
    if cabeza is None or cabeza.size == 0:
        return None

    # OpenCV usa BGR y MediaPipe necesita RGB
    rgb = cv2.cvtColor(
        cabeza,
        cv2.COLOR_BGR2RGB
    )

    # convierte la imagen a formato MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # detecta el rostro
    resultado = detector_rostro.detect(mp_image)

    if not resultado.detections:
        return None

    # selecciona la detección con mayor confianza
    deteccion = max(
        resultado.detections,
        key=lambda d: d.categories[0].score
    )

    bbox = deteccion.bounding_box

    # MediaPipe devuelve coordenadas en píxeles
    x1 = bbox.origin_x
    y1 = bbox.origin_y
    x2 = bbox.origin_x + bbox.width
    y2 = bbox.origin_y + bbox.height

    alto, ancho = cabeza.shape[:2]

    # agrega margen alrededor del rostro
    margen_x = int(bbox.width * 0.15)
    margen_y = int(bbox.height * 0.15)

    x1 = max(0, x1 - margen_x)
    y1 = max(0, y1 - margen_y)
    x2 = min(ancho, x2 + margen_x)
    y2 = min(alto, y2 + margen_y)

    # valida coordenadas
    if x2 <= x1 or y2 <= y1:
        return None

    rostro = cabeza[y1:y2, x1:x2]

    if rostro.size == 0:
        return None

    return rostro


# prepara el rostro para EfficientNet-B0
def preparar_rostro(rostro, transformacion):
    # convierte BGR a RGB
    rostro = cv2.cvtColor(
        rostro,
        cv2.COLOR_BGR2RGB
    )

    rostro = Image.fromarray(rostro)

    # transforma y agrega dimensión batch
    return transformacion(rostro).unsqueeze(0)


# detecta cabezas y clasifica edades
def detectar_edades_effi(
    imagen_path,
    confianza=0.35,
    modelos=None
):
    imagen_path = Path(imagen_path)

    if not imagen_path.exists():
        raise FileNotFoundError(
            f"No se encontró la imagen: {imagen_path}"
        )

    # permite reutilizar los modelos entre varias imágenes
    if modelos is None:
        modelos = cargar_modelos()

    (
        detector,
        clasificador,
        clases,
        transformacion,
        detector_rostro,
        device
    ) = modelos

    # carga imagen
    imagen = cv2.imread(str(imagen_path))

    if imagen is None:
        raise ValueError(
            f"No se pudo leer la imagen: {imagen_path}"
        )

    alto_imagen, ancho_imagen = imagen.shape[:2]

    # YOLO usa GPU si CUDA está disponible
    yolo_device = 0 if device.type == "cuda" else "cpu"

    # detecta cabezas
    resultados = detector.predict(
        source=imagen,
        conf=confianza,
        device=yolo_device,
        verbose=False
    )

    detecciones = []

    # procesa todas las cabezas detectadas
    for resultado in resultados:
        for caja in resultado.boxes:

            x1, y1, x2, y2 = map(
                int,
                caja.xyxy[0].tolist()
            )

            # limita coordenadas al tamaño real
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(ancho_imagen, x2)
            y2 = min(alto_imagen, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # recorta la cabeza detectada
            cabeza = imagen[y1:y2, x1:x2]

            if cabeza.size == 0:
                continue

            # intenta localizar un rostro dentro de la cabeza
            rostro = extraer_rostro(
                cabeza,
                detector_rostro
            )

            ancho_rostro = 0
            alto_rostro = 0

            # si no existe rostro visible, no estima edad
            if rostro is None:
                edad = "No detectada"
                confianza_edad = 0.0

            else:
                alto_rostro, ancho_rostro = rostro.shape[:2]

                # evita clasificar rostros demasiado pequeños
                if (
                    ancho_rostro < MIN_ROSTRO
                    or alto_rostro < MIN_ROSTRO
                ):
                    edad = "No detectada"
                    confianza_edad = 0.0

                else:
                    # prepara el rostro para EfficientNet
                    entrada = preparar_rostro(
                        rostro,
                        transformacion
                    ).to(device)

                    # clasifica la edad
                    with torch.no_grad():
                        salida = clasificador(entrada)

                        probabilidades = F.softmax(
                            salida,
                            dim=1
                        )

                        conf, indice = probabilidades.max(1)

                    edad = clases[indice.item()]
                    confianza_edad = conf.item()

                    # evita aceptar predicciones con baja confianza
                    if confianza_edad < MIN_EDAD_CONF:
                        edad = "No detectada"
                        confianza_edad = 0.0

            # guarda información de la detección
            detecciones.append({
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "edad": edad,
                "confianza": confianza_edad,
                "rostro_size": (
                    ancho_rostro,
                    alto_rostro
                )
            })

            # texto que aparece sobre la imagen
            texto = (
                "No detectada"
                if edad == "No detectada"
                else f"{edad} {confianza_edad:.1%}"
            )

            # dibuja caja de cabeza
            cv2.rectangle(
                imagen,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # muestra edad o estado
            cv2.putText(
                imagen,
                texto,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    return imagen, detecciones
