from pathlib import Path

import cv2
import mediapipe as mp
import torch

from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# importa exactamente la arquitectura usada durante el entrenamiento
from src.entrenar_clasificador_ssr import SSRNetEdad


# raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# modelos entrenados
DETECTOR_PATH = BASE_DIR / "models" / "detector" / "best_head.pt"
CLASIFICADOR_PATH = BASE_DIR / "models" / "clasificador_ssr" / "best_age.pth"

# detector facial MediaPipe
FACE_DETECTOR_PATH = (
    BASE_DIR
    / "models"
    / "mediapipe"
    / "blaze_face_short_range.tflite"
)

# configuración
MIN_ROSTRO = 20
MIN_FACE_CONF = 0.50


# selecciona automáticamente CUDA o CPU
def obtener_dispositivo():
    if torch.cuda.is_available():
        print(
            f"Dispositivo: CUDA | "
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        return torch.device("cuda")

    print("Dispositivo: CPU")
    return torch.device("cpu")


# convierte edad numérica a nuestros cuatro rangos
def edad_a_rango(edad):
    if edad <= 15:
        return "01_15"

    if edad <= 35:
        return "16_35"

    if edad <= 55:
        return "36_55"

    return "56_mas"


# carga YOLO, SSR-Net y MediaPipe una sola vez
def cargar_modelos():
    # verifica modelos
    if not DETECTOR_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró YOLO: {DETECTOR_PATH}"
        )

    if not CLASIFICADOR_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró SSR-Net: {CLASIFICADOR_PATH}"
        )

    if not FACE_DETECTOR_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró MediaPipe: {FACE_DETECTOR_PATH}"
        )

    device = obtener_dispositivo()

    # YOLO entrenado con CrowdHuman
    detector = YOLO(
        str(DETECTOR_PATH)
    )

    # reconstruye SSR-Net
    clasificador = SSRNetEdad()

    # carga checkpoint entrenado
    checkpoint = torch.load(
        CLASIFICADOR_PATH,
        map_location=device
    )

    clasificador.load_state_dict(
        checkpoint["model_state_dict"]
    )

    clasificador = clasificador.to(device)
    clasificador.eval()

    # transformación utilizada en validación
    transformacion = transforms.Compose([
        transforms.Resize((64, 64)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # MediaPipe Tasks
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

    detector_rostro = FaceDetector.create_from_options(
        opciones
    )

    return (
        detector,
        clasificador,
        transformacion,
        detector_rostro,
        device
    )


# localiza rostro dentro de la cabeza encontrada por YOLO
def extraer_rostro(cabeza, detector_rostro):
    if cabeza is None or cabeza.size == 0:
        return None

    # BGR → RGB
    rgb = cv2.cvtColor(
        cabeza,
        cv2.COLOR_BGR2RGB
    )

    # convierte a imagen MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # detección facial
    resultado = detector_rostro.detect(
        mp_image
    )

    if not resultado.detections:
        return None

    # selecciona el rostro de mayor confianza
    deteccion = max(
        resultado.detections,
        key=lambda d: d.categories[0].score
    )

    bbox = deteccion.bounding_box

    x1 = bbox.origin_x
    y1 = bbox.origin_y

    x2 = bbox.origin_x + bbox.width
    y2 = bbox.origin_y + bbox.height

    alto, ancho = cabeza.shape[:2]

    # margen alrededor del rostro
    margen_x = int(
        bbox.width * 0.15
    )

    margen_y = int(
        bbox.height * 0.15
    )

    x1 = max(
        0,
        x1 - margen_x
    )

    y1 = max(
        0,
        y1 - margen_y
    )

    x2 = min(
        ancho,
        x2 + margen_x
    )

    y2 = min(
        alto,
        y2 + margen_y
    )

    # valida coordenadas
    if x2 <= x1 or y2 <= y1:
        return None

    rostro = cabeza[
        y1:y2,
        x1:x2
    ]

    if rostro.size == 0:
        return None

    return rostro


# prepara rostro para SSR-Net
def preparar_rostro(
    rostro,
    transformacion
):
    # OpenCV → RGB
    rostro = cv2.cvtColor(
        rostro,
        cv2.COLOR_BGR2RGB
    )

    # OpenCV → PIL
    rostro = Image.fromarray(
        rostro
    )

    # 64x64 + tensor + normalización
    return transformacion(
        rostro
    ).unsqueeze(0)


# pipeline completo YOLO + MediaPipe + SSR-Net
def detectar_edades_ssr(
    imagen_path,
    confianza=0.35,
    modelos=None
):
    imagen_path = Path(
        imagen_path
    )

    if not imagen_path.exists():
        raise FileNotFoundError(
            f"No se encontró: {imagen_path}"
        )

    # permite reutilizar modelos para varias imágenes
    if modelos is None:
        modelos = cargar_modelos()

    (
        detector,
        clasificador,
        transformacion,
        detector_rostro,
        device
    ) = modelos

    # carga imagen original
    imagen = cv2.imread(
        str(imagen_path)
    )

    if imagen is None:
        raise ValueError(
            f"No se pudo leer: {imagen_path}"
        )

    alto_imagen, ancho_imagen = (
        imagen.shape[:2]
    )

    # YOLO usa CUDA cuando está disponible
    yolo_device = (
        0
        if device.type == "cuda"
        else "cpu"
    )

    # detecta cabezas
    resultados = detector.predict(
        source=imagen,
        conf=confianza,
        device=yolo_device,
        verbose=False
    )

    detecciones = []

    # procesa todas las cabezas
    for resultado in resultados:
        for caja in resultado.boxes:

            x1, y1, x2, y2 = map(
                int,
                caja.xyxy[0].tolist()
            )

            # limita caja a la imagen
            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                ancho_imagen,
                x2
            )

            y2 = min(
                alto_imagen,
                y2
            )

            if x2 <= x1 or y2 <= y1:
                continue

            # recorta cabeza
            cabeza = imagen[
                y1:y2,
                x1:x2
            ]

            if cabeza.size == 0:
                continue

            # busca rostro dentro de la cabeza
            rostro = extraer_rostro(
                cabeza,
                detector_rostro
            )

            ancho_rostro = 0
            alto_rostro = 0

            edad_estimada = None
            rango = "No detectada"
            motivo = ""

            # no existe rostro visible
            if rostro is None:
                motivo = "rostro no encontrado"

            else:
                alto_rostro, ancho_rostro = (
                    rostro.shape[:2]
                )

                # evita rostros extremadamente pequeños
                if (
                    ancho_rostro < MIN_ROSTRO
                    or alto_rostro < MIN_ROSTRO
                ):
                    motivo = "rostro demasiado pequeño"

                else:
                    # prepara rostro
                    entrada = preparar_rostro(
                        rostro,
                        transformacion
                    ).to(device)

                    # estima edad numérica
                    with torch.no_grad():
                        salida = clasificador(
                            entrada
                        )

                    edad_estimada = float(
                        salida.item()
                    )

                    # limita posibles valores extremos
                    edad_estimada = max(
                        0.0,
                        min(
                            116.0,
                            edad_estimada
                        )
                    )

                    # convierte edad a rango
                    rango = edad_a_rango(
                        edad_estimada
                    )

            # guarda resultado
            detecciones.append({
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "edad_estimada":
                    edad_estimada,

                "edad":
                    rango,

                "rostro_size": (
                    ancho_rostro,
                    alto_rostro
                ),

                "motivo":
                    motivo
            })

            # texto sobre la imagen
            if edad_estimada is None:
                texto = "No detectada"

            else:
                texto = (
                    f"{edad_estimada:.0f} anos "
                    f"({rango})"
                )

            # caja YOLO
            cv2.rectangle(
                imagen,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # resultado
            cv2.putText(
                imagen,
                texto,
                (
                    x1,
                    max(
                        20,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 0),
                2
            )

    return imagen, detecciones
