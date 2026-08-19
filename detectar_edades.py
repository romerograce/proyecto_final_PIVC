from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from src.entrenar_clasificador import ClasificadorEdad

# raíz del proyecto y modelos finales
BASE_DIR = Path(__file__).resolve().parent.parent
DETECTOR_PATH = BASE_DIR / "models" / "detector" / "best_head.pt"
CLASIFICADOR_PATH = BASE_DIR / "models" / "clasificador" / "best_age.pth"

# selecciona automáticamente CUDA o CPU
def obtener_dispositivo():
    if torch.cuda.is_available():
        print(f"Dispositivo: CUDA | GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("Dispositivo: CPU")
    return torch.device("cpu")

# carga YOLO y la CNN entrenada
def cargar_modelos():
    if not DETECTOR_PATH.exists():
        raise FileNotFoundError(f"No se encontró el detector: {DETECTOR_PATH}")
    if not CLASIFICADOR_PATH.exists():
        raise FileNotFoundError(f"No se encontró el clasificador: {CLASIFICADOR_PATH}")

    device = obtener_dispositivo()

    # detector YOLO entrenado con CrowdHuman
    detector = YOLO(str(DETECTOR_PATH))

    # recupera pesos y clases de la CNN
    checkpoint = torch.load(CLASIFICADOR_PATH, map_location=device)
    clases = checkpoint["classes"]

    clasificador = ClasificadorEdad(num_clases=len(clases))
    clasificador.load_state_dict(checkpoint["model_state_dict"])
    clasificador.to(device)
    clasificador.eval()

    return detector, clasificador, clases, device

# prepara el recorte para la CNN
def preparar_rostro(rostro):
    transformacion = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
    rostro = Image.fromarray(rostro)
    return transformacion(rostro).unsqueeze(0)

# detecta cabezas y clasifica el rango de edad
def detectar_edades(imagen_path, confianza=0.35):
    imagen_path = Path(imagen_path)

    if not imagen_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {imagen_path}")

    detector, clasificador, clases, device = cargar_modelos()
    imagen = cv2.imread(str(imagen_path))

    if imagen is None:
        raise ValueError(f"No se pudo leer: {imagen_path}")

    # YOLO utiliza GPU cuando está disponible
    yolo_device = 0 if device.type == "cuda" else "cpu"
    resultados = detector.predict(
        source=imagen,
        conf=confianza,
        device=yolo_device,
        verbose=False
    )

    detecciones = []

    # procesa cada cabeza detectada
    for resultado in resultados:
        for caja in resultado.boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())

            # limita las coordenadas al tamaño real de la imagen
            alto, ancho = imagen.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ancho, x2), min(alto, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # recorta la cabeza detectada por YOLO
            rostro = imagen[y1:y2, x1:x2]

            if rostro.size == 0:
                continue

            entrada = preparar_rostro(rostro).to(device)

            # clasifica el rango de edad
            with torch.no_grad():
                salida = clasificador(entrada)
                probabilidades = F.softmax(salida, dim=1)
                confianza_edad, indice = probabilidades.max(1)

            clase = clases[indice.item()]
            confianza_clase = confianza_edad.item()

            detecciones.append({
                "bbox": (x1, y1, x2, y2),
                "edad": clase,
                "confianza": confianza_clase
            })

            # muestra rango y confianza sobre la imagen
            texto = f"{clase} {confianza_clase:.1%}"
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                imagen, texto, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    return imagen, detecciones