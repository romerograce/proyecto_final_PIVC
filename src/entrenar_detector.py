from pathlib import Path
import shutil
import torch
from ultralytics import YOLO

# raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# rutas principales
DATASET_YAML = BASE_DIR / "crowdhuman.yaml"
MODELS_DIR = BASE_DIR / "models" / "detector"
RUNS_DIR = BASE_DIR / "runs_detector"

# selecciona automáticamente CUDA o CPU
def obtener_dispositivo():
    if torch.cuda.is_available():
        print(f"Dispositivo: CUDA | GPU: {torch.cuda.get_device_name(0)}")
        return 0

    print("Dispositivo: CPU")
    return "cpu"

# entrena el detector de cabezas
def entrenar_detector(
    modelo="yolo11n.pt",
    epochs=20,
    imgsz=640,
    batch=8,
    workers=4,
    patience=5
):
    # valida que exista la configuración del dataset
    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"No se encontró: {DATASET_YAML}")

    device = obtener_dispositivo()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Modelo base: {modelo}")
    print(f"Épocas: {epochs}")
    print(f"Imagen: {imgsz}x{imgsz}")
    print(f"Batch: {batch}")
    print("Iniciando entrenamiento...\n")

    # carga pesos preentrenados para realizar fine-tuning
    model = YOLO(modelo)

    # entrena y valida automáticamente en cada época
    resultados = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        pretrained=True,
        patience=patience,
        project=str(RUNS_DIR),
        name="crowdhuman_head",
        exist_ok=True,
        verbose=True
    )

    # ubicación generada por Ultralytics
    best_original = RUNS_DIR / "crowdhuman_head" / "weights" / "best.pt"
    last_original = RUNS_DIR / "crowdhuman_head" / "weights" / "last.pt"

    # copia los modelos importantes a models/detector
    if best_original.exists():
        shutil.copy2(best_original, MODELS_DIR / "best_head.pt")
        print(f"\nMejor modelo: {MODELS_DIR / 'best_head.pt'}")

    if last_original.exists():
        shutil.copy2(last_original, MODELS_DIR / "last_head.pt")

    return resultados

# permite ejecutar desde terminal
if __name__ == "__main__":
    entrenar_detector()
