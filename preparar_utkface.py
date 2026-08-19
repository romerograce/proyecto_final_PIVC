import random
import shutil
from pathlib import Path

# raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
UTKFACE_DIR = BASE_DIR / "UTKface_inthewild"
OUTPUT_DIR = BASE_DIR / "datasets" / "utkface_edades"

# rangos utilizados por el clasificador
CLASES = {
    "01_15": (0, 15),
    "16_35": (16, 35),
    "36_55": (36, 55),
    "56_mas": (56, 116)
}

# obtiene la clase correspondiente a una edad
def obtener_clase(edad):
    for clase, (minimo, maximo) in CLASES.items():
        if minimo <= edad <= maximo:
            return clase
    return None

# busca las imágenes JPG dentro del dataset original
def buscar_imagenes():
    return list(UTKFACE_DIR.rglob("*.jpg"))

# prepara UTKFace para entrenar la CNN
def preparar_utkface(seed=42):
    print("Preparando UTKFace para clasificación de edad...")

    if not UTKFACE_DIR.exists():
        print(f"No se encontró: {UTKFACE_DIR}")
        return None

    # elimina únicamente el dataset procesado anteriormente
    if OUTPUT_DIR.exists():
        print("Eliminando dataset procesado anterior...")
        shutil.rmtree(OUTPUT_DIR)

    # crea las carpetas train, val y test para cada clase
    for split in ["train", "val", "test"]:
        for clase in CLASES:
            (OUTPUT_DIR / split / clase).mkdir(parents=True, exist_ok=True)

    # clasifica las imágenes según la edad indicada en el nombre
    por_clase = {clase: [] for clase in CLASES}
    errores = 0

    for imagen in buscar_imagenes():
        try:
            # formato UTKFace: edad_genero_raza_fecha.jpg
            edad = int(imagen.name.split("_")[0])
            clase = obtener_clase(edad)

            if clase:
                por_clase[clase].append(imagen)
            else:
                errores += 1
        except (ValueError, IndexError):
            errores += 1

    random.seed(seed)
    resultados = {}

    # divide cada clase para mantener representación en todos los conjuntos
    for clase, imagenes in por_clase.items():
        random.shuffle(imagenes)
        total = len(imagenes)

        # 70 % entrenamiento, 15 % validación y 15 % prueba
        limite_train = int(total * 0.70)
        limite_val = limite_train + int(total * 0.15)

        divisiones = {
            "train": imagenes[:limite_train],
            "val": imagenes[limite_train:limite_val],
            "test": imagenes[limite_val:]
        }

        resultados[clase] = {}

        for split, archivos in divisiones.items():
            destino = OUTPUT_DIR / split / clase

            for imagen in archivos:
                shutil.copy2(imagen, destino / imagen.name)

            resultados[clase][split] = len(archivos)

    # muestra distribución final
    print("\nDistribución del dataset:")
    for clase, datos in resultados.items():
        print(
            f"{clase:8} | "
            f"Train: {datos['train']:5} | "
            f"Val: {datos['val']:5} | "
            f"Test: {datos['test']:5}"
        )

    print(f"\nArchivos ignorados: {errores}")
    print(f"Dataset generado en: {OUTPUT_DIR}")

    return resultados

# permite ejecutar desde terminal
if __name__ == "__main__":
    preparar_utkface()