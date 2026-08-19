import json
import shutil
from pathlib import Path
from PIL import Image

# raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# rutas del dataset original
CROWDHUMAN_DIR = BASE_DIR / "crowdhuman"
IMAGES_DIR = CROWDHUMAN_DIR / "Images"
TRAIN_ANNOTATIONS = CROWDHUMAN_DIR / "annotation_train.odgt"
VAL_ANNOTATIONS = CROWDHUMAN_DIR / "annotation_val.odgt"

# dataset convertido a formato YOLO
OUTPUT_DIR = BASE_DIR / "datasets" / "crowdhuman_yolo"

# crea desde cero la estructura del dataset procesado
def crear_estructura():
    if OUTPUT_DIR.exists():
        print("Eliminando dataset procesado anterior...")
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# convierte hbox [x, y, ancho, alto] al formato YOLO normalizado
def convertir_hbox(hbox, img_w, img_h):
    x, y, w, h = hbox

    # descarta cajas inválidas
    if w <= 0 or h <= 0:
        return None

    # ajusta la caja a los límites reales de la imagen
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return None

    # YOLO utiliza centro, ancho y alto normalizados entre 0 y 1
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = w / img_w
    height = h / img_h

    return x_center, y_center, width, height

# procesa train o val según el archivo de anotaciones
def procesar_split(annotation_file, split):
    total_imagenes = 0
    total_cabezas = 0
    total_ignoradas = 0
    errores = 0

    output_images = OUTPUT_DIR / "images" / split
    output_labels = OUTPUT_DIR / "labels" / split

    with open(annotation_file, "r", encoding="utf-8") as archivo:
        for linea in archivo:
            try:
                registro = json.loads(linea)
                image_id = registro["ID"]
                image_path = IMAGES_DIR / f"{image_id}.jpg"

                # verifica que exista la imagen
                if not image_path.exists():
                    errores += 1
                    print(f"[AVISO] Imagen no encontrada: {image_path}")
                    continue

                # obtiene las dimensiones reales de la imagen
                with Image.open(image_path) as img:
                    img_w, img_h = img.size

                etiquetas = []

                # recorre todas las anotaciones de la imagen
                for persona in registro.get("gtboxes", []):

                    # utiliza únicamente anotaciones correspondientes a personas
                    if persona.get("tag") != "person":
                        total_ignoradas += 1
                        continue

                    # ignora cabezas marcadas como no válidas para entrenamiento
                    if persona.get("head_attr", {}).get("ignore", 0) == 1:
                        total_ignoradas += 1
                        continue

                    # obtiene la caja de la cabeza
                    hbox = persona.get("hbox")
                    if not hbox:
                        total_ignoradas += 1
                        continue

                    # convierte la caja al formato YOLO
                    caja = convertir_hbox(hbox, img_w, img_h)
                    if caja is None:
                        total_ignoradas += 1
                        continue

                    x_center, y_center, width, height = caja

                    # clase 0 = head
                    etiquetas.append(
                        f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )
                    total_cabezas += 1

                # copia la imagen al dataset YOLO
                shutil.copy2(image_path, output_images / image_path.name)

                # genera el archivo txt correspondiente a la imagen
                label_path = output_labels / f"{image_id}.txt"
                label_path.write_text("\n".join(etiquetas), encoding="utf-8")

                total_imagenes += 1

                # muestra avance cada 500 imágenes
                if total_imagenes % 500 == 0:
                    print(
                        f"[{split.upper()}] {total_imagenes} imágenes | "
                        f"{total_cabezas} cabezas | "
                        f"{total_ignoradas} ignoradas"
                    )

            except Exception as e:
                errores += 1
                print(f"[ERROR] {e}")

    print(
        f"\n{split.upper()} finalizado | "
        f"Imágenes: {total_imagenes} | "
        f"Cabezas: {total_cabezas} | "
        f"Ignoradas: {total_ignoradas} | "
        f"Errores: {errores}"
    )

    return {
        "split": split,
        "imagenes": total_imagenes,
        "cabezas": total_cabezas,
        "ignoradas": total_ignoradas,
        "errores": errores
    }

# función principal reutilizable desde terminal o Jupyter
def preparar_crowdhuman():
    print("Preparando CrowdHuman para detección de cabezas con YOLO...")

    # valida que existan las rutas necesarias antes de comenzar
    rutas_requeridas = [
        IMAGES_DIR,
        TRAIN_ANNOTATIONS,
        VAL_ANNOTATIONS
    ]

    faltantes = [ruta for ruta in rutas_requeridas if not ruta.exists()]

    if faltantes:
        print("\nNo se encontraron las siguientes rutas:")
        for ruta in faltantes:
            print(f"- {ruta}")
        return None

    # elimina el dataset anterior y crea uno nuevo
    crear_estructura()

    # procesa entrenamiento y validación
    resultado_train = procesar_split(TRAIN_ANNOTATIONS, "train")
    resultado_val = procesar_split(VAL_ANNOTATIONS, "val")

    print(f"\nDataset generado en: {OUTPUT_DIR}")

    return {
        "train": resultado_train,
        "val": resultado_val,
        "output": OUTPUT_DIR
    }

# permite ejecutar el archivo directamente desde terminal
if __name__ == "__main__":
    preparar_crowdhuman()
