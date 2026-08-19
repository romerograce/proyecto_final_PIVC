from pathlib import Path
import copy

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# dataset UTKFace previamente preparado
DATASET_DIR = BASE_DIR / "datasets" / "utkface_edades"

# modelos generados por EfficientNet
MODELS_DIR = BASE_DIR / "models" / "clasificador_effi"


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


# crea las transformaciones de entrenamiento y validación
def crear_transformaciones():
    # entrenamiento:
    # se agregan degradaciones para aproximarse a rostros pequeños,
    # borrosos y con condiciones menos ideales como CrowdHuman
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),

        # modifica encuadre y escala del rostro
        transforms.RandomResizedCrop(
            224,
            scale=(0.60, 1.0)
        ),

        # variaciones normales de orientación
        transforms.RandomHorizontalFlip(p=0.5),

        # pequeñas variaciones de inclinación
        transforms.RandomRotation(8),

        # cambios moderados de iluminación
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.10
        ),

        # simula rostros de baja resolución
        transforms.RandomApply([
            transforms.Resize((64, 64)),
            transforms.Resize((224, 224))
        ], p=0.35),

        # simula desenfoque moderado
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.5)
            )
        ], p=0.25),

        transforms.ToTensor(),

        # normalización utilizada por los pesos de ImageNet
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # validación:
    # no utiliza transformaciones aleatorias
    weights = EfficientNet_B0_Weights.DEFAULT
    transform_val = weights.transforms()

    return transform_train, transform_val


# calcula pesos para compensar el desbalance de las clases
def calcular_pesos(dataset, device):
    targets = torch.tensor(
        dataset.targets,
        dtype=torch.long
    )

    conteos = torch.bincount(targets)

    pesos = len(dataset) / (
        len(conteos) * conteos.float()
    )

    print("\nDistribución de entrenamiento:")

    for clase, cantidad in zip(
        dataset.classes,
        conteos.tolist()
    ):
        print(
            f"  {clase:8}: "
            f"{cantidad:5} imágenes"
        )

    print("\nPesos utilizados:")

    for clase, peso in zip(
        dataset.classes,
        pesos.tolist()
    ):
        print(
            f"  {clase:8}: "
            f"{peso:.4f}"
        )

    return pesos.to(device)


# crea EfficientNet-B0 mediante transfer learning
def crear_modelo(num_clases):
    # pesos preentrenados de ImageNet
    weights = EfficientNet_B0_Weights.DEFAULT

    modelo = efficientnet_b0(
        weights=weights
    )

    # congela inicialmente todo el extractor
    for parametro in modelo.features.parameters():
        parametro.requires_grad = False

    # descongela los dos últimos bloques
    # para adaptar características a edad facial
    for parametro in modelo.features[-2:].parameters():
        parametro.requires_grad = True

    # reemplaza la capa final original
    entradas = modelo.classifier[1].in_features

    modelo.classifier[1] = nn.Linear(
        entradas,
        num_clases
    )

    return modelo


# ejecuta una época de entrenamiento
def entrenar_epoca(
    modelo,
    loader,
    criterio,
    optimizador,
    device
):
    modelo.train()

    perdida_total = 0.0
    correctos = 0
    total = 0

    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(
            device,
            non_blocking=True
        )

        etiquetas = etiquetas.to(
            device,
            non_blocking=True
        )

        # limpia gradientes
        optimizador.zero_grad()

        # forward
        salida = modelo(imagenes)

        # calcula pérdida
        perdida = criterio(
            salida,
            etiquetas
        )

        # backward
        perdida.backward()

        # actualiza parámetros
        optimizador.step()

        perdida_total += (
            perdida.item()
            * imagenes.size(0)
        )

        predicciones = salida.argmax(dim=1)

        correctos += (
            predicciones == etiquetas
        ).sum().item()

        total += etiquetas.size(0)

    perdida_media = perdida_total / total
    accuracy = correctos / total

    return perdida_media, accuracy


# evalúa el modelo sin modificar sus pesos
def evaluar(
    modelo,
    loader,
    criterio,
    device
):
    modelo.eval()

    perdida_total = 0.0
    correctos = 0
    total = 0

    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes = imagenes.to(
                device,
                non_blocking=True
            )

            etiquetas = etiquetas.to(
                device,
                non_blocking=True
            )

            salida = modelo(imagenes)

            perdida = criterio(
                salida,
                etiquetas
            )

            perdida_total += (
                perdida.item()
                * imagenes.size(0)
            )

            predicciones = salida.argmax(dim=1)

            correctos += (
                predicciones == etiquetas
            ).sum().item()

            total += etiquetas.size(0)

    perdida_media = perdida_total / total
    accuracy = correctos / total

    return perdida_media, accuracy


# entrenamiento principal de EfficientNet-B0
def entrenar_clasificador_effi(
    epochs=15,
    batch_size=32,
    learning_rate=0.0001,
    workers=4,
    patience=4
):
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    # verifica dataset
    if not train_dir.exists():
        raise FileNotFoundError(
            f"No existe: {train_dir}\n"
            "Primero ejecuta preparar_utkface.py"
        )

    if not val_dir.exists():
        raise FileNotFoundError(
            f"No existe: {val_dir}\n"
            "Primero ejecuta preparar_utkface.py"
        )

    # crea carpeta de salida
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # selecciona GPU o CPU
    device = obtener_dispositivo()

    # transformaciones
    (
        transform_train,
        transform_val
    ) = crear_transformaciones()

    # datasets
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=transform_train
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=transform_val
    )

    # verifica que ambos conjuntos tengan las mismas clases
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Las clases de train y val no coinciden."
        )

    usar_cuda = device.type == "cuda"

    # cargador de entrenamiento
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=usar_cuda
    )

    # cargador de validación
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=usar_cuda
    )

    # crea EfficientNet
    modelo = crear_modelo(
        len(train_dataset.classes)
    )

    modelo = modelo.to(device)

    # pesos para compensar desbalance
    pesos = calcular_pesos(
        train_dataset,
        device
    )

    # función de pérdida
    criterio = nn.CrossEntropyLoss(
        weight=pesos
    )

    # únicamente parámetros descongelados
    parametros_entrenables = [
        parametro
        for parametro in modelo.parameters()
        if parametro.requires_grad
    ]

    # optimizador con learning rate bajo
    optimizador = torch.optim.Adam(
        parametros_entrenables,
        lr=learning_rate
    )

    print("\nConfiguración:")
    print("  Modelo: EfficientNet-B0")
    print("  Transfer Learning: Sí")
    print("  Fine-tuning: últimos 2 bloques")
    print(
        f"  Train: {len(train_dataset)} imágenes"
    )
    print(
        f"  Val:   {len(val_dataset)} imágenes"
    )
    print(
        f"  Clases: {train_dataset.classes}"
    )
    print(
        f"  Batch: {batch_size}"
    )
    print(
        f"  Learning rate: {learning_rate}"
    )
    print(
        f"  Épocas máximas: {epochs}"
    )
    print(
        f"  Early stopping: {patience}"
    )

    # muestra cantidad de parámetros
    total_parametros = sum(
        p.numel()
        for p in modelo.parameters()
    )

    parametros_entrenables_num = sum(
        p.numel()
        for p in modelo.parameters()
        if p.requires_grad
    )

    print(
        f"  Parámetros totales: "
        f"{total_parametros:,}"
    )

    print(
        f"  Parámetros entrenables: "
        f"{parametros_entrenables_num:,}"
    )

    # variables de seguimiento
    mejor_accuracy = 0.0

    mejores_pesos = copy.deepcopy(
        modelo.state_dict()
    )

    historial = []

    sin_mejora = 0

    # entrenamiento
    for epoch in range(
        1,
        epochs + 1
    ):
        train_loss, train_acc = entrenar_epoca(
            modelo,
            train_loader,
            criterio,
            optimizador,
            device
        )

        val_loss, val_acc = evaluar(
            modelo,
            val_loader,
            criterio,
            device
        )

        historial.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(
            f"\nÉpoca {epoch:02d}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

        # guarda el mejor estado en memoria
        if val_acc > mejor_accuracy:
            mejor_accuracy = val_acc

            mejores_pesos = copy.deepcopy(
                modelo.state_dict()
            )

            sin_mejora = 0

            print(
                "  -> Nuevo mejor modelo"
            )

        else:
            sin_mejora += 1

            print(
                f"  -> Sin mejora: "
                f"{sin_mejora}/{patience}"
            )

        # early stopping
        if sin_mejora >= patience:
            print(
                "\nEarly stopping: "
                f"{patience} épocas sin mejora."
            )
            break

    # guarda el último estado
    torch.save(
        {
            "model_state_dict":
                modelo.state_dict(),

            "classes":
                train_dataset.classes,

            "architecture":
                "efficientnet_b0",

            "fine_tuning":
                "last_2_blocks",

            "image_size":
                224
        },
        MODELS_DIR / "last_age.pth"
    )

    # recupera el mejor modelo
    modelo.load_state_dict(
        mejores_pesos
    )

    # guarda el mejor modelo
    torch.save(
        {
            "model_state_dict":
                modelo.state_dict(),

            "classes":
                train_dataset.classes,

            "architecture":
                "efficientnet_b0",

            "fine_tuning":
                "last_2_blocks",

            "image_size":
                224
        },
        MODELS_DIR / "best_age.pth"
    )

    print("\nEntrenamiento finalizado.")

    print(
        f"Mejor accuracy validación: "
        f"{mejor_accuracy:.4f} "
        f"({mejor_accuracy:.2%})"
    )

    print(
        "Mejor modelo:"
    )

    print(
        MODELS_DIR / "best_age.pth"
    )

    return (
        modelo,
        historial,
        train_dataset.classes
    )


# permite ejecutar directamente desde terminal
if __name__ == "__main__":
    entrenar_clasificador_effi(
        epochs=15,
        batch_size=32,
        learning_rate=0.0001,
        workers=4,
        patience=4
    )
