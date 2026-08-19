from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# dataset UTKFace ya preparado
DATASET_DIR = BASE_DIR / "datasets" / "utkface_edades"

# salida del modelo SSR-Net
MODELS_DIR = BASE_DIR / "models" / "clasificador_ssr"

# edad máxima manejada por UTKFace
EDAD_MAXIMA = 116.0


# selecciona automáticamente CUDA o CPU
def obtener_dispositivo():
    if torch.cuda.is_available():
        print(f"Dispositivo: CUDA | GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    print("Dispositivo: CPU")
    return torch.device("cpu")


# dataset de regresión: obtiene la edad desde el nombre del archivo
class UTKFaceEdadDataset(Dataset):
    def __init__(self, carpeta, transform=None):
        self.carpeta = Path(carpeta)
        self.transform = transform
        self.imagenes = []

        # busca imágenes dentro de todas las subcarpetas de edad
        for extension in ("*.jpg", "*.jpeg", "*.png"):
            self.imagenes.extend(
                self.carpeta.rglob(extension)
            )

        # conserva solo imágenes cuyo nombre comienza con una edad válida
        validas = []

        for imagen in self.imagenes:
            try:
                edad = int(imagen.name.split("_")[0])

                if 0 <= edad <= EDAD_MAXIMA:
                    validas.append(imagen)

            except (ValueError, IndexError):
                continue

        self.imagenes = validas

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, indice):
        ruta = self.imagenes[indice]

        # edad real incluida en el nombre UTKFace
        edad = float(ruta.name.split("_")[0])

        imagen = Image.open(ruta).convert("RGB")

        if self.transform:
            imagen = self.transform(imagen)

        # SSR-Net trabaja como regresión
        edad = torch.tensor(
            edad,
            dtype=torch.float32
        )

        return imagen, edad


# bloque convolucional compacto
class BloqueSSR(nn.Module):
    def __init__(self, entrada, salida):
        super().__init__()

        self.bloque = nn.Sequential(
            nn.Conv2d(
                entrada,
                salida,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                salida,
                salida,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.bloque(x)


# versión PyTorch compacta inspirada en SSR-Net
class SSRNetEdad(nn.Module):
    def __init__(self):
        super().__init__()

        # extracción progresiva de características
        self.stage1 = BloqueSSR(3, 32)
        self.stage2 = BloqueSSR(32, 32)
        self.stage3 = BloqueSSR(32, 64)

        # reducción espacial
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # cada etapa genera una estimación parcial
        self.regresor1 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.regresor2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.regresor3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # primera escala
        x1 = self.stage1(x)

        # segunda escala
        x2 = self.stage2(x1)

        # tercera escala
        x3 = self.stage3(x2)

        # convierte mapas a vectores
        f1 = self.pool(x1).flatten(1)
        f2 = self.pool(x2).flatten(1)
        f3 = self.pool(x3).flatten(1)

        # estimaciones coarse-to-fine
        e1 = self.regresor1(f1)
        e2 = self.regresor2(f2)
        e3 = self.regresor3(f3)

        # combinación progresiva
        edad = (
            0.20 * e1
            + 0.30 * e2
            + 0.50 * e3
        )

        # restringe la salida a un rango razonable
        edad = torch.sigmoid(edad) * EDAD_MAXIMA

        return edad.squeeze(1)


# transformaciones de entrenamiento
def crear_transformaciones():
    # SSR-Net PyTorch de referencia trabaja con entrada 64x64
    transform_train = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.RandomResizedCrop(
            64,
            scale=(0.55, 1.0)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),

        # simula rostros CrowdHuman pequeños
        transforms.RandomApply([
            transforms.Resize((32, 32)),
            transforms.Resize((64, 64))
        ], p=0.35),

        transforms.RandomApply([
            transforms.Resize((20, 20)),
            transforms.Resize((64, 64))
        ], p=0.20),

        # agrega degradación moderada
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.2)
            )
        ], p=0.20),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15
        ),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # validación sin augmentation aleatorio
    transform_val = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform_train, transform_val


# una época de entrenamiento
def entrenar_epoca(
    modelo,
    loader,
    criterio,
    optimizador,
    device
):
    modelo.train()

    perdida_total = 0.0
    error_total = 0.0
    total = 0

    for imagenes, edades in loader:
        imagenes = imagenes.to(
            device,
            non_blocking=True
        )

        edades = edades.to(
            device,
            non_blocking=True
        )

        optimizador.zero_grad()

        predicciones = modelo(imagenes)

        # MAE / L1
        perdida = criterio(
            predicciones,
            edades
        )

        perdida.backward()
        optimizador.step()

        cantidad = edades.size(0)

        perdida_total += (
            perdida.item()
            * cantidad
        )

        error_total += torch.abs(
            predicciones - edades
        ).sum().item()

        total += cantidad

    loss = perdida_total / total
    mae = error_total / total

    return loss, mae


# evaluación sobre validación
def evaluar(
    modelo,
    loader,
    criterio,
    device
):
    modelo.eval()

    perdida_total = 0.0
    error_total = 0.0
    total = 0

    with torch.no_grad():
        for imagenes, edades in loader:
            imagenes = imagenes.to(
                device,
                non_blocking=True
            )

            edades = edades.to(
                device,
                non_blocking=True
            )

            predicciones = modelo(imagenes)

            perdida = criterio(
                predicciones,
                edades
            )

            cantidad = edades.size(0)

            perdida_total += (
                perdida.item()
                * cantidad
            )

            error_total += torch.abs(
                predicciones - edades
            ).sum().item()

            total += cantidad

    loss = perdida_total / total
    mae = error_total / total

    return loss, mae


# entrenamiento principal
def entrenar_clasificador_ssr(
    epochs=60,
    batch_size=50,
    learning_rate=0.001,
    weight_decay=0.0001,
    workers=4,
    patience=10
):
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"No existe: {train_dir}"
        )

    if not val_dir.exists():
        raise FileNotFoundError(
            f"No existe: {val_dir}"
        )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = obtener_dispositivo()

    transform_train, transform_val = (
        crear_transformaciones()
    )

    train_dataset = UTKFaceEdadDataset(
        train_dir,
        transform_train
    )

    val_dataset = UTKFaceEdadDataset(
        val_dir,
        transform_val
    )

    usar_cuda = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=usar_cuda
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=usar_cuda
    )

    modelo = SSRNetEdad().to(device)

    # SSR-Net PyTorch de referencia utiliza L1Loss
    criterio = nn.L1Loss()

    optimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # reduce LR durante entrenamiento
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizador,
        step_size=20,
        gamma=0.1
    )

    print("\nConfiguración SSR-Net:")
    print(
        f"  Train: {len(train_dataset)} imágenes"
    )
    print(
        f"  Val:   {len(val_dataset)} imágenes"
    )
    print(
        f"  Entrada: 64x64"
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

    total_parametros = sum(
        p.numel()
        for p in modelo.parameters()
    )

    print(
        f"  Parámetros: {total_parametros:,}"
    )

    mejor_mae = float("inf")
    mejores_pesos = copy.deepcopy(
        modelo.state_dict()
    )

    historial = []
    sin_mejora = 0

    for epoch in range(
        1,
        epochs + 1
    ):
        train_loss, train_mae = entrenar_epoca(
            modelo,
            train_loader,
            criterio,
            optimizador,
            device
        )

        val_loss, val_mae = evaluar(
            modelo,
            val_loader,
            criterio,
            device
        )

        scheduler.step()

        historial.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_mae": val_mae
        })

        print(
            f"Época {epoch:02d}/{epochs} | "
            f"Train MAE: {train_mae:.2f} años | "
            f"Val MAE: {val_mae:.2f} años | "
            f"LR: {optimizador.param_groups[0]['lr']:.6f}"
        )

        # en regresión, menor MAE es mejor
        if val_mae < mejor_mae:
            mejor_mae = val_mae

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

        if sin_mejora >= patience:
            print(
                "\nEarly stopping."
            )
            break

    # guarda último modelo
    torch.save(
        {
            "model_state_dict":
                modelo.state_dict(),

            "architecture":
                "ssrnet_age",

            "input_size":
                64,

            "max_age":
                EDAD_MAXIMA
        },
        MODELS_DIR / "last_age.pth"
    )

    # restaura mejor modelo
    modelo.load_state_dict(
        mejores_pesos
    )

    # guarda mejor modelo
    torch.save(
        {
            "model_state_dict":
                modelo.state_dict(),

            "architecture":
                "ssrnet_age",

            "input_size":
                64,

            "max_age":
                EDAD_MAXIMA
        },
        MODELS_DIR / "best_age.pth"
    )

    print("\nEntrenamiento finalizado.")

    print(
        f"Mejor MAE validación: "
        f"{mejor_mae:.2f} años"
    )

    print(
        f"Modelo: "
        f"{MODELS_DIR / 'best_age.pth'}"
    )

    return modelo, historial


# permite ejecutar desde terminal
if __name__ == "__main__":
    entrenar_clasificador_ssr()