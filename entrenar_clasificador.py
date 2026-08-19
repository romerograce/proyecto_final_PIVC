from pathlib import Path
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "utkface_edades"
MODELS_DIR = BASE_DIR / "models" / "clasificador"

# CNN para clasificación de los cuatro rangos de edad
class ClasificadorEdad(nn.Module):
    def __init__(self, num_clases=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_clases)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# selecciona CUDA si existe; de lo contrario utiliza CPU
def obtener_dispositivo():
    if torch.cuda.is_available():
        print(f"Dispositivo: CUDA | GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("Dispositivo: CPU")
    return torch.device("cpu")

# calcula pesos para compensar diferencias entre las clases
def calcular_pesos(dataset, device):
    conteos = torch.bincount(torch.tensor(dataset.targets))
    pesos = len(dataset) / (len(conteos) * conteos.float())
    print("Imágenes por clase:", dict(zip(dataset.classes, conteos.tolist())))
    return pesos.to(device)

# ejecuta una época de entrenamiento
def entrenar_epoca(modelo, loader, criterio, optimizador, device):
    modelo.train()
    perdida_total, correctos, total = 0.0, 0, 0

    for imagenes, etiquetas in loader:
        imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
        optimizador.zero_grad()
        salida = modelo(imagenes)
        perdida = criterio(salida, etiquetas)
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * imagenes.size(0)
        correctos += (salida.argmax(1) == etiquetas).sum().item()
        total += etiquetas.size(0)

    return perdida_total / total, correctos / total

# evalúa el modelo sin modificar sus pesos
def evaluar(modelo, loader, criterio, device):
    modelo.eval()
    perdida_total, correctos, total = 0.0, 0, 0

    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            salida = modelo(imagenes)
            perdida = criterio(salida, etiquetas)

            perdida_total += perdida.item() * imagenes.size(0)
            correctos += (salida.argmax(1) == etiquetas).sum().item()
            total += etiquetas.size(0)

    return perdida_total / total, correctos / total

# entrenamiento principal reutilizable desde terminal o Jupyter
def entrenar_clasificador(epochs=20, batch_size=32, learning_rate=0.001, workers=4):
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Primero ejecuta preparar_utkface.py")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = obtener_dispositivo()

    # aumento de datos únicamente para entrenamiento
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # validación sin modificaciones aleatorias
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    # pin_memory acelera transferencia CPU → GPU cuando CUDA está disponible
    usar_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=usar_cuda
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=usar_cuda
    )

    print("Clases:", train_dataset.classes)
    print(f"Train: {len(train_dataset)} imágenes")
    print(f"Val: {len(val_dataset)} imágenes")
    print(f"Batch: {batch_size} | Épocas: {epochs}")

    modelo = ClasificadorEdad(len(train_dataset.classes)).to(device)
    pesos = calcular_pesos(train_dataset, device)
    criterio = nn.CrossEntropyLoss(weight=pesos)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=learning_rate)

    mejor_accuracy = 0.0
    mejores_pesos = copy.deepcopy(modelo.state_dict())
    historial = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = entrenar_epoca(
            modelo, train_loader, criterio, optimizador, device
        )
        val_loss, val_acc = evaluar(
            modelo, val_loader, criterio, device
        )

        historial.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(
            f"Época {epoch:02d}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

        # conserva los pesos con mejor accuracy de validación
        if val_acc > mejor_accuracy:
            mejor_accuracy = val_acc
            mejores_pesos = copy.deepcopy(modelo.state_dict())

    # guarda último modelo
    torch.save({
        "model_state_dict": modelo.state_dict(),
        "classes": train_dataset.classes,
        "image_size": 224
    }, MODELS_DIR / "last_age.pth")

    # guarda el mejor modelo
    modelo.load_state_dict(mejores_pesos)
    torch.save({
        "model_state_dict": modelo.state_dict(),
        "classes": train_dataset.classes,
        "image_size": 224
    }, MODELS_DIR / "best_age.pth")

    print(f"\nMejor accuracy validación: {mejor_accuracy:.4f}")
    print(f"Mejor modelo: {MODELS_DIR / 'best_age.pth'}")

    return modelo, historial, train_dataset.classes

# permite ejecutar directamente desde terminal
if __name__ == "__main__":
    entrenar_clasificador()