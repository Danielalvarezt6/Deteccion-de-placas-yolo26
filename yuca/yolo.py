import os
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO

os.environ["YOLO_OFFLINE"] = "true"

# Carpeta base: la que contiene datasets/. En el clúster es la misma carpeta de yolo.py
# (p. ej. /lustre/home/<usuario>/YOLO); en el repo es la raíz (yuca/..). Se puede forzar con BASE_DIR.
_SCRIPT_DIR = Path(__file__).resolve().parent
if os.environ.get("BASE_DIR"):
    BASE_DIR = Path(os.environ["BASE_DIR"])
elif (_SCRIPT_DIR / "datasets").exists():
    BASE_DIR = _SCRIPT_DIR
else:
    BASE_DIR = _SCRIPT_DIR.parent

# Dataset generado por entrenamiento/preparar_dataset.py (TENAX-LPR v2 + data augmentation)
DATASET_DIR = Path(os.environ.get("DATASET_DIR", BASE_DIR / "datasets" / "TENAX-LPR-aug"))
# Modelo base: yolo26n (rápido en el celular) o MODEL=yolo26s.pt (más preciso con placas pequeñas, ~3x más lento)
MODEL_NAME = os.environ.get("MODEL", "yolo26n.pt")
MODEL_WEIGHTS = str(BASE_DIR / MODEL_NAME) if (BASE_DIR / MODEL_NAME).exists() else MODEL_NAME
DATA_YAML = str(DATASET_DIR / "data.yaml")
PROJECT_DIR = str(BASE_DIR / "runs" / "tenax_lpr")
# Cada modelo/experimento en su propia carpeta para no sobrescribir resultados anteriores
RUN_NAME = os.environ.get("RUN_NAME", f"{Path(MODEL_NAME).stem}_placas")

IMG_SIZE = 640
# En el clúster (2x MI210) se usa batch=64; en una laptop (p.ej. RTX 3050 6GB) usar BATCH=16 y AMP=1
BATCH_SIZE = int(os.environ.get("BATCH", 64))
EPOCHS = int(os.environ.get("EPOCHS", 120))
WORKERS = int(os.environ.get("WORKERS", 8))
AMP = os.environ.get("AMP", "0") == "1"


def normalize_polygon_labels_to_boxes(dataset_root: Path) -> None:
    """Convierte etiquetas de polígonos de Roboflow a formato caja estándar (cls cx cy w h)."""
    for split in ("train", "valid", "test"):
        labels_dir = dataset_root / split / "labels"
        if not labels_dir.exists():
            continue

        modified = False
        for txt_file in labels_dir.glob("*.txt"):
            lines = txt_file.read_text(encoding="utf-8").strip().splitlines()
            new_lines = []
            file_changed = False

            for line in lines:
                parts = line.strip().split()
                if len(parts) > 5:
                    cls_id = parts[0]
                    coords = [float(x) for x in parts[1:]]
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    cx = (x_min + x_max) / 2.0
                    cy = (y_min + y_max) / 2.0
                    w = x_max - x_min
                    h = y_max - y_min
                    new_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    file_changed = True
                elif len(parts) == 5:
                    new_lines.append(line.strip())

            if file_changed:
                txt_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                modified = True

        cache_file = dataset_root / split / "labels.cache"
        if modified and cache_file.exists():
            cache_file.unlink()


def fix_data_yaml_path(dataset_root: Path) -> None:
    """El data.yaml se genera en Windows con una ruta absoluta; la reescribe a la ruta real
    del dataset en esta máquina (p. ej. /lustre/home/.../datasets/TENAX-LPR-aug)."""
    yaml_file = dataset_root / "data.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError(
            f"No existe {yaml_file}. Genera el dataset con entrenamiento/preparar_dataset.py "
            "y súbelo a esta ruta (o define DATASET_DIR)."
        )
    lines = yaml_file.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if not l.startswith("path:")]
    lines.insert(0, f"path: {dataset_root.resolve().as_posix()}")
    yaml_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(f"[INFO] PyTorch: {torch.__version__} | HIP (ROCm): {getattr(torch.version, 'hip', None)}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch no detecta ninguna GPU (MI210 con ROCm o NVIDIA con CUDA). "
            "Ejecuta: pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2"
        )

    device = 0
    print(f"[INFO] Dataset: {DATASET_DIR}")
    print(f"[INFO] GPU activa: {torch.cuda.get_device_name(0)} (Total visibles: {torch.cuda.device_count()})")

    fix_data_yaml_path(DATASET_DIR)
    normalize_polygon_labels_to_boxes(DATASET_DIR)

    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=RUN_NAME,
        # La validación tiene pocas imágenes y las métricas oscilan: más paciencia antes de parar
        patience=40,
        cos_lr=True,
        amp=AMP,
        # El dataset ya trae aumento offline (blur, noche, lluvia, ruido, perspectiva);
        # se mantiene el aumento en línea de Ultralytics pero SIN volteos (la placa quedaría en espejo).
        mosaic=1.0,
        close_mosaic=15,
        degrees=0.0,
        fliplr=0.0,
        flipud=0.0,
        exist_ok=True,
    )

    best_weights = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"

    if best_weights.exists():
        shutil.copy(best_weights, BASE_DIR / "license_plate_detector.pt")

        best_model = YOLO(str(best_weights))
        # Evaluación final: valid (usado para elegir best.pt) y test (imágenes que el modelo nunca vio).
        # El conjunto test es la cifra honesta para reportar.
        resumen = []
        for split in ("val", "test"):
            m = best_model.val(data=DATA_YAML, split=split, imgsz=IMG_SIZE, device=device,
                               project=PROJECT_DIR, name=f"{RUN_NAME}_{split}", exist_ok=True, plots=True)
            resumen.append((split, m.box.mp, m.box.mr, m.box.map50, m.box.map))
        print("\n[RESULTADOS]  split   precision  recall  mAP50  mAP50-95")
        for split, p, r, m50, m5095 in resumen:
            print(f"[RESULTADOS]  {split:5s}  {p:9.3f}  {r:6.3f}  {m50:5.3f}  {m5095:8.3f}")
        onnx_file = best_model.export(format="onnx", imgsz=IMG_SIZE, simplify=True, dynamic=False, device="cpu")
        if onnx_file and Path(onnx_file).exists():
            shutil.copy(onnx_file, BASE_DIR / "best.onnx")
