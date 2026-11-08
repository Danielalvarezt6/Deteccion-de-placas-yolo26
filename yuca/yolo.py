import os
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO

os.environ["YOLO_OFFLINE"] = "true"

BASE_DIR = Path("/lustre/home/estudiante_02/YOLO")
if not BASE_DIR.exists():
    BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "TENAX-LPR.v4i.yolo26"
MODEL_WEIGHTS = str(BASE_DIR / "yolo26n.pt") if (BASE_DIR / "yolo26n.pt").exists() else "yolo26n.pt"
DATA_YAML = str(DATASET_DIR / "data.yaml")
PROJECT_DIR = str(BASE_DIR / "runs" / "tenax_lpr")
RUN_NAME = "yolo26_placas"

IMG_SIZE = 640
BATCH_SIZE = 64
EPOCHS = 120
WORKERS = 8


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


if __name__ == "__main__":
    print(f"[INFO] PyTorch: {torch.__version__} | HIP (ROCm): {getattr(torch.version, 'hip', None)}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch no detecta las GPUs MI210. "
            "Ejecuta: pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2"
        )

    device = 0
    print(f"[INFO] GPU activa: {torch.cuda.get_device_name(0)} (Total visibles: {torch.cuda.device_count()})")

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
        patience=25,
        cos_lr=True,
        amp=False,
        fliplr=0.0,
        flipud=0.0,
        exist_ok=True,
    )

    best_weights = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"

    if best_weights.exists():
        shutil.copy(best_weights, BASE_DIR / "license_plate_detector.pt")

        best_model = YOLO(str(best_weights))
        best_model.val(data=DATA_YAML, imgsz=IMG_SIZE, conf=0.5, device=device)
        onnx_file = best_model.export(format="onnx", imgsz=IMG_SIZE, simplify=True, dynamic=False, device="cpu")
        if onnx_file and Path(onnx_file).exists():
            shutil.copy(onnx_file, BASE_DIR / "best.onnx")
