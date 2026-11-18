"""
Prepara el dataset TENAX-LPR (Roboflow, formato YOLO26) y le aplica data augmentation offline.

Pasos:
  1. Extrae uno o varios .zip de Roboflow en datasets/<nombre>_original/
  2. Convierte las etiquetas poligonales (cls x1 y1 x2 y2 ...) a cajas YOLO (cls cx cy w h)
  3. Copia valid/ y test/ SIN aumentar (para que las métricas sean honestas)
  4. Para train/ guarda cada imagen original + N variantes aumentadas que simulan
     condiciones reales de una cámara de celular apuntando a la calle:
       - desenfoque por movimiento (autos en movimiento)
       - cambios de iluminación: noche, contraluz, sombras, sol
       - lluvia / niebla
       - ruido de sensor y compresión JPEG
       - perspectiva, rotación ligera, escala (placa lejos / cerca)
     Sin volteo horizontal: los caracteres de la placa quedarían en espejo.
  5. Agrega imágenes NEGATIVAS (sin placa): recortes de las mismas fotos que no tocan
     ninguna placa (faros, parrillas, letreros, calle) y, opcionalmente, fotos propias
     sin placas (--fondos). Enseñan al modelo qué NO es una placa => menos falsas detecciones.

Los puntos del polígono se transforman como keypoints, así la caja final se ajusta
exactamente a la placa incluso después de perspectiva o rotación.

Uso:
    python entrenamiento/preparar_dataset.py --zip "C:/Users/.../TENAX-LPR.v2i.yolo26.zip"
    # Combinando con tus propias fotos etiquetadas en Roboflow y fotos sin placas:
    python entrenamiento/preparar_dataset.py --zip TENAX.zip MisFotos.zip --fondos fotos_sin_placas/
"""

from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- etiquetas
def leer_etiquetas(txt: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    """Devuelve [(cls, [(x, y), ...]), ...] en coordenadas normalizadas.
    Una caja (cls cx cy w h) se convierte en sus 4 esquinas para tratarla igual que un polígono."""
    objetos = []
    if not txt.exists():
        return objetos
    for linea in txt.read_text(encoding="utf-8").splitlines():
        p = linea.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        v = [float(x) for x in p[1:]]
        if len(v) == 4:
            cx, cy, w, h = v
            pts = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                   (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
        else:
            pts = list(zip(v[0::2], v[1::2]))
        objetos.append((cls, pts))
    return objetos


def puntos_a_caja(pts: np.ndarray, W: int, H: int, min_px: float = 4.0):
    """pts en píxeles -> (cx, cy, w, h) normalizado o None si la placa quedó fuera/minúscula."""
    xs = np.clip(pts[:, 0], 0, W - 1)
    ys = np.clip(pts[:, 1], 0, H - 1)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if (x1 - x0) < min_px or (y1 - y0) < min_px:
        return None
    # Si más del 60% del polígono original quedó fuera de la imagen, descartar
    dentro = ((pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)).mean()
    if dentro < 0.4:
        return None
    return ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H)


def escribir_etiquetas(txt: Path, cajas: list[tuple[int, tuple[float, float, float, float]]]):
    txt.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, (cx, cy, w, h) in cajas),
                   encoding="utf-8")


# --------------------------------------------------------------------------- augmentation
def pipeline_aumento(img_size: int) -> A.Compose:
    return A.Compose(
        [
            # Geometría: placa más cerca/lejos, cámara inclinada, ángulo de la calle
            A.Affine(scale=(0.6, 1.25), translate_percent=(-0.1, 0.1), rotate=(-8, 8),
                     shear=(-6, 6), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7),
            A.Perspective(scale=(0.02, 0.07), p=0.4),

            # Movimiento / enfoque
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 11)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.Defocus(radius=(1, 3)),
            ], p=0.45),

            # Iluminación: día, noche, contraluz
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=(-0.45, 0.3), contrast_limit=0.3),
                A.RandomGamma(gamma_limit=(60, 160)),
                A.CLAHE(clip_limit=3.0),
            ], p=0.8),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=20, p=0.4),
            A.RandomShadow(shadow_roi=(0, 0.3, 1, 1), num_shadows_limit=(1, 2), p=0.2),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=120, p=0.05),

            # Clima
            A.OneOf([
                A.RandomRain(slant_range=(-10, 10), drop_length=15, blur_value=3, brightness_coefficient=0.8),
                A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08),
            ], p=0.15),

            # Calidad de cámara de celular / video comprimido
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.08)),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            ], p=0.35),
            A.Downscale(scale_range=(0.4, 0.8), p=0.2),
            A.ImageCompression(quality_range=(35, 90), p=0.4),
            A.ToGray(p=0.05),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


# --------------------------------------------------------------------------- pasos
def extraer(zip_path: Path, destino: Path) -> Path:
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            # protección contra rutas maliciosas dentro del zip
            if m.startswith("/") or ".." in Path(m).parts:
                raise ValueError(f"Ruta sospechosa en el zip: {m}")
        z.extractall(destino)
    return destino


def listar_imagenes(src: Path) -> list[Path]:
    d = src / "images"
    if not d.exists():
        return []
    return [f for f in sorted(d.iterdir()) if f.suffix.lower() in IMG_EXTS]


def copiar_split(src: Path, dst: Path, prefijo: str = "") -> int:
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for img in listar_imagenes(src):
        im = cv2.imread(str(img))
        if im is None:
            continue
        H, W = im.shape[:2]
        cajas = []
        for cls, pts in leer_etiquetas(src / "labels" / f"{img.stem}.txt"):
            arr = np.array(pts, dtype=np.float32) * [W, H]
            b = puntos_a_caja(arr, W, H)
            if b:
                cajas.append((cls, b))
        shutil.copy2(img, dst / "images" / f"{prefijo}{img.name}")
        escribir_etiquetas(dst / "labels" / f"{prefijo}{img.stem}.txt", cajas)
        n += 1
    return n


class VerificadorPlacas:
    """Usa un detector ya entrenado (best.onnx) para revisar que un recorte "negativo" de verdad
    no tenga placa. Hace falta porque algunas fotos de Roboflow traen placas SIN etiquetar
    (p. ej. un segundo auto al fondo); un recorte así enseñaría al modelo que una placa no es placa."""

    def __init__(self, onnx_path: Path, umbral: float = 0.35):
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
        self.umbral = umbral

    def tiene_placa(self, im: np.ndarray) -> bool:
        S = 640
        h, w = im.shape[:2]
        s = S / max(h, w)
        nh, nw = round(h * s), round(w * s)
        lienzo = np.full((S, S, 3), 114, np.uint8)
        py, px = (S - nh) // 2, (S - nw) // 2
        lienzo[py:py + nh, px:px + nw] = cv2.resize(im, (nw, nh))
        self.net.setInput(cv2.dnn.blobFromImage(lienzo, 1 / 255, (S, S), swapRB=True))
        out = self.net.forward()
        if out.ndim == 3 and out.shape[2] == 6:   # salida end2end [1, N, 6]
            conf = out[0, :, 4]
        else:                                     # salida clásica [1, 4+nc, N]
            conf = out[0, 4:].max(axis=0)
        return float(conf.max()) >= self.umbral


def recorte_sin_placa(H: int, W: int, cajas_px: list[tuple[float, float, float, float]],
                      intentos: int = 30):
    """Busca un recorte (x0, y0, x1, y1) que no toque ninguna placa (con 15% de margen)."""
    for _ in range(intentos):
        f = random.uniform(0.3, 0.6)
        cw, ch = int(W * f), int(H * f)
        if cw < 64 or ch < 64:
            return None
        x0, y0 = random.randint(0, W - cw), random.randint(0, H - ch)
        x1, y1 = x0 + cw, y0 + ch
        libre = True
        for bx0, by0, bx1, by1 in cajas_px:
            mx, my = (bx1 - bx0) * 0.15, (by1 - by0) * 0.15
            if x0 < bx1 + mx and x1 > bx0 - mx and y0 < by1 + my and y1 > by0 - my:
                libre = False
                break
        if libre:
            return x0, y0, x1, y1
    return None


def aumentar_train(src: Path, dst: Path, copias: int, negativos: int, img_size: int,
                   prefijo: str = "", verificador: VerificadorPlacas | None = None,
                   descartes: list | None = None) -> tuple[int, int, int]:
    """Copia train, genera `copias` variantes aumentadas por imagen y `negativos` recortes
    SIN placa por imagen (defensas, faros, letreros, calle)."""
    aug = pipeline_aumento(img_size)
    n_orig = copiar_split(src, dst, prefijo)
    n_aug = n_neg = 0
    for img in listar_imagenes(src):
        im = cv2.imread(str(img))
        if im is None:
            continue
        H, W = im.shape[:2]
        objetos = leer_etiquetas(src / "labels" / f"{img.stem}.txt")
        kps, dueños, cajas_px = [], [], []
        for i, (_, pts) in enumerate(objetos):
            arr = np.array(pts, dtype=np.float32) * [W, H]
            cajas_px.append((arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max()))
            for x, y in arr:
                kps.append((float(x), float(y)))
                dueños.append(i)

        generadas, intentos = 0, 0
        while generadas < copias and intentos < copias * 4:
            intentos += 1
            out = aug(image=im, keypoints=kps)
            oim = out["image"]
            oh, ow = oim.shape[:2]
            okp = np.array(out["keypoints"], dtype=np.float32).reshape(-1, 2)
            cajas = []
            for i, (cls, _) in enumerate(objetos):
                sel = okp[[j for j, d in enumerate(dueños) if d == i]]
                b = puntos_a_caja(sel, ow, oh)
                if b:
                    cajas.append((cls, b))
            # Si había placas y todas se perdieron, reintentar (evita falsos negativos)
            if objetos and not cajas:
                continue
            nombre = f"{prefijo}{img.stem}_aug{generadas}"
            cv2.imwrite(str(dst / "images" / f"{nombre}.jpg"), oim, [cv2.IMWRITE_JPEG_QUALITY, 92])
            escribir_etiquetas(dst / "labels" / f"{nombre}.txt", cajas)
            generadas += 1
            n_aug += 1

        for k in range(negativos):
            r = recorte_sin_placa(H, W, cajas_px)
            if r is None:
                break
            x0, y0, x1, y1 = r
            # Se revisa el recorte sin aumentar: si el detector ve una placa, había una sin etiquetar
            if verificador and verificador.tiene_placa(im[y0:y1, x0:x1]):
                if descartes is not None:
                    descartes.append(f"{prefijo}{img.stem}_neg{k}")
                continue
            crop = aug(image=im[y0:y1, x0:x1], keypoints=[])["image"]
            nombre = f"{prefijo}{img.stem}_neg{k}"
            cv2.imwrite(str(dst / "images" / f"{nombre}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            escribir_etiquetas(dst / "labels" / f"{nombre}.txt", [])  # sin placas
            n_neg += 1
    return n_orig, n_aug, n_neg


def agregar_fondos(carpeta: Path, dst: Path, verificador: VerificadorPlacas | None = None,
                   descartes: list | None = None) -> int:
    """Copia fotos propias SIN placas (calle vacía, letreros, rejas...) como negativos de train."""
    n = 0
    for f in sorted(carpeta.iterdir()):
        im = cv2.imread(str(f)) if f.suffix.lower() in IMG_EXTS else None
        if im is None:
            continue
        if verificador and verificador.tiene_placa(im):
            if descartes is not None:
                descartes.append(f"fondo_{f.name}")
            continue
        shutil.copy2(f, dst / "images" / f"fondo_{f.name}")
        escribir_etiquetas(dst / "labels" / f"fondo_{f.stem}.txt", [])
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, type=Path, nargs="+",
                    help="Uno o varios zips de Roboflow (formato YOLO26). Ej.: TENAX + tus propias fotos")
    ap.add_argument("--salida", type=Path, default=ROOT / "datasets" / "TENAX-LPR-aug")
    ap.add_argument("--copias", type=int, default=5, help="Variantes aumentadas por imagen de train")
    ap.add_argument("--negativos", type=int, default=1,
                    help="Recortes sin placa por imagen de train (0 para desactivar)")
    ap.add_argument("--fondos", type=Path, default=None,
                    help="Carpeta opcional con fotos propias SIN placas (se agregan a train como negativos)")
    ap.add_argument("--verificar", type=Path, default=ROOT / "best.onnx",
                    help="Detector .onnx para descartar negativos que contengan placas sin etiquetar "
                         "(por defecto best.onnx del proyecto; usa --verificar '' para no revisar)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.salida.exists():
        shutil.rmtree(args.salida)
    args.salida.mkdir(parents=True)
    originales = args.salida.parent / f"{args.salida.name}_original"
    if originales.exists():
        shutil.rmtree(originales)

    verificador = None
    if args.verificar and str(args.verificar) not in ("", ".") and args.verificar.is_file():
        verificador = VerificadorPlacas(args.verificar)
        print(f"[INFO] Revisando negativos con {args.verificar}")
    descartes: list[str] = []

    resumen = {"valid": 0, "test": 0}
    n_orig = n_aug = n_neg = 0
    meta = None
    for i, z in enumerate(args.zip):
        crudo = extraer(z, originales / f"{i}_{z.stem}")
        prefijo = "" if len(args.zip) == 1 else f"d{i}_"
        if meta is None:
            meta = yaml.safe_load((crudo / "data.yaml").read_text(encoding="utf-8"))
        for split in ("valid", "test"):
            if (crudo / split).exists():
                resumen[split] += copiar_split(crudo / split, args.salida / split, prefijo)
        o, a, n = aumentar_train(crudo / "train", args.salida / "train", args.copias, args.negativos,
                                 args.imgsz, prefijo, verificador, descartes)
        n_orig, n_aug, n_neg = n_orig + o, n_aug + a, n_neg + n

    n_fondos = agregar_fondos(args.fondos, args.salida / "train", verificador, descartes) if args.fondos else 0
    resumen["train"] = n_orig + n_aug + n_neg + n_fondos

    data = {
        "path": str(args.salida.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": meta.get("nc", 1),
        "names": meta.get("names", ["placa"]),
    }
    (args.salida / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                                           encoding="utf-8")

    print(f"[OK] Dataset listo en {args.salida}")
    print(f"     train: {n_orig} originales + {n_aug} aumentadas + {n_neg} negativas (sin placa)"
          + (f" + {n_fondos} fondos propios" if n_fondos else "") + f" = {resumen['train']}")
    for s in ("valid", "test"):
        print(f"     {s}: {resumen[s]} (sin aumentar)")
    if descartes:
        print(f"     negativos descartados por contener una placa sin etiquetar: {len(descartes)}")
        for d in descartes:
            print(f"       - {d}")


if __name__ == "__main__":
    main()
