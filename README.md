# Radar Vial · Detección de Placas, Velocidad e Infracciones (3 Modelos ONNX)

*Integrantes del equipo: Daniel Alvarez, Karol Burrola y Melina Gonzalez*

Sistema de detección de vehículos, estimación de velocidad vial, lectura OCR de placas mexicanas y generación automática de boletas de infracción en PDF. Incluye el flujo de preparación de datos y entrenamiento de **YOLO26** con GPUs AMD ROCm o NVIDIA, la aplicación web/móvil 100% en navegador desplegada en **Netlify** y la presentación del proyecto.

**Demo en vivo:** https://reconocimientodeplacas.netlify.app

---

## Arquitectura en Cascada (3 Modelos ONNX)

La aplicación web ([index.html](index.html)) ejecuta 3 modelos en *Web Workers* mediante **ONNX Runtime Web** (WebGPU con respaldo automático a WebAssembly):

| Etapa | Modelo | Función |
|---|---|---|
| **1 · Vehículos y Velocidad** | `yolo26n.onnx` | Preentrenado en COCO. Corre en el fotograma completo detectando solo vehículos (`car`, `motorcycle`, `bus`, `truck`), hace seguimiento (IoU) y calcula la velocidad real. |
| **2 · Detector de Placas** | `best.onnx` | Entrenado con YOLO26 sobre TENAX-LPR v2 aumentado. Corre únicamente sobre el **recorte congelado del vehículo** cuando está en zona cercana (o solo en autos con exceso si se activa el modo ahorro), logrando placas más grandes y nítidas. |
| **3 · Lectura OCR** | `cct_s_v2_global.onnx` | Modelo ONNX global de `fast-plate-ocr` v2 (`uint8 [1, 64, 128, 3]`). Promedia probabilidades entre recortes y valida máscaras de placas mexicanas. |

### Estimación de velocidad y multas
- **Zona Radar (2 líneas de control):** líneas virtuales sobre el pavimento (`56%` y `80%` del alto) con distancia real ajustable en *Ajustes del radar* (por defecto `12 m`). Al cruzar ambas líneas con la base inferior del vehículo `(cx, y2)`, calcula:
  $$v_{\text{km/h}} = \frac{D}{|t_2 - t_1|} \times 3.6$$
- **Velocidad continua:** antes de completar el cruce estima en tiempo real con regresión lineal sobre los últimos 2 s y la escala física de cada tipo de vehículo (`Auto: 1.80 m`, `Moto: 0.75 m`, `Camión: 2.45 m`, `Autobús: 2.55 m`).
- **Reglas anti-error:** límite ajustable desde **`10 km/h` hasta `160 km/h`**, margen de tolerancia **`+5 km/h`**, mínimo **5 detecciones** (`minHitsTicket: 5`) y **solo se emiten multas y alertas cuando la placa fue efectivamente leída**.
- **Boleta PDF (`jsPDF`):** incluye folio, fecha, velocidad, exceso, sanción en MXN, evidencia fotográfica completa del vehículo y recorte de la placa.

---

## Ejecutar en Local

Con los 3 modelos (`yolo26n.onnx`, `best.onnx` y `cct_s_v2_global.onnx`) en la raíz del proyecto, levanta un servidor local:

```bash
python -m http.server 8000
```

Abre `http://localhost:8000` (arriba a la derecha aparecerá `3/3 modelos · GPU` o `3/3 modelos`).

---

## Dataset y Aumento de Datos

El detector de matrículas (`best.onnx`) se entrena con el dataset público **TENAX-LPR (v2)** de Roboflow Universe (licencia CC BY 4.0):
- **Enlace:** https://universe.roboflow.com/fernandos-workspace-ovm5c/tenax-lpr/dataset/2
- 223 imágenes originales, 1 clase (`placa`): 169 train · 32 valid · 22 test.

### Data augmentation ([`entrenamiento/preparar_dataset.py`](entrenamiento/preparar_dataset.py))
Se generan **5 variantes por imagen de entrenamiento** más **1 recorte negativo (sin placa)** por imagen (169 → **1,176 imágenes**):

| Condición real | Transformaciones (Albumentations) |
|---|---|
| Auto en movimiento / cámara movida | `MotionBlur`, `GaussianBlur`, `Defocus` |
| Placa lejos, cerca o en ángulo | `Affine` (escala 0.6–1.25, rotación ±8°), `Perspective` |
| Noche, contraluz, sombras, sol | `RandomBrightnessContrast`, `RandomGamma`, `CLAHE`, `RandomShadow`, `RandomSunFlare` |
| Lluvia / niebla | `RandomRain`, `RandomFog` |
| Sensor de celular / video comprimido | `GaussNoise`, `ISONoise`, `Downscale`, `ImageCompression` |

- **Sin volteo horizontal (`fliplr=0`):** evita que los caracteres de la placa queden en espejo.
- **Negativos verificados:** recortes sin placas (faros, emblemas, parrillas, pavimento, letreros) para reducir falsos positivos.
- `valid` y `test` se conservan sin aumento para obtener métricas honestas.

```bash
pip install albumentations opencv-python-headless pyyaml
python entrenamiento/preparar_dataset.py --zip "ruta/a/TENAX-LPR.v2i.yolo26.zip" --copias 5
```

### Resultados en evaluación (conjunto `test`, nunca visto)
- **21 / 22** placas detectadas en `test` (`conf = 0.6`, F1 máx. `0.98`).
- **0 / 162** falsas detecciones en imágenes negativas sin placa.
- Inferencia en navegador con WebGPU: **~45 ms** por fotograma en detector y **~75 ms** por lectura OCR en *Web Workers*.

---

## Entrenamiento y Exportación (`yuca/`)

- [`yuca/yolo.py`](yuca/yolo.py): convierte polígonos a cajas estándar, entrena a partir de `yolo26n.pt` (120 épocas, `imgsz=640`, `batch=64`), evalúa en `val` y `test`, y exporta `best.onnx`.
- [`yuca/yolo_rocm.yml`](yuca/yolo_rocm.yml): entorno Conda (`yolo-rocm`) con Python 3.10, PyTorch ROCm 7.2, `ultralytics`, `onnx`, `onnxslim` y `onnxruntime`.

#### Entrenamiento local (GPU NVIDIA)
```bash
pip install ultralytics onnx onnxslim onnxruntime
BATCH=16 AMP=1 python yuca/yolo.py
```

#### Entrenamiento en clúster AMD ROCm
```bash
conda env create -f yuca/yolo_rocm.yml
conda activate yolo-rocm
python yuca/yolo.py
```

---

## Presentación del Proyecto

- [`presentacion.html`](presentacion.html): presentación web 16:9 (8 diapositivas) con reproductor integrado de [`VIDEO EJEMPLO.mp4`](VIDEO%20EJEMPLO.mp4) y boleta de multa generada. Controles: `←` / `→`, `F` (pantalla completa) y `E` (edición rápida).

---

## Despliegue

https://reconocimientodeplacas.netlify.app
