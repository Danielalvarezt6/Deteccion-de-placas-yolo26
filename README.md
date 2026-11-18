# Radar Vial · Detección de Placas e Infracciones (YOLO26 + ONNX)

*Integrantes del equipo: Daniel Alvarez Terrazas y Melina González Méndez*


Sistema de detección de matrículas vehiculares, lectura OCR de placas mexicanas, estimación de velocidad y generación de boletas de infracción en PDF. Incluye el flujo de entrenamiento de **YOLO26** en clúster HPC con GPUs AMD ROCm y la aplicación web/móvil 100% en navegador lista para desplegar en **Netlify**.

---

## Dataset

El detector de matrículas se entrena con el dataset público **TENAX-LPR (v2)** de Roboflow Universe (licencia CC BY 4.0), exportado en formato YOLO26:
- **Enlace del dataset:** https://universe.roboflow.com/fernandos-workspace-ovm5c/tenax-lpr/dataset/2
- 223 imágenes, 1 clase (`placa`): 169 train · 32 valid · 22 test. Las etiquetas vienen como polígonos.

### Data augmentation (`entrenamiento/preparar_dataset.py`)
Como el dataset es pequeño, se generan **5 variantes por imagen de entrenamiento** más **1 recorte negativo (sin placa)** por imagen (169 → 1,176 imágenes) que simulan lo que ve la cámara de un celular apuntando a la calle:

| Condición real | Transformación |
|---|---|
| Auto en movimiento / cámara movida | `MotionBlur`, `GaussianBlur`, `Defocus` |
| Placa lejos o cerca, ángulo de la calle | `Affine` (escala 0.6–1.25, rotación ±8°, shear), `Perspective` |
| Noche, contraluz, sombras, sol | `RandomBrightnessContrast`, `RandomGamma`, `CLAHE`, `RandomShadow`, `RandomSunFlare` |
| Lluvia / niebla | `RandomRain`, `RandomFog` |
| Sensor de celular / video comprimido | `GaussNoise`, `ISONoise`, `Downscale`, `ImageCompression` |

- **Sin volteo horizontal**: los caracteres de la placa quedarían en espejo.
- Los vértices del polígono se transforman como *keypoints*, así la caja final se ajusta a la placa aun después de perspectiva o rotación.
- **Negativos**: recortes de las mismas fotos que no tocan ninguna placa (faros, emblemas, parrillas, pavimento, letreros). Enseñan al modelo qué *no* es una placa y reducen detecciones falsas. Se pueden agregar fotos propias sin placas con `--fondos carpeta/`. Cada negativo se revisa con el detector ya entrenado (`--verificar best.onnx`, activo por defecto) y se descarta si contiene una placa: algunas fotos de Roboflow traen placas sin etiquetar (p. ej. un segundo auto al fondo).
- Se pueden combinar varios zips de Roboflow (p. ej. TENAX + fotogramas propios grabados con el celular): `--zip TENAX.zip MisFotos.zip`.
- Solo se aumenta `train`; `valid` y `test` se dejan originales para que las métricas sean honestas.

```bash
pip install albumentations opencv-python-headless pyyaml
python entrenamiento/preparar_dataset.py --zip "ruta/a/TENAX-LPR.v2i.yolo26.zip" --copias 5
```
Genera `datasets/TENAX-LPR-aug/` con su `data.yaml` (la carpeta `datasets/` no se sube a git).

### Evaluación
Al terminar el entrenamiento, `yuca/yolo.py` evalúa el mejor modelo en **valid** y en **test** (imágenes que el modelo nunca vio) e imprime una tabla `[RESULTADOS]` con precision, recall, mAP50 y mAP50-95. Las gráficas quedan en `runs/tenax_lpr/<modelo>_placas/`, `..._val/` y `..._test/`.

Variables de entorno opcionales: `MODEL=yolo26s.pt` (modelo más grande), `EPOCHS`, `BATCH`, `RUN_NAME`, `DATASET_DIR`.

### Velocidad de la app
- **Detector y OCR en la GPU (WebGPU)** dentro de *Web Workers*: si el navegador no tiene WebGPU, usan la CPU (WebAssembly) automáticamente. Medido en PC: detector **~45 ms** por fotograma con GPU contra ~340–415 ms con CPU; OCR ~75 ms por lectura sin bloquear la pantalla.
- **OCR corregido**: `cct_s_v1_global.onnx` espera la placa a color (`uint8 [1, 64, 128, 3]`); antes se le enviaba en gris, fallaba y la app caía a Tesseract (más lento y menos preciso).
- **Cajas fluidas**: entre detecciones, cada caja se mueve con la velocidad estimada de la placa en vez de ir a saltos.
- WebGPU funciona en Chrome/Edge de escritorio y en Chrome para Android reciente; en Safari/iPhone depende de la versión de iOS.

### Prueba del modelo nuevo en la app (test, 22 imágenes que el modelo nunca vio)
- Modelo con umbral 0.6: **21/22 placas encontradas, 0 falsas**.
- Pipeline completo de la app (con filtros de forma/tamaño/contraste relajados): **17/22, 0 falsas**. Las que pierde son sobre todo fotos giradas 90° o estiradas por Roboflow, que no ocurren con la cámara.
- 162 imágenes negativas (sin placa): **0 falsas detecciones**.

### Ajustes de la app para reducir errores
- `confThreshold: 0.6` — elegido con la curva F1 del modelo (F1 0.98 en conf 0.70).
- `minHitsTicket: 3` — la placa debe detectarse al menos 3 veces antes de multar.
- `speedToleranceKmh: 5` — solo se multa por encima de límite + 5 km/h (compensa el temblor de las cajas).

---

## Estructura del Proyecto

### 1. Entrenamiento y Exportación a ONNX (`yuca/`)
Contiene los archivos utilizados para entrenar el modelo **YOLO26** en el clúster **La Yuca** (GPUs AMD Instinct `MI210` con ROCm) y exportarlo a ONNX:
- (yolo.py): Script de entrenamiento y exportación. Convierte automáticamente las anotaciones poligonales de Roboflow a cajas delimitadoras estándar (`cls cx cy w h`), entrena a partir de `yolo26n.pt` durante 120 épocas (`imgsz=640`, `batch=64`), valida el mejor checkpoint (`best.pt`) y lo exporta a formato ONNX (`best.onnx`).
- [`yolo_rocm.yml`]: Archivo de definición del entorno Conda (`yolo-rocm`) con Python 3.10, PyTorch para ROCm 7.2, `ultralytics`, `onnx`, `onnxslim` y `onnxruntime`.
- [job_train.slurm]: Script de trabajo **SLURM** configurado para la partición GPU (`--gres=gpu:MI210:2`, módulo `rocm/7.2.2`), que activa el entorno `yolo-rocm` y lanza `python yolo.py`.

#### Entrenamiento local (GPU NVIDIA de laptop, p. ej. RTX 3050 6GB):
```bash
pip install ultralytics onnx onnxslim onnxruntime
BATCH=16 AMP=1 python yuca/yolo.py
```
Al terminar, el script reemplaza `best.onnx` en la raíz del proyecto y la app web lo usa sin más cambios.

#### Comandos de entrenamiento en el clúster:
```bash
# 0. Desde tu PC: subir el dataset aumentado y el script actualizado
scp datasets/TENAX-LPR-aug.zip yuca/yolo.py <usuario>@<host-yuca>:/lustre/home/<usuario>/YOLO/

# 0b. En el clúster: descomprimir en YOLO/datasets/
cd /lustre/home/<usuario>/YOLO
mkdir -p datasets && unzip -o TENAX-LPR-aug.zip -d datasets/
# 1. Crear el entorno Conda 
conda env create -f yolo_rocm.yml

# 2. Enviar el trabajo de entrenamiento a SLURM
sbatch job_train.slurm
```

---

### 2. Aplicación Web y Móvil (Despliegue en Netlify)
La aplicación web corre completamente en el navegador del usuario (PC o teléfono móvil) mediante **ONNX Runtime Web (WebAssembly)** sin requerir servidor backend:
- [`index.html`](index.html): Interfaz interactiva con soporte para subir video o usar la cámara en vivo, botón de **Detectar placas** (YOLO26 en Web Worker + OCR en vivo con formato de placas mexicanas) y botón de **Modo infracciones** con límite de velocidad ajustable, alertas en tiempo real y descarga de boletas de multa en PDF (`jsPDF`).
- [`best.onnx`](best.onnx): Modelo YOLO26 detector de matrículas exportado por `yuca/yolo.py`.
- [`cct_s_v1_global.onnx`](cct_s_v1_global.onnx): Modelo ONNX de lectura OCR de placas (`fast-plate-ocr`).
---

## Despliegue en Netlify
 
 https://detectorplacasdani.netlify.app/
