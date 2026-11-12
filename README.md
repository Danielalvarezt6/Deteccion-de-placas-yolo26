# Radar Vial · Detección de Placas e Infracciones (YOLO26 + ONNX)

*Integrantes del equipo: Daniel Alvarez Terrazas y Melina González Méndez*


Sistema de detección de matrículas vehiculares, lectura OCR de placas mexicanas, estimación de velocidad y generación de boletas de infracción en PDF. Incluye el flujo de entrenamiento de **YOLO26** en clúster HPC con GPUs AMD ROCm y la aplicación web/móvil 100% en navegador lista para desplegar en **Netlify**.

---

## Dataset

El modelo detector de matrículas fue entrenado con el dataset público **TENAX-LPR (v4)** de Roboflow Universe:
- **Enlace del dataset:** https://universe.roboflow.com/fernandos-workspace-ovm5c/tenax-lpr/dataset/4

---


## Estructura del Proyecto

### 1. Entrenamiento y Exportación a ONNX (`yuca/`)
Contiene los archivos utilizados para entrenar el modelo **YOLO26** en el clúster **La Yuca** (GPUs AMD Instinct `MI210` con ROCm) y exportarlo a ONNX:
- (yolo.py): Script de entrenamiento y exportación. Convierte automáticamente las anotaciones poligonales de Roboflow a cajas delimitadoras estándar (`cls cx cy w h`), entrena a partir de `yolo26n.pt` durante 120 épocas (`imgsz=640`, `batch=64`), valida el mejor checkpoint (`best.pt`) y lo exporta a formato ONNX (`best.onnx`).
- [`yolo_rocm.yml`]: Archivo de definición del entorno Conda (`yolo-rocm`) con Python 3.10, PyTorch para ROCm 6.2, `ultralytics`, `onnx`, `onnxslim` y `onnxruntime`.
- [job_train.slurm]: Script de trabajo **SLURM** configurado para la partición GPU (`--gres=gpu:MI210:2`, módulo `rocm/7.2.2`), que activa el entorno `yolo-rocm` y lanza `python yolo.py`.

#### Comandos de entrenamiento en el clúster:
```bash
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
