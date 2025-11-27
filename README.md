# Trabajo de Grado – Análisis de Presiones Plantares

Este repositorio contiene el código desarrollado para el Trabajo de Grado de Ingeniería Biomédica, orientado al **análisis del Centro de Presión (CoP)** y parámetros baropodométricos a partir de plataformas de presión plantares.

El objetivo principal es ofrecer un **pipeline unificado** para trabajar con distintas bases de datos de presiones plantares y obtener métricas comparables de forma reproducible.

---

## Contenido del repositorio

- `main.py`  
  Script principal con un **menú interactivo** que organiza el flujo de trabajo:
  - selección de dataset,
  - descarga / carga de archivos,
  - preprocesamiento y estandarización,
  - cálculo de parámetros,
  - generación de resultados.

- `Compute_cop_static.py`  
  Cálculo del CoP en huellas estáticas.

- `Download_mat.py`, `Download_nii.py`  
  Utilidades para descargar y organizar archivos de los distintos datasets.

- `Parameters_StepUp.py`, `Parameters_CAD.py`  
  Cálculo de parámetros regionales y temporales específicos para cada base de datos.

- `Pixel_to_cm.py`, `Rotate.py`, `Standarization.py`, `GIF_Generator.py`  
  Herramientas de apoyo para conversión de unidades, corrección geométrica, estandarización de huellas y generación de visualizaciones.

---

## Datasets

El código fue desarrollado para trabajar con las siguientes bases de datos de presiones plantares:

- **CASIA-D** (Chinese Academy of Sciences)
- **CADDataset** (Footscan®)
- **StepUpDataset** (StepUp-P150)

> **Importante:** por cuestiones de tamaño y licenciamiento, **los datos crudos NO están incluidos en este repositorio**.  
- 📂 [Directorio de datos en Google Drive] (https://drive.google.com/drive/folders/11P81Wghr5bg1aoXZy2MfJH4IRUWzATlu?usp=drive_link)
  
> El usuario debe descargar cada base de datos desde sus fuentes originales (o desde el enlace provisto por la autora) y ubicarlas en las carpetas esperadas por el código:
TFG/
  CADDataset/
  CASIA-DDataset/
  StepUpDataset/

---

## Requisitos
- Python 3.9+
- Bibliotecas principales:
  - numpy
  - opencv-python
  - Pillow
  - matplotlib (opcional, para visualizaciones)
  - otras dependencias estándar indicadas en el código

Se recomienda trabajar en un entorno virtual (venv) e instalar los paquetes con pip.

---

## Uso básico
1. Clonar el repositorio
2. Crear entorno e instalar dependencias
3. Colocar las carpetas de los datasets en las rutas indicadas
4. Ejecutar el script principal

---

## Licencia y uso

El código se distribuye bajo una licencia de uso académico y no comercial.
Se permite su reutilización y modificación con fines de investigación y docencia, siempre que se cite el trabajo de grado correspondiente y se mantenga la autoría original en los archivos.

Para consultas académicas o colaboración, contactar a la autora a través de GitHub.
