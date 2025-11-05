# Clasificación de Señales EEG de Imaginación Motora: Bag of Features vs. Deep Learning

Este proyecto implementa y compara dos enfoques metodológicos para la clasificación de señales de electroencefalografía (EEG) durante tareas de imaginación motora (MI): un modelo basado en **Bag of Features (BoF) combinado con SVM** y una arquitectura de **Red Neuronal Convolucional (DeepConvNet)**.

## 📋 Descripción del Proyecto

El objetivo principal es clasificar señales EEG en dos clases: **imaginación de movimiento de mano izquierda (MI-L)** y **imaginación de movimiento de mano derecha (MI-R)**. Se utilizan datos de 20 sujetos sanos, cada uno con 22 ensayos por tarea (880 ensayos totales), registrados mediante 64 electrodos con una frecuencia de muestreo de 128 Hz.

### Enfoques Implementados

1. **BoF-SVM**: Extracción de características mediante transformadas wavelet (CWT y DWT), representación mediante Bag of Features, y clasificación con Máquinas de Vectores de Soporte (SVM).
2. **DeepConvNet**: Arquitectura de red neuronal convolucional profunda que aprende representaciones directamente de las señales EEG preprocesadas.

### Resultados Principales

- **BoF-SVM**: Accuracy: 52.84%, F1-Score: 0.5451
- **DeepConvNet**: Accuracy: 67.42%, F1-Score: 0.6742

El modelo DeepConvNet demostró un mejor desempeño en todas las métricas evaluadas, superando al modelo BoF-SVM por aproximadamente 15 puntos porcentuales en accuracy.

## 🏗️ Estructura del Proyecto

```
datos_BCI/
├── Notebooks de Análisis
│   ├── 01_EDA_Analysis.ipynb           # Análisis exploratorio (PSD, correlación)
│   ├── 02_Wavelet_Analysis.ipynb       # Extracción de características wavelet
│   ├── 03_BoF_Clasificacion.ipynb      # Modelo Bag of Features + SVM
│   └── 04_DeepConvNet_CNN.ipynb        # Modelo DeepConvNet (CNN)
│
├── Datos de Entrada
│   ├── left_imag/                      # 20 archivos .set/.fdt (mano izquierda)
│   └── right_imag/                     # 20 archivos .set/.fdt (mano derecha)
│
├── Datos Procesados (generados por los notebooks)
│   ├── data/
│   │   ├── preprocessed/               # Datos preprocesados
│   │   └── bof_features/               # Características para BoF
│   └── results/
│       ├── eda/                        # Resultados análisis exploratorio
│       ├── wavelets/                   # Resultados análisis wavelet
│       ├── bof_svm/                    # Resultados BoF-SVM
│       ├── deepconvnet/                # Resultados DeepConvNet
│       └── figures/                    # Gráficas para el artículo
│
├── Documentación
│   ├── README.md                       # Este archivo
│   ├── articulo.md                     # Artículo científico en LaTeX
│   ├── pyproject.toml                  # Dependencias del proyecto
│   └── venv/                           # Entorno virtual Python
│
└── Archivos de Configuración
    └── .gitignore
```

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.11 o superior
- pip (gestor de paquetes de Python)
- Jupyter Notebook o Jupyter Lab

### Instalación

#### Opción 1: Usar el entorno virtual existente (Recomendado)

```bash
# Clonar o descargar el proyecto
cd datos_BCI

# Activar el entorno virtual
source venv/bin/activate  # En macOS/Linux
# O: venv\Scripts\activate  # En Windows

# Las dependencias ya están instaladas
```

#### Opción 2: Crear un entorno virtual nuevo

```bash
# Crear entorno virtual con Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # macOS/Linux
# O: venv\Scripts\activate  # Windows

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias desde pyproject.toml
pip install -e .

# Las dependencias se instalarán automáticamente según pyproject.toml:
# - mne>=1.10.0, scipy>=1.16.0, numpy>=1.26.0
# - matplotlib>=3.10.0, seaborn>=0.13.0, pandas>=2.3.0
# - PyWavelets>=1.9.0, scikit-learn>=1.7.0
# - torch>=2.2.0, jupyter>=1.1.0, tqdm>=4.67.0
```

#### Opción 3: Instalación desde notebooks

Si ejecutas los notebooks en un entorno nuevo, ejecuta la **primera celda** de cada notebook que instala las dependencias automáticamente.

### Ejecución del Pipeline Completo

El proyecto sigue un pipeline secuencial. **Es importante ejecutar los notebooks en orden**:

```bash
# 1. Activar entorno virtual
source venv/bin/activate

# 2. Iniciar Jupyter
jupyter notebook
# O: jupyter lab

# 3. Ejecutar notebooks en orden:
#    a) 01_EDA_Analysis.ipynb
#    b) 02_Wavelet_Analysis.ipynb
#    c) 03_BoF_Clasificacion.ipynb
#    d) 04_DeepConvNet_CNN.ipynb
```

#### Ejecución Automatizada (sin interfaz gráfica)

```bash
# Ejecutar todos los notebooks en orden
jupyter nbconvert --to notebook --execute 01_EDA_Analysis.ipynb
jupyter nbconvert --to notebook --execute 02_Wavelet_Analysis.ipynb
jupyter nbconvert --to notebook --execute 03_BoF_Clasificacion.ipynb
jupyter nbconvert --to notebook --execute 04_DeepConvNet_CNN.ipynb
```

## 📚 Descripción Detallada de los Notebooks

### 1. Análisis Exploratorio (EDA) - `01_EDA_Analysis.ipynb`

**Objetivo**: Analizar las características básicas de los datos EEG y verificar su calidad.

**Qué hace**:
- Carga los archivos .set/.fdt desde `left_imag/` y `right_imag/`
- Calcula la Densidad Espectral de Potencia (PSD) usando el método de Welch
- Analiza las bandas de frecuencia μ (10-12 Hz) y β (18-26 Hz)
- Calcula correlaciones intercanales
- Genera visualizaciones y reportes

**Salidas** (en `results/eda/`):
- `psd_avg.png`: Gráfico de PSD promedio
- `corr_heatmap.png`: Mapa de calor de correlaciones
- `psd_bandpower_per_channel.csv`: Potencia por banda y canal

**Tiempo estimado**: 5-10 minutos

### 2. Análisis de Wavelets - `02_Wavelet_Analysis.ipynb`

**Objetivo**: Extraer características tiempo-frecuencia usando transformadas wavelet.

**Qué hace**:
- Aplica **Transformada Wavelet Continua (CWT)** con wavelet Morlet compleja
- Aplica **Transformada Wavelet Discreta (DWT)** con wavelet Daubechies 4
- Extrae características por canal:
  - Energía en bandas alfa y beta
  - Frecuencia dominante
  - Entropía espectral
  - Estadísticas de coeficientes DWT
- Genera un descriptor de 9 dimensiones por canal

**Salidas** (en `data/bof_features/` y `results/wavelets/`):
- `X_bof_features.npy`: Matriz de características (880 ensayos × 64 canales × 9 descriptores)
- `y_labels.npy`: Etiquetas de clase (0=left, 1=right)
- `trial_to_subject.npy`: Mapeo de ensayos a sujetos
- `bof_metadata.json`: Metadatos del dataset

**Tiempo estimado**: 15-30 minutos

### 3. Clasificación BoF-SVM - `03_BoF_Clasificacion.ipynb`

**Objetivo**: Implementar y optimizar el modelo Bag of Features + SVM.

**Qué hace**:
- Redimensiona los datos a formato BoF: (ensayos, canales, descriptores)
- Construye un vocabulario visual mediante clustering K-means (MiniBatchKMeans)
- Codifica cada ensayo en un histograma de "palabras visuales"
- Realiza **Grid Search** para optimizar hiperparámetros:
  - Número de clusters K: {50, 100, 150}
  - Parámetro de regularización SVM C: {1.0, 10.0, 50.0}
- Evalúa mediante **Validación Cruzada por Grupos (GroupKFold)** con 5 pliegues
- Genera matriz de confusión y métricas de evaluación

**Salidas** (en `results/bof_svm/`):
- `best_params.json`: Mejores hiperparámetros encontrados
- `grid_search_results.csv`: Resultados de todas las combinaciones
- `confusion_matrix.npy`: Matriz de confusión final
- `summary.txt`: Resumen de resultados

**Tiempo estimado**: 10-20 minutos

### 4. DeepConvNet (CNN) - `04_DeepConvNet_CNN.ipynb`

**Objetivo**: Implementar y entrenar una arquitectura CNN profunda para clasificación.

**Qué hace**:
- Implementa arquitectura DeepConvNet adaptada de Schirrmeister et al. (2017)
- Arquitectura: 4 bloques convolucionales + capas totalmente conectadas
- Divide datos: 80% entrenamiento, 10% validación, 10% prueba
- Entrena con optimizador Adam, Early Stopping
- Evalúa con métricas de clasificación

**Salidas** (en `results/deepconvnet/`):
- `deepconvnet_baseline.pth`: Modelo entrenado guardado
- `metrics.npy`: Métricas de evaluación
- `summary.txt`: Resumen de resultados

**Tiempo estimado**: 30-60 minutos (depende del hardware)

## 📊 Resultados

### Comparación de Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| BoF-SVM (K=50, C=10.0) | 52.84% | 52.37% | 57.12% | 0.5451 |
| **DeepConvNet** | **67.42%** | **67.42%** | **67.42%** | **0.6742** |

### Visualizaciones Disponibles

Las gráficas generadas están en `results/figures/`:
- `confusion_matrix_bof_svm.pdf`: Matriz de confusión del modelo BoF-SVM
- `metrics_comparison.pdf`: Comparación de métricas entre ambos modelos
- `grid_search_heatmap.pdf`: Resultados del grid search de hiperparámetros

### Artículo Científico

El artículo completo en LaTeX está disponible en `articulo.md`, incluyendo:
- Revisión de literatura
- Metodología detallada
- Resultados y análisis comparativo
- Discusión y conclusiones

## 🔧 Configuración y Parámetros

### Parámetros Principales del Análisis

- **Filtrado de frecuencia**: 8-30 Hz (bandas μ y β)
- **Banda μ**: 10-12 Hz
- **Banda β**: 18-26 Hz
- **Duración de trial**: 9 segundos
- **Canales EEG**: 64 (estándar 10-20)
- **Sujetos**: 20 (S001-S020)
- **Ensayos totales**: 880 (44 por sujeto, balanceado)

### Parámetros del Modelo BoF-SVM

- **Clusters K**: 50 (óptimo encontrado por grid search)
- **SVM C**: 10.0 (óptimo encontrado por grid search)
- **Kernel SVM**: RBF (radial)
- **Validación**: GroupKFold con 5 pliegues
- **Semilla aleatoria**: 42 (reproducibilidad)

### Parámetros del Modelo DeepConvNet

- **Arquitectura**: 4 bloques convolucionales
- **Tasa de aprendizaje**: 0.001
- **Batch size**: 16
- **Épocas máximas**: 100 (con Early Stopping)
- **División de datos**: 80/10/10 (train/val/test)

## 📦 Dependencias Principales

El proyecto utiliza las siguientes librerías (especificadas en `pyproject.toml`):

- **Procesamiento de señales**: `mne>=1.10.0`, `scipy>=1.16.0`
- **Wavelets**: `PyWavelets>=1.9.0`
- **Machine Learning**: `scikit-learn>=1.7.0`
- **Deep Learning**: `torch>=2.2.0` (para DeepConvNet)
- **Análisis de datos**: `numpy>=1.26.0`, `pandas>=2.3.0`
- **Visualización**: `matplotlib>=3.10.0`, `seaborn>=0.13.0`
- **Utilidades**: `tqdm>=4.67.0`, `joblib>=1.5.0` (barras de progreso)
- **Jupyter**: `jupyter>=1.1.0`, `ipykernel>=6.0.0`

**Nota**: Todas las dependencias están definidas en `pyproject.toml`. Se recomienda usar `pip install -e .` para instalar todas las dependencias de forma automática.

## ❓ Preguntas Frecuentes

### ¿Puedo ejecutar los notebooks en cualquier orden?

**No**. Los notebooks tienen dependencias:
1. `01_EDA_Analysis.ipynb` debe ejecutarse primero
2. `02_Wavelet_Analysis.ipynb` depende de los datos generados por el EDA
3. `03_BoF_Clasificacion.ipynb` depende de las características wavelet
4. `04_DeepConvNet_CNN.ipynb` puede ejecutarse independientemente (usa datos preprocesados)

### ¿Qué pasa si ya existen archivos de salida?

Los notebooks sobrescriben los archivos de salida. Si quieres conservar resultados anteriores, haz una copia antes de ejecutar.

### ¿Cuánto tiempo toma ejecutar todo el pipeline?

- EDA: ~5-10 minutos
- Wavelets: ~15-30 minutos
- BoF-SVM: ~10-20 minutos
- DeepConvNet: ~30-60 minutos

**Total estimado**: 1-2 horas (depende del hardware)

### ¿Necesito GPU para ejecutar DeepConvNet?

No es estrictamente necesario, pero acelerará el entrenamiento significativamente. El modelo puede entrenarse en CPU, pero tomará más tiempo.

### ¿Cómo interpreto los resultados?

- **Accuracy > 50%**: Mejor que el azar (clasificación binaria)
- **F1-Score**: Equilibrio entre precisión y recall
- **GroupKFold**: Evalúa generalización a nuevos sujetos (más conservador que validación estándar)

## 📝 Notas Adicionales

- Los datos están en formato **EEGLAB** (.set/.fdt)
- Todos los procesos utilizan **semilla aleatoria fija (42)** para reproducibilidad
- Los resultados se guardan en formato NumPy, CSV, JSON y PNG/PDF
- El proyecto está optimizado para validación **inter-sujeto** (más realista para BCI)

## 📄 Licencia y Referencias

Este proyecto es parte de un estudio comparativo entre métodos clásicos y de deep learning para clasificación de señales EEG. Para más detalles, consulta el artículo en `articulo.md`.

### Referencias Principales

- Schirrmeister, R. T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391-5420.
- Asghar, M. A., et al. (2019). EEG-Based Multi-Modal Emotion Recognition using Bag of Deep Features. *Sensors*, 19(23), 5218.

## 🤝 Contribuciones

Este es un proyecto de investigación académica. Para preguntas o sugerencias, consulta el artículo o la documentación en los notebooks.

---

**Última actualización**: Noviembre 2024  
**Estado del proyecto**: ✅ Completado - Todos los análisis implementados y resultados disponibles