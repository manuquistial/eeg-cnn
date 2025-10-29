# Análisis de Datos MI-EEG con Wavelets (Notebooks)

Este proyecto analiza datos de interfaz cerebro-computadora (BCI) de imaginación motora (MI) usando EEGLAB, implementando análisis exploratorio y transformadas wavelet mediante notebooks de Jupyter. El proyecto está completamente preparado para la implementación de Bag of Features con datos optimizados.

## Estructura del Proyecto

```
datos_BCI/
├── 01_EDA_Analysis.ipynb           # Notebook: Análisis exploratorio (PSD, correlación)
├── 02_Wavelet_Analysis.ipynb       # Notebook: Análisis de wavelets (CWT y DWT)
├── bof_template.py                 # Template para implementar Bag of Features
├── left_imag/                      # Datos de imaginación motora mano izquierda (20 archivos)
├── right_imag/                     # Datos de imaginación motora mano derecha (20 archivos)
├── venv/                           # Entorno virtual Python
├── pyproject.toml                  # Configuración del proyecto y dependencias
├── README.md                       # Este archivo
├── shared_data/                    # Datos compartidos entre notebooks
│   ├── X_data.npy                  # Datos concatenados (trials, channels, time)
│   ├── ch_names.npy                # Nombres de canales
│   ├── sfreq.npy                   # Frecuencia de muestreo
│   ├── data_dimensions.npy         # Dimensiones de los datos
│   ├── subjects_info.csv           # Información de sujetos
│   ├── region_info.json            # Información de regiones cerebrales
│   └── config_params.json          # Parámetros de configuración
├── bof_data/                       # Datos específicos para Bag of Features
│   ├── X_bof_features.npy          # Características normalizadas para BoF
│   ├── y_labels.npy                # Etiquetas de clase (0=left, 1=right)
│   ├── bof_feature_names.txt       # Nombres de características seleccionadas
│   ├── scaler_bof.pkl              # Normalizador entrenado para BoF
│   ├── bof_metadata.json           # Metadatos completos del dataset
│   ├── bof_config.json             # Configuración y parámetros recomendados
│   ├── trial_to_subject.npy        # Mapeo de trials a sujetos
│   └── trial_to_task.npy           # Mapeo de trials a tareas
├── reports/                        # Directorio de salida EDA
│   ├── psd_avg.png                 # PSD promedio con bandas μ/β
│   ├── corr_heatmap.png            # Mapa de calor de correlación
│   ├── psd_bandpower_per_channel.csv # Potencia por banda y canal
│   └── corr_region_summary.txt     # Resumen de correlaciones por región
└── wavelet_reports/                # Directorio de salida wavelets (generado por notebook)
    ├── wavelet_features.npy        # Características normalizadas para BoF
    ├── feature_names.txt           # Nombres de características
    ├── channel_info.csv            # Información de canales y regiones
    ├── subjects_info.csv           # Información de sujetos y tareas
    └── wavelet_config.json         # Parámetros de configuración wavelets
```

## Configuración del Entorno

### Opción 1: Usando el entorno virtual (Recomendado)
```bash
# Activar entorno virtual existente
source venv/bin/activate

# Las dependencias ya están instaladas
```

### Opción 2: Instalación manual
```bash
# Instalar dependencias principales
pip install mne PyWavelets scikit-learn matplotlib pandas numpy scipy tqdm

# O usar pyproject.toml
pip install -e .
```

### Opción 3: Desde los notebooks
Si ejecutas los notebooks en un entorno nuevo, ejecuta la **primera celda** de cada notebook que contiene la instalación automática:

**Celda 1 - Instalación de dependencias:**
```python
# Celda de instalación de dependencias
# Ejecutar esta celda SOLO si necesitas instalar las librerías en un entorno nuevo

%pip install mne
%pip install PyWavelets
%pip install scikit-learn
%pip install matplotlib
%pip install pandas
%pip install numpy
%pip install scipy
%pip install tqdm

print("Instalación de dependencias completada")
```

**Celda 2 - Importación de librerías:**
```python
# Importar librerías necesarias
import os
import re
# ... resto de imports
```

### Crear entorno virtual nuevo (si es necesario)
```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -e .
```

## Configuración del Proyecto

Este proyecto utiliza `pyproject.toml` para la gestión de dependencias y configuración, siguiendo las mejores prácticas modernas de Python.

### Dependencias principales:
- `mne>=1.5.0`: Para procesamiento de señales EEG
- `scipy>=1.11.0`: Operaciones científicas
- `numpy>=1.24.0`: Arrays numéricos
- `matplotlib>=3.7.0`: Visualización
- `seaborn>=0.12.0`: Visualización estadística
- `pandas>=2.0.0`: Manipulación de datos
- `tqdm>=4.65.0`: Barras de progreso
- `PyWavelets>=1.4.0`: Transformadas wavelet
- `scikit-learn>=1.3.0`: Machine learning y clustering

### Dependencias de desarrollo (opcionales):
- `pytest>=7.0.0`: Testing
- `black>=22.0.0`: Formateador de código
- `flake8>=4.0.0`: Linter
- `mypy>=0.950`: Verificación de tipos

### Ventajas de pyproject.toml:
- ✅ **Estándar moderno**: PEP 518/621 compliant
- ✅ **Metadatos completos**: Información del proyecto, autores, descripción
- ✅ **Herramientas integradas**: Configuración para Black, MyPy, etc.
- ✅ **Gestión de dependencias**: Dependencias principales y opcionales
- ✅ **Instalación editable**: `pip install -e .` para desarrollo
- ✅ **Estructura simplificada**: Fácil de mantener y usar

## Análisis Realizado

### 1. Análisis Exploratorio de Datos (EDA) - `eda.py`

El script `eda.py` realiza un análisis espectral y de correlación que incluye:

1. **Análisis espectral (PSD)**: 
   - Densidad espectral de potencia en 8-30 Hz
   - Análisis de bandas μ (10-12 Hz) y β (18-26 Hz)
   - Picos por canal en cada banda

2. **Correlación intercanal**:
   - Mapa de calor de correlación entre canales
   - Resumen por regiones cerebrales (Frontal, Central, Parietal, Occipital)

3. **Resultados** (guardados en `reports/`):
   - `psd_avg.png`: PSD promedio con bandas μ/β sombreadas
   - `corr_heatmap.png`: Mapa de calor de correlación intercanal
   - `psd_bandpower_per_channel.csv`: Potencia media por banda y picos por canal
   - `corr_region_summary.txt`: Resumen de correlaciones por regiones

### 2. Análisis de Wavelets - `wavelet_analysis.py`

El script `wavelet_analysis.py` implementa análisis avanzado de wavelets:

1. **Transformada Wavelet Continua (CWT)**:
   - Análisis tiempo-frecuencia usando wavelet de Morlet
   - Escalas logarítmicas para cobertura completa del espectro
   - Extracción de características por bandas de frecuencia

2. **Transformada Wavelet Discreta (DWT)**:
   - Descomposición multiresolución usando Daubechies 4
   - Análisis por niveles de descomposición
   - Características estadísticas por nivel

3. **Características extraídas**:
   - Energía por banda de frecuencia (delta, theta, alpha, beta, gamma)
   - Potencia máxima por escala
   - Frecuencia dominante
   - Entropía espectral
   - Estadísticas por nivel DWT

4. **Resultados** (guardados en `wavelet_reports/`):
   - `wavelet_features.csv`: Características extraídas para BoF
   - `wavelet_spectrogram.png`: Espectrograma wavelet promedio
   - `wavelet_energy_distribution.png`: Distribución de energía por escalas
   - `wavelet_cwt_coefficients.npy`: Coeficientes CWT completos (opcional)
   - `wavelet_dwt_coefficients.npy`: Coeficientes DWT completos (opcional)

## Próximos Pasos: Implementación de Bag of Features (BoF)

El proyecto está preparado para implementar Bag of Features como siguiente paso. Las características wavelet extraídas están listas para ser utilizadas en el pipeline BoF.

### Archivos Preparados para BoF

El análisis de wavelets genera los siguientes archivos que serán utilizados por BoF:

- **`wavelet_reports/wavelet_features.csv`**: Características extraídas por trial y canal
- **`wavelet_reports/wavelet_features_matrix.npy`**: Matriz de características en formato numpy
- **`wavelet_reports/channel_info.csv`**: Información de canales EEG

### Implementación Sugerida de BoF

Para implementar Bag of Features, se recomienda crear un script `bof_classification.py` que incluya:

1. **Construcción del vocabulario visual**:
   - Cargar características desde `wavelet_features.csv`
   - Aplicar clustering K-means para crear vocabulario visual
   - Normalizar características antes del clustering

2. **Codificación de características**:
   - Convertir características a histogramas de palabras visuales
   - Normalizar histogramas para clasificación
   - Preparar datos para entrenamiento/test

3. **Clasificación**:
   - Implementar múltiples algoritmos (SVM, Random Forest, Logistic Regression)
   - División train/test estratificada
   - Validación cruzada opcional

4. **Evaluación**:
   - Métricas: accuracy, precision, recall, F1-score, AUC
   - Matrices de confusión
   - Curvas ROC
   - Importancia de características

5. **Visualización**:
   - PCA para visualización de histogramas BoF
   - Distribución de palabras visuales
   - Comparación de rendimiento entre algoritmos

### Template de Implementación

Se incluye un archivo template `bof_template.py` que muestra la estructura completa para implementar BoF:

```bash
# Copiar template y completar implementación
cp bof_template.py bof_classification.py

# Editar bof_classification.py para completar las funciones marcadas con TODO
# Ejecutar implementación
python bof_classification.py
```

El template incluye:
- Clase `BagOfFeatures` con métodos para vocabulario y codificación
- Funciones para carga de características wavelet
- Pipeline de entrenamiento y evaluación
- Estructura para múltiples clasificadores (SVM, Random Forest, Logistic Regression)
- Marcadores TODO para guiar la implementación

### Estructura del Template

```python
# bof_template.py
class BagOfFeatures:
    def __init__(self, vocab_size=50):
        # TODO: Inicializar componentes
    
    def build_vocabulary(self, features):
        # TODO: Implementar clustering K-means
    
    def encode_features(self, features):
        # TODO: Convertir a histogramas BoF

def load_wavelet_features(features_file):
    # TODO: Cargar características desde CSV

def train_classifiers(X_train, y_train):
    # TODO: Entrenar múltiples clasificadores

def evaluate_classifiers(classifiers, X_test, y_test):
    # TODO: Evaluar y visualizar resultados
```

## Configuración del Análisis

El análisis está configurado para:
- **Bandas de frecuencia**: 8-30 Hz (filtro μ/β)
- **Banda μ**: 10-12 Hz
- **Banda β**: 18-26 Hz
- **Ventana de trial**: 9 segundos por defecto
- **Método PSD**: Welch con segmentos de 2 segundos y 50% overlap

## Uso

### Ejecutar Notebooks

#### 1. Análisis Exploratorio (EDA)
```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar notebook de EDA
jupyter notebook 01_EDA_Analysis.ipynb

# O ejecutar directamente
jupyter nbconvert --to notebook --execute 01_EDA_Analysis.ipynb
```

#### 2. Análisis de Wavelets
```bash
# Ejecutar notebook de wavelets
jupyter notebook 02_Wavelet_Analysis.ipynb

# O ejecutar directamente
jupyter nbconvert --to notebook --execute 02_Wavelet_Analysis.ipynb
```

### Pipeline Recomendado

```bash
# 1. Activar entorno virtual
source venv/bin/activate

# 2. Ejecutar análisis exploratorio (genera datos compartidos)
jupyter nbconvert --to notebook --execute 01_EDA_Analysis.ipynb

# 3. Ejecutar análisis de wavelets (usa datos del EDA)
jupyter nbconvert --to notebook --execute 02_Wavelet_Analysis.ipynb

# 4. Implementar Bag of Features
# Los archivos en bof_data/ están listos para implementar BoF
```

**Nota importante**: 
- El notebook de wavelets depende de los datos generados por el EDA
- Siempre ejecuta primero `01_EDA_Analysis.ipynb` para generar el directorio `shared_data/`
- Los datos específicos para BoF se generan en `bof_data/` después de ejecutar el análisis de wavelets

### Desarrollo Interactivo

```bash
# Iniciar Jupyter Lab para desarrollo interactivo
source venv/bin/activate
jupyter lab

# O Jupyter Notebook clásico
jupyter notebook
```

## Datos

- **left_imag/**: 20 archivos .set/.fdt con datos de imaginación motora mano izquierda
- **right_imag/**: 20 archivos .set/.fdt con datos de imaginación motora mano derecha
- Cada archivo corresponde a un sujeto (S001-S020)
- Los archivos están en formato EEGLAB (.set/.fdt)

## Estado Actual del Proyecto

### ✅ Implementado

1. **Análisis Exploratorio de Datos (EDA)** - `01_EDA_Analysis.ipynb`
   - Análisis espectral (PSD) con bandas μ/β
   - Correlación intercanal
   - Visualizaciones y reportes
   - Generación de datos compartidos
   - ✅ **Completado**

2. **Análisis de Wavelets** - `02_Wavelet_Analysis.ipynb`
   - Transformada Wavelet Continua (CWT) con Morlet
   - Transformada Wavelet Discreta (DWT) con Daubechies 4
   - Extracción de características por bandas de frecuencia
   - Preparación específica de datos para BoF
   - ✅ **Completado**

3. **Preparación Completa para BoF**
   - Template completo en `bof_template.py`
   - Datos específicos optimizados en `bof_data/`
   - Características seleccionadas y normalizadas
   - Etiquetas de clase y metadatos completos
   - ✅ **Listo para implementar**

### 🔄 Próximo Paso: Bag of Features

El proyecto está completamente preparado para implementar Bag of Features. Los archivos generados por el análisis de wavelets contienen todas las características necesarias:

- **Características CWT**: Energía por banda, frecuencia dominante, entropía espectral
- **Características DWT**: Energía y estadísticas por nivel de descomposición
- **Formato preparado**: CSV con características por trial y canal, matriz numpy para procesamiento eficiente

### 📋 Archivos Listos para BoF

Después de ejecutar `python wavelet_analysis.py`, tendrás:

```
wavelet_reports/
├── wavelet_features.csv         # ← Archivo principal para BoF
├── wavelet_features_matrix.npy  # ← Matriz numpy para clustering
├── channel_info.csv            # ← Información de canales
├── wavelet_spectrogram.png     # ← Visualización CWT
└── wavelet_energy_distribution.png # ← Distribución de energía
```

## Notas

- El script procesa todos los archivos disponibles en ambos directorios
- Los datos ya vienen epocados o se segmentan automáticamente si son continuos
- Los resultados se guardan en directorios separados (`reports/` y `wavelet_reports/`)
- Los directorios de salida se crean automáticamente si no existen
- Las características wavelet están optimizadas para clustering K-means