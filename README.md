# Análisis de Datos MI-EEG

Este proyecto analiza datos de interfaz cerebro-computadora (BCI) de imaginación motora (MI) usando EEGLAB.

## Estructura del Proyecto

```
datos_BCI/
├── eda.py                          # Script principal de análisis
├── left_imag/                      # Datos de imaginación motora mano izquierda (20 archivos)
├── right_imag/                     # Datos de imaginación motora mano derecha (20 archivos)
├── venv/                           # Entorno virtual Python
├── pyproject.toml                  # Configuración del proyecto y dependencias
├── .gitignore                      # Archivos a ignorar en git
└── reports/                        # Directorio de salida (generado por eda.py)
    ├── psd_avg.png                 # PSD promedio con bandas μ/β
    ├── corr_heatmap.png            # Mapa de calor de correlación
    ├── psd_bandpower_per_channel.csv # Potencia por banda y canal
    └── corr_region_summary.txt     # Resumen de correlaciones por región
```

## Configuración del Entorno

```bash
# Crear entorno virtual (si no existe)
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias usando pip
pip install -e .

# O instalar dependencias directamente desde pyproject.toml
pip install mne scipy numpy matplotlib seaborn pandas tqdm

# Ejecutar análisis principal
python eda.py
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

## Configuración del Análisis

El análisis está configurado para:
- **Bandas de frecuencia**: 8-30 Hz (filtro μ/β)
- **Banda μ**: 10-12 Hz
- **Banda β**: 18-26 Hz
- **Ventana de trial**: 9 segundos por defecto
- **Método PSD**: Welch con segmentos de 2 segundos y 50% overlap

## Uso

```bash
# Análisis con directorio por defecto
python eda.py

# Análisis con directorio personalizado
python eda.py --data-root /ruta/a/datos

# Especificar directorio de salida personalizado
python eda.py --output-dir mi_reports

# Combinar opciones
python eda.py --data-root /ruta/datos --output-dir /ruta/salida

# Verificar instalación
python eda.py --help
```

## Datos

- **left_imag/**: 20 archivos .set/.fdt con datos de imaginación motora mano izquierda
- **right_imag/**: 20 archivos .set/.fdt con datos de imaginación motora mano derecha
- Cada archivo corresponde a un sujeto (S001-S020)
- Los archivos están en formato EEGLAB (.set/.fdt)

## Notas

- El script procesa todos los archivos disponibles en ambos directorios
- Los datos ya vienen epocados o se segmentan automáticamente si son continuos
- Los resultados se guardan en el directorio `reports/` por defecto
- El directorio de salida se crea automáticamente si no existe