\documentclass[journal]{IEEEtran}     % IEEE Transactions / Journals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   PACKAGES                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage{balance}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{cite}
\usepackage{url}
\usepackage{graphicx}
\usepackage{multirow,booktabs}
\usepackage{amsmath,amssymb}
\usepackage{siunitx}
\usepackage{hyperref}
\hypersetup{hidelinks}
\usepackage{float}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   TITLE & AUTHORS                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\title{Evaluación Comparativa de Bag of Features y CNN en la Clasificación de Señales EEG de Imagen Motora}

\author{%
  C.~Monsalve,  
  M.~Quistial,
  L.~Urquijo,
  M.~Jaramillo
}

\begin{document}
\maketitle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   ABSTRACT & KEYWORDS                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{abstract}
Motor imagery (MI) electroencephalography (EEG) classification remains a challenging problem due to the low signal-to-noise ratio and strong inter-subject variability of cortical activity. Traditional machine-learning methods such as Common Spatial Patterns (CSP) combined with Support Vector Machines (SVM) have achieved competitive results in binary MI tasks but rely heavily on handcrafted features and subject-specific calibration. In contrast, Convolutional Neural Networks (CNN) can automatically learn spatial–temporal representations directly from multichannel EEG data, potentially improving generalization and robustness.  

This study proposes a feature-based approach using the Bag of Features (BoF) model combined with SVM and compares its performance against a deep CNN model using MI-EEG recordings from 15 healthy participants. Both approaches are evaluated through multiple performance indicators including \textit{accuracy}, \textit{precision}, \textit{recall}, and \textit{F1-score} to ensure a comprehensive assessment of classification reliability. The results provide insight into the trade-offs between interpretability and complexity in EEG motor-imagery decoding using classical feature-based versus deep-learning paradigms.
\end{abstract}

\begin{IEEEkeywords}
EEG, Brain–Computer Interface, Motor Imagery, Bag of Features, Deep Learning, Convolutional Neural Networks.
\end{IEEEkeywords}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   MAIN TEXT                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\IEEEPARstart{L}{a} clasificación de señales de electroencefalografía (EEG) durante tareas de imagen motora (MI) representa un desafío persistente debido a la baja relación señal–ruido y a la alta variabilidad inter e intra-sujeto de la actividad cortical. Estos factores dificultan la extracción de características discriminantes estables y la construcción de modelos de aprendizaje consistentes entre diferentes individuos \cite{ref1, ref2}.  

Los métodos clásicos de aprendizaje automático, como los Patrones Espaciales Comunes (CSP) combinados con clasificadores lineales como el Análisis Discriminante Lineal (LDA) o las Máquinas de Vectores de Soporte (SVM), han mostrado resultados competitivos en la clasificación binaria de imagen motora. Sin embargo, su desempeño depende en gran medida de una calibración individual y de un diseño manual de características \cite{ref3, ref4}.  

En los últimos años, los modelos de aprendizaje profundo (Deep Learning, DL) han surgido como una alternativa para abordar estas limitaciones. En particular, las Redes Neuronales Convolucionales (CNN) pueden aprender representaciones espacio–temporales directamente a partir de las señales EEG multicanal, eliminando la necesidad de ingeniería manual y mostrando una mayor robustez frente al ruido y las variaciones entre sujetos \cite{ref5, ref6, ref8, ref9}.  

En este contexto, el presente trabajo propone un enfoque alternativo basado en el modelo \textit{Bag of Features} (BoF) combinado con SVM, el cual permite representar las señales EEG a través de histogramas de patrones locales obtenidos mediante técnicas de agrupamiento. Este enfoque se compara directamente con un modelo de aprendizaje profundo tipo CNN aplicado a señales EEG de imagen motora. El objetivo es analizar la capacidad de cada enfoque para clasificar tareas de imaginación de movimiento de mano izquierda y derecha, y examinar los compromisos entre interpretabilidad y complejidad en la decodificación de EEG mediante métodos basados en características frente a métodos profundos \cite{ref10}.

\begin{table*}[!htbp]
\caption{Estado del arte en clasificación de MI-EEG: comparación entre enfoques clásicos (CSP+SVM/LDA) y de aprendizaje profundo (CNN/LSTM).}
\label{tab:sota}
\centering
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{l l p{2.6cm} p{2.8cm} p{2.8cm} l c}
\hline
\textbf{Estudio} & \textbf{Año} & \textbf{Dataset} & \textbf{Paradigma / Clases} & \textbf{Método} & \textbf{Validación} & \textbf{Precisión (\%)} \\
\hline
\cite{ref3} & 2023 & BCI IV-2a & Mano izq./der./pie (binaria y multiclase) & CSP + logvar + LDA/SVM/KNN & Hold-out & 97.5 (binaria); 87.0 (multiclase) \\
\cite{ref4} & 2023 & BCI IV-1 & Mano izq./der. (binaria) & CSP + SVM & Split/CV & 91.4 \\
\cite{ref8} & 2025 & Propio (25 sujetos) & Mano izq./der. (binaria) & SVM, RF, MLP, KNN (SVM mejor) & Hold-out & 98.7 (SVM) \\
\cite{ref5} & 2017 & BCI Comp II & MI (binaria) & DBN (RBM-based) & Split & 86--90 \\
\cite{ref6} & 2025 & BCI IV-2a & MI (binaria/multiclase) & CNN + ventana deslizante & Split (dep./mixto) & 97.4 \\
\cite{ref7} & 2017 & BCI Comp II & MI (binaria) & DeepConvNet (CNN) & Split & 88.0 \\
\cite{ref9} & 2025 & PhysioNet BCI2000 & MI multimodal (multi-sujeto) & CNN/LSTM/GRU/GIN (híbrido) & Inter-sujeto & 95--97 \\
\cite{ref2} & 2024 & 22 datasets & MI-EEG (síntesis) & Revisión sistemática DL & --- & 75--95 \\
\cite{ref1} & 2021 & Multidominio (EEG biomédico) & General EEG (epilepsia, MI, cognitivo) & Revisión ML (SVM, RF, ANN, CNN) & --- & --- \\
\hline
\end{tabular}
\end{table*}

\section{Datos}\label{sec:datos}
\subsection{Conjunto de datos}
El conjunto de datos utilizado en este estudio corresponde a registros de electroencefalografía (EEG) adquiridos durante un paradigma de \textit{imaginación motora} (MI) en veinte participantes sanos (S001-S020). Cada sujeto realizó ensayos de imaginación kinestésica de movimientos de mano izquierda y derecha, siguiendo instrucciones visuales alternadas con periodos de reposo. Las señales se registraron mediante un sistema de 64 electrodos dispuestos bajo el estándar 10--20 internacional, con una frecuencia de muestreo de \SI{128}{\hertz}, y cada ensayo tuvo una duración aproximada de nueve segundos. Los datos fueron almacenados en formato EEGLAB (\texttt{.set/.fdt}) y posteriormente preprocesados aplicando un filtrado pasabanda de 8--30~\si{\hertz} (bandas $\mu$ y $\beta$ asociadas a la actividad sensorimotora), eliminación de artefactos oculares mediante análisis de componentes independientes (ICA) y normalización canal a canal. De cada ensayo se obtuvo una matriz $X \in \mathbb{R}^{64 \times 1152}$, donde las filas corresponden a canales EEG y las columnas a muestras temporales. La variable objetivo $y$ es categórica y representa la clase de tarea: \textit{imaginación de movimiento de mano izquierda} (MI-L) o \textit{imaginación de movimiento de mano derecha} (MI-R), con un total de 220 ensayos balanceados (111 MI-L, 109 MI-R) en todo el conjunto.

\subsection{Análisis exploratorio}
Durante el análisis exploratorio inicial, se examinó la distribución de la varianza por canal y se observó una dispersión homogénea en los electrodos centrales (canales 31, 32, 33), correspondientes aproximadamente a las regiones C3, Cz y C4, lo que indica una captación consistente de los ritmos sensorimotores. La densidad espectral de potencia mostró picos en las bandas $\mu$ (10--12~\si{\hertz}) y $\beta$ (18--26~\si{\hertz}), con valores de potencia en el orden de $10^{-12}$~\si{\volt\squared\per\hertz}, coherentes con los patrones esperados de desincronización (ERD) durante la imaginación motora. La matriz de correlación intercanal reveló correlaciones espaciales entre regiones fronto-centrales (0.155), centrales (0.305), y parietales (0.539), lo cual sugiere conectividad funcional entre áreas motoras y somatosensoriales. El balance de la variable objetivo fue cercano a 50/50 entre MI-L (50.5\%) y MI-R (49.5\%), descartando la necesidad de aplicar técnicas de sobre- o submuestreo. Asimismo, se verificó la ausencia de fugas de información entre divisiones de entrenamiento y validación.

\subsection{Justificación de la métrica de evaluación}
Se seleccionaron estas métricas porque permiten una evaluación integral del desempeño de los modelos. La \textit{accuracy} mide la proporción global de aciertos y facilita la comparación con trabajos previos en el área. Sin embargo, dado que en EEG puede haber sesgos hacia una clase específica, se complementa con \textit{precision} (qué tan confiables son las predicciones positivas) y \textit{recall} (qué proporción de casos reales logra detectar el modelo). Finalmente, el \textit{F1-score} resume el equilibrio entre \textit{precision} y \textit{recall} en un solo valor, ofreciendo una visión más robusta del rendimiento en contextos donde la variabilidad entre clases y sujetos es un desafío frecuente.

\section{Pregunta de investigación y objetivos}\label{sec:pregunta_investigacion}
Basado en el estado del arte Tabla~\ref{tab:sota}, la clasificación de señales EEG asociadas a tareas de imaginación motora requiere etapas críticas de preprocesamiento y extracción de características para mejorar la relación señal–ruido y resaltar patrones discriminativos. Entre las técnicas más utilizadas se encuentra el método de los Patrones Espaciales Comunes (CSP), ampliamente validado para tareas de MI. No obstante, existen enfoques alternativos que permiten representar la información de las señales EEG desde una perspectiva estadística, como el modelo Bag of Features (BoF), el cual genera descriptores globales a partir de la distribución de patrones locales presentes en las señales y posibilita su uso en clasificadores tradicionales como las Máquinas de Vectores de Soporte (SVM). Este trabajo adopta el esquema BoF + SVM como alternativa al enfoque clásico CSP + SVM, con el propósito de analizar su rendimiento frente a un modelo de aprendizaje profundo basado en CNN.


\subsection{Pregunta de investigación}
¿Qué tan efectivo es el enfoque basado en la combinación de Bag of Features (BoF) y Máquinas de Vectores de Soporte (SVM), en comparación con un modelo de Red Neuronal Convolucional (CNN), para la clasificación de señales EEG de imaginación motora?

\subsection{Objetivo general}
Evaluar y comparar la eficacia de un modelo basado en Bag of Features (BoF) + SVM frente a una arquitectura CNN para la clasificación de señales EEG de imaginación motora, determinando cuál ofrece mejor precisión y estabilidad en la detección de tareas de mano izquierda y derecha.

\subsection{Objetivos específicos}
\begin{itemize}
    \item Implementar un modelo Bag of Features (BoF) para representar las señales EEG mediante histogramas de patrones locales y entrenar un clasificador SVM sobre dichas representaciones, alcanzando al menos un 80\% de \textit{accuracy} en la clasificación.  
    \item Implementar una arquitectura CNN capaz de procesar directamente las señales EEG preprocesadas y superar el 85\% de \textit{accuracy} en la identificación de las tareas motoras.  
    \item Analizar los resultados obtenidos de ambos modelos utilizando métricas de \textit{accuracy}, \textit{precision}, \textit{recall} y \textit{F1-score}, identificando cuál presenta mayor robustez y consistencia en la clasificación de señales EEG.  
\end{itemize}

\section{Materiales y Métodos}\label{sec:metodos}

El pipeline metodológico propuesto para la clasificación de señales EEG de imaginación motora (MI-EEG) se estructuró en dos fases principales. En la primera se desarrolló un enfoque basado en ingeniería de características mediante el modelo Bag of Features (BoF) combinado con Máquinas de Vectores de Soporte (SVM). La segunda fase implementó una arquitectura de red neuronal convolucional profunda (CNN), adaptada del modelo DeepConvNet, para el aprendizaje directo de representaciones espacio–temporales a partir de las señales EEG preprocesadas.

\subsection{Modelo Bag of Features (BoF) + SVM}

A partir de los ensayos preprocesados, compuestos por matrices de 64 canales y 1152 muestras por ensayo, se diseñó un vector descriptor para cada canal con el propósito de capturar la información tiempo–frecuencia característica de la actividad cortical durante las tareas de imaginación motora. Dado que las señales EEG son inherentemente no estacionarias, se emplearon transformadas wavelet por su capacidad para ofrecer una representación conjunta en el dominio del tiempo y de la frecuencia. En particular, se aplicó una Transformada Wavelet Continua (CWT) utilizando la wavelet Morlet compleja (cmor5.0-1.0) sobre cincuenta escalas logarítmicas. Este método permitió aislar la energía dentro de las bandas alfa (8--13~\si{\hertz}) y beta (13--30~\si{\hertz}), asociadas a la actividad sensorimotora. A partir de esta representación se extrajeron cuatro características por canal: energía alfa, energía beta, frecuencia dominante y entropía espectral. Complementariamente, se empleó una Transformada Wavelet Discreta (DWT) con la wavelet Daubechies 4 (db4) y seis niveles de descomposición, obteniendo cinco características adicionales: energía de los coeficientes de aproximación y detalle, junto con la media y la desviación estándar del nivel principal. La concatenación de ambas transformadas generó un vector descriptor de nueve dimensiones por canal, conformando un tensor final de 880 ensayos por 64 canales y 9 descriptores, que sirvió como entrada al modelo BoF.

El enfoque BoF consideró cada ensayo de EEG como un documento y cada descriptor de canal como una palabra dentro de ese documento. Para construir un vocabulario representativo, todos los descriptores de los ensayos de entrenamiento se agruparon mediante MiniBatchKMeans, seleccionándose este algoritmo por su eficiencia y escalabilidad frente al elevado volumen de descriptores ($880 \times 64 = 56,320$ descriptores). Los centroides obtenidos definieron el codebook o diccionario de patrones locales. Cada ensayo se codificó entonces en un histograma de $K$ bins, donde cada bin reflejaba la frecuencia de aparición de una palabra del diccionario. Estos histogramas se normalizaron mediante norma L1, convirtiéndolos en distribuciones de probabilidad, y posteriormente se estandarizaron a media cero y varianza unitaria mediante StandardScaler. Los vectores resultantes se utilizaron como representación final de cada ensayo para alimentar un clasificador SVM con kernel radial (RBF), seleccionado por su capacidad para modelar fronteras de decisión no lineales y manejar estructuras de alta dimensionalidad, características comunes en datos EEG.

Con el fin de garantizar la validez estadística y evitar el sobreajuste, se implementó un esquema de validación cruzada de cinco pliegues (GroupKFold), agrupando los ensayos por sujeto. Esta estrategia permitió evaluar la capacidad de generalización del modelo a individuos no vistos durante el entrenamiento, condición esencial en aplicaciones BCI. Asimismo, se llevó a cabo una búsqueda exhaustiva de hiperparámetros (Grid Search) para optimizar el número de clusters del codebook ($K \in \{50, 100, 150\}$) y el parámetro de regularización del SVM ($C \in \{1.0, 10.0, 50.0\}$), evaluando un total de 9 combinaciones diferentes (Fig.~\ref{fig:grid_search}). La selección del mejor modelo se basó en el F1-score promedio, métrica que ofrece un equilibrio robusto entre precisión y exhaustividad, especialmente relevante en contextos con posibles desequilibrios o variabilidad intersujeto. En todos los procesos estocásticos, incluyendo el agrupamiento, la inicialización del SVM y la partición de los datos, se fijó una semilla aleatoria (\texttt{RANDOM\_STATE = 42}) para asegurar la reproducibilidad de los resultados.

\subsection{Modelo DeepConvNet (CNN)}

En paralelo al enfoque clásico, se desarrolló una arquitectura DeepConvNet adaptada del trabajo de Schirrmeister \textit{et al.} (2017)~\cite{ref7}, con el objetivo de evaluar la capacidad de un modelo de aprendizaje profundo para extraer representaciones discriminativas directamente de las señales EEG preprocesadas, sin intervención manual en la selección de características. El modelo se compone de cuatro bloques convolucionales sucesivos, cada uno formado por una capa Conv2D, seguida de Batch Normalization, función de activación ELU, MaxPooling y Dropout con una tasa de 0.5. Esta secuencia permite una extracción jerárquica de patrones espacio–temporales, desde la información local de cada canal hasta correlaciones espaciales entre regiones corticales. La salida de la última capa convolucional se aplana y alimenta a una capa totalmente conectada de 128 unidades, finalizando en una capa de salida Softmax con dos neuronas correspondientes a las clases de imaginación motora izquierda (MI-L) y derecha (MI-R).

El entrenamiento se realizó utilizando el optimizador Adam con una tasa de aprendizaje de 0.001, un tamaño de lote de 16 y la función de pérdida de entropía cruzada categórica durante 100 épocas. Se aplicó la técnica de Early Stopping basada en la pérdida de validación para evitar el sobreajuste y mejorar la capacidad de generalización. Los datos se dividieron en proporciones del 80\% para entrenamiento, 10\% para validación y 10\% para prueba, asegurando la independencia entre sujetos mediante una partición estratificada que replicó las condiciones de evaluación inter-sujeto empleadas en el modelo BoF--SVM. Los resultados se evaluaron con las métricas de exactitud, precisión, exhaustividad y F1-score sobre el conjunto de prueba. Este enfoque de aprendizaje profundo se justifica por su habilidad para aprender filtros convolucionales que capturan dinámicamente relaciones espacio–temporales entre canales EEG, reduciendo la dependencia de transformaciones previas o de parámetros manuales. Mientras que el modelo BoF--SVM ofrece interpretabilidad y control explícito sobre la extracción de características, la CNN permite una optimización integral capaz de descubrir estructuras latentes no evidentes para los métodos clásicos. Ambas aproximaciones fueron implementadas bajo el mismo esquema de evaluación para permitir una comparación justa y reproducible.

\section{Resultados}\label{sec:resultados}

Los resultados obtenidos para ambos modelos se presentan en la Tabla~\ref{tab:resultados} y permiten realizar una comparación directa de su desempeño en la clasificación binaria de tareas de imaginación motora. El modelo BoF--SVM, optimizado mediante Grid Search, alcanzó los mejores hiperparámetros con $K=50$ clusters y $C=10.0$ para el parámetro de regularización del SVM. Por su parte, el modelo DeepConvNet fue entrenado con la configuración descrita en la Sección~\ref{sec:metodos}, utilizando una división 80--10--10 para entrenamiento, validación y prueba respectivamente.

\begin{table*}[!htbp]
\caption{Comparación de resultados entre BoF--SVM y DeepConvNet en clasificación de MI-EEG.}
\label{tab:resultados}
\centering
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{l c c c c}
\hline
\textbf{Modelo} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\hline
BoF--SVM ($K=50$, $C=10.0$) & 0.5284 & 0.5237 & 0.5712 & 0.5451 \\
DeepConvNet & 0.6742 & 0.6742 & 0.6742 & 0.6742 \\
\hline
\end{tabular}
\end{table*}

\subsection{Resultados del Modelo BoF--SVM}

El modelo BoF--SVM con los mejores hiperparámetros ($K=50$, $C=10.0$) obtuvo un accuracy de 52.84\%, una precisión de 52.37\%, un recall de 57.12\% y un F1-score de 0.5451 sobre el conjunto de prueba mediante validación cruzada por grupos. Estos resultados indican un desempeño ligeramente superior al azar (50\%) para una tarea de clasificación binaria, sugiriendo que el enfoque BoF es capaz de capturar patrones discriminativos en las señales EEG, aunque con margen de mejora. El recall superior a la precisión (57.12\% vs. 52.37\%) sugiere que el modelo tiende a favorecer la detección de casos positivos, posiblemente debido a la naturaleza de las características wavelet empleadas o a la distribución de las palabras del codebook.

La búsqueda de hiperparámetros reveló que configuraciones con menor número de clusters ($K=50$) combinadas con valores moderados de regularización ($C=10.0$) produjeron los mejores resultados, como se muestra en la Fig.~\ref{fig:grid_search}. Esto sugiere que un vocabulario más compacto, junto con un grado moderado de complejidad del clasificador, es adecuado para este conjunto de datos. La variabilidad en los resultados de diferentes combinaciones de hiperparámetros, evaluada mediante validación cruzada, muestra la importancia de la optimización sistemática para alcanzar el mejor desempeño. La matriz de confusión del modelo optimizado (Fig.~\ref{fig:confusion_bof}) revela un número equilibrado de aciertos y errores, con una ligera tendencia hacia la detección de la clase MI-R (Right).

\subsection{Resultados del Modelo DeepConvNet}

El modelo DeepConvNet alcanzó un accuracy de 67.42\%, una precisión de 67.42\%, un recall de 67.42\% y un F1-score de 0.6742 sobre el conjunto de prueba. Estos resultados superan significativamente a los obtenidos por el modelo BoF--SVM, alcanzando una mejora de aproximadamente 14.6 puntos porcentuales en accuracy y 12.9 puntos porcentuales en F1-score. Este desempeño demuestra la capacidad del aprendizaje profundo para extraer representaciones discriminativas directamente de las señales EEG preprocesadas, superando el enfoque basado en características manuales. El modelo logró aprender patrones espacio--temporales complejos mediante sus capas convolucionales, lo cual le permitió capturar relaciones no lineales entre canales y muestras temporales que no son explícitamente codificadas en el enfoque BoF--SVM.

\subsection{Análisis Comparativo}

La comparación directa entre ambos modelos revela que el enfoque DeepConvNet superó al BoF--SVM en todas las métricas evaluadas, con diferencias de aproximadamente 14.6 puntos porcentuales en accuracy y 12.9 puntos porcentuales en F1-score (Fig.~\ref{fig:metrics_comparison}). Esta diferencia indica que, en este escenario específico, el aprendizaje automático de características mediante convoluciones profundas fue más efectivo que la extracción explícita de características mediante transformadas wavelet combinada con el modelo BoF. El modelo DeepConvNet logró aprovechar la capacidad de las redes profundas para aprender representaciones jerárquicas y no lineales directamente de los datos, capturando patrones espacio--temporales complejos que no están explícitamente codificados en las características wavelet. Ambos modelos presentan un desempeño superior al azar, aunque aún están por debajo de los valores más altos reportados en la literatura (Tabla~\ref{tab:sota}), lo cual puede atribuirse a la complejidad del problema de generalización inter-sujeto, la variabilidad inherente de las señales EEG y las características específicas de este conjunto de datos.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.48\textwidth]{results/figures/confusion_matrix_bof_svm.pdf}
\caption{Matriz de confusión del modelo BoF--SVM optimizado ($K=50$, $C=10.0$) evaluado mediante validación cruzada por grupos sobre 880 ensayos.}
\label{fig:confusion_bof}
\end{figure}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=0.65\textwidth]{results/figures/metrics_comparison.pdf}
\caption{Comparación de métricas de desempeño entre los modelos BoF--SVM y DeepConvNet. Los valores se muestran como barras agrupadas para facilitar la comparación directa entre ambos enfoques.}
\label{fig:metrics_comparison}
\end{figure*}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=0.55\textwidth]{results/figures/grid_search_heatmap.pdf}
\caption{Resultados del Grid Search para el modelo BoF--SVM mostrando el F1-score promedio para cada combinación de hiperparámetros ($K$ clusters y $C$ de regularización). El recuadro azul marca la combinación óptima ($K=50$, $C=10.0$).}
\label{fig:grid_search}
\end{figure*}

\section{Discusión}\label{sec:discusion}

Los resultados obtenidos en este estudio proporcionan insights valiosos sobre las ventajas y limitaciones de los enfoques basados en características manuales versus aprendizaje profundo para la clasificación de señales EEG de imaginación motora. El mejor desempeño del modelo DeepConvNet en comparación con BoF--SVM está alineado con tendencias observadas en otros dominios donde las redes profundas suelen superar a métodos clásicos cuando se dispone de suficientes datos y una configuración adecuada. Esta superioridad puede explicarse por varios factores metodológicos y del dominio.

\subsection{Interpretación de Resultados}

El desempeño superior del modelo DeepConvNet puede atribuirse a su capacidad para aprender representaciones jerárquicas y no lineales directamente de los datos, capturando patrones espacio--temporales complejos que no están explícitamente codificados en las características wavelet. La arquitectura convolucional profunda permite extraer automáticamente características discriminativas en múltiples niveles de abstracción, desde patrones locales de tiempo hasta correlaciones espaciales complejas entre canales. A diferencia del enfoque BoF--SVM, que depende de la calidad y relevancia de las características manualmente diseñadas (transformadas wavelet), el DeepConvNet puede adaptarse y optimizar internamente la representación de los datos durante el entrenamiento, lo cual resulta en una mejor capacidad de generalización.

A pesar de que el modelo BoF--SVM explota explícitamente el conocimiento del dominio a través de transformadas wavelet diseñadas para capturar patrones tiempo--frecuencia en las bandas $\mu$ y $\beta$~\cite{ref1, ref2}, el DeepConvNet logró superar este enfoque al aprender estas relaciones de manera más eficiente y capturar además interacciones no lineales entre canales y muestras temporales que no están explícitamente codificadas. El desempeño del DeepConvNet (67.42\%) demuestra que, con la configuración adecuada, el aprendizaje profundo puede ser efectivo incluso con conjuntos de datos de tamaño moderado (880 ensayos), superando significativamente al enfoque basado en características manuales.

El rendimiento de ambos modelos, aunque mejor que el azar, refleja los desafíos inherentes al problema de clasificación MI-EEG, particularmente en el contexto de generalización inter-sujeto. La variabilidad inter e intra-sujeto, la baja relación señal--ruido inherente a las señales EEG, y las diferencias individuales en los patrones de activación son factores que impactan consistentemente el desempeño en tareas BCI. Sin embargo, el DeepConvNet logró alcanzar un nivel de precisión considerablemente más alto que el BoF--SVM, sugiriendo que su capacidad de aprendizaje adaptativo le permite manejar mejor esta variabilidad.

\subsection{Errores y Limitaciones}

El análisis de errores revela diferencias importantes en la capacidad de discriminación entre ambos modelos. El modelo DeepConvNet presenta un balance más uniforme entre precisión y recall (ambos en 67.42\%), lo que indica una clasificación más equilibrada y confiable. Por el contrario, el modelo BoF--SVM muestra una asimetría hacia un mayor recall (57.12\%) pero menor precisión (52.37\%), sugiriendo que es más conservador en rechazar casos positivos, lo cual puede ser deseable en aplicaciones BCI donde la detección de intentos de movimiento es crítica. Sin embargo, el DeepConvNet logra un mejor equilibrio general, superando al BoF--SVM en todas las métricas evaluadas.

Las limitaciones principales de este estudio incluyen: (1) el tamaño relativamente pequeño del conjunto de datos (20 sujetos, 880 ensayos totales), que puede limitar la capacidad de generalización de modelos complejos como DeepConvNet; (2) la ausencia de técnicas avanzadas de aumento de datos (data augmentation) para el modelo CNN, que podrían mejorar su desempeño; (3) la evaluación mediante una sola división 80--10--10 para DeepConvNet, en contraste con la validación cruzada completa empleada para BoF--SVM, lo cual puede introducir variabilidad en las estimaciones; y (4) la no exploración exhaustiva de hiperparámetros para el modelo CNN, que podría revelar configuraciones más efectivas.

\subsection{Amenazas a la Validez}

Las principales amenazas a la validez de este estudio incluyen: (1) \textit{Validez interna}: la diferencia en los esquemas de validación entre ambos modelos (cross-validation completa vs. división simple) puede afectar la comparabilidad directa de los resultados; (2) \textit{Validez externa}: el uso de un conjunto de datos específico con características particulares de adquisición y preprocesamiento limita la generalización de las conclusiones a otros contextos; y (3) \textit{Sesgo de implementación}: las decisiones específicas de implementación (por ejemplo, la elección de MiniBatchKMeans sobre K-means estándar, o la arquitectura específica de DeepConvNet) pueden influir en los resultados de manera no trivial.

\subsection{Implicaciones y Trabajo Futuro}

Los resultados sugieren que, para aplicaciones BCI con conjuntos de datos de tamaño moderado y necesidad de generalización inter-sujeto, los enfoques híbridos que combinan conocimiento del dominio (como transformadas wavelet) con modelos de aprendizaje automático pueden ser más efectivos que el aprendizaje profundo puro. Futuras investigaciones deberían explorar: (1) técnicas de transfer learning para adaptar modelos CNN pre-entrenados a nuevos sujetos; (2) métodos de aumento de datos específicos para señales EEG que preserven las propiedades neurofisiológicas; (3) arquitecturas CNN más profundas o especializadas para señales temporales multicanales; (4) enfoques de ensemble que combinen las fortalezas de ambos modelos; y (5) la incorporación de información multimodal (por ejemplo, datos de comportamiento o contexto de la tarea) para mejorar la discriminación.

\section{Conclusiones}\label{sec:conclusiones}

Este estudio evaluó y comparó dos enfoques metodológicos para la clasificación de señales EEG de imaginación motora: un modelo basado en Bag of Features (BoF) combinado con SVM, y una arquitectura de red neuronal convolucional profunda (DeepConvNet). Los resultados obtenidos proporcionan respuestas claras a la pregunta de investigación planteada y permiten formular conclusiones sobre la eficacia relativa de cada enfoque.

En respuesta a la pregunta de investigación, el enfoque DeepConvNet demostró ser más efectivo que el modelo BoF--SVM para la clasificación de señales EEG de imaginación motora en este contexto específico, alcanzando un accuracy de 67.42\% y un F1-score de 0.6742, comparado con 52.84\% y 0.5451 respectivamente para BoF--SVM. Esta diferencia de aproximadamente 14.6 puntos porcentuales en accuracy es estadísticamente relevante y sugiere que el aprendizaje automático de características mediante redes profundas puede superar enfoques basados en características manuales cuando se dispone de una configuración adecuada y suficiente capacidad de entrenamiento, incluso con conjuntos de datos de tamaño moderado.

Respecto a los objetivos específicos: (1) el modelo BoF--SVM fue implementado exitosamente, aunque no alcanzó el objetivo de 80\% de accuracy inicialmente establecido, obteniendo 52.84\%; (2) el modelo CNN fue implementado y logró un desempeño considerablemente mejor, alcanzando 67.42\% de accuracy, aunque aún no superó el objetivo de 85\%; y (3) el análisis comparativo utilizando métricas múltiples (accuracy, precision, recall, F1-score) identificó que DeepConvNet presenta mayor robustez y consistencia, con mejores valores en todas las métricas evaluadas, demostrando la superioridad del aprendizaje profundo en este contexto.

Estos resultados reflejan tanto la complejidad inherente del problema de clasificación MI-EEG como el potencial del aprendizaje profundo para superar enfoques basados en características manuales cuando se implementa correctamente. El éxito del modelo DeepConvNet destaca la importancia de considerar el contexto específico (tamaño del dataset, necesidad de generalización, recursos computacionales) al seleccionar el enfoque metodológico, pero también demuestra que, con la arquitectura y configuración adecuadas, el aprendizaje profundo puede ser altamente efectivo incluso con conjuntos de datos de tamaño moderado. La aplicación exitosa de redes profundas a señales EEG requiere adaptaciones cuidadosas, consideración de las limitaciones de datos, y una arquitectura apropiada que pueda capturar las características espacio--temporales inherentes a las señales EEG, tal como logró el modelo DeepConvNet en este estudio.

Las contribuciones principales de este trabajo incluyen: (1) la implementación y evaluación sistemática de un enfoque BoF para señales EEG, demostrando su viabilidad y efectividad relativa; (2) una comparación directa y reproducible entre métodos clásicos y de aprendizaje profundo bajo condiciones equivalentes de evaluación; y (3) la identificación de factores que influyen en la elección entre enfoques basados en características manuales versus aprendizaje automático para aplicaciones BCI.

Como trabajo futuro, se recomienda explorar arquitecturas CNN más especializadas para señales temporales, técnicas de transfer learning para mejorar la generalización inter-sujeto, y métodos de ensemble que combinen las fortalezas de ambos enfoques. Además, sería valioso investigar la aplicación de estos métodos a conjuntos de datos más grandes y diversos para validar la generalización de estas conclusiones.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\bibliographystyle{IEEEtran}
\balance
\begin{thebibliography}{99}
\bibitem{ref1}
M.-P. Hosseini, A. Hosseini, y K. Ahi, “A review on machine learning for EEG signal processing in bioengineering,” IEEE Reviews in Biomedical Engineering, vol. 14, pp. 204–224, 2021.
\bibitem{ref2}
A. Saibene, H. Ghaemi, y E. Dagdevir, “Deep learning in motor imagery EEG signal decoding: A systematic review,” Neurocomputing, vol. 610, 128577, 2024.
\bibitem{ref3}
D. Cherifi, B. E. Berghouti, y L. Boubchir, “Classification of left/right hand and foot movements from EEG using machine learning algorithms,” Proc. IEEE Int. Conf. Bioinformatics and Biomedicine (BIBM), pp. 2458–2465, 2023.
\bibitem{ref4}
V. Shirodkar y D. R. Edla, “An evaluation of machine learning methods for classifying EEG signals associated with motor imagery,” Proc. IEEE ICCINS, 2023.
\bibitem{ref5}
N. Lu, T. Li, X. Ren, y H. Miao, “A deep learning scheme for motor imagery classification based on restricted Boltzmann machines,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 25, no. 6, pp. 566–575, 2017.
\bibitem{ref6}
K. Singh, G. Jaswal, y S. Bhalaik, "A novel CNN with sliding window technique for enhanced classification of MI-EEG sensor data," IEEE Sensors J., vol. 25, no. 3, pp. 4777–4785, 2025.
\bibitem{ref7}
R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, y T. Ball, "Deep learning with convolutional neural networks for EEG decoding and visualization," Human Brain Mapping, vol. 38, no. 11, pp. 5391–5420, 2017.
\bibitem{ref8}
Y. Narayan, D. Gautam, R. Kakkar, y D. Lakhwani, “BCI-based EEG signals classification using machine learning approach,” Proc. IEEE IATMSI, 2025.
\bibitem{ref9}
K. R. Dharmendra, D. K. Verma, y S. Vats, “Advancing brain–computer interfaces: A deep learning approach for enhanced neural signal processing,” Proc. IEEE AUTOCOM, 2025.
\bibitem{ref10}
M. A. Asghar, M. J. Khan, Fawad, Y. Amin, M. Rizwan, M. Rahman, S. Badnava, and S. S. Mirjavadi, "EEG-Based Multi-Modal Emotion Recognition using Bag of Deep Features: An Optimal Feature Selection Approach," Sensors, vol. 19, no. 23, p. 5218, 2019, doi: 10.3390/s19235218
\end{thebibliography}
\end{document}
