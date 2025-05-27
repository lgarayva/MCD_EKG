

# 1. Introducción

Planteamiento del problema, que se busca realizar. Oigen de los datos. Importancia del proyecto.

Un electrocardiograma (ECG) es un indicador no intrusivo que sirve como diagnóstico para identificar enfermedades cardiobasculares.

Las enfermedades cardiobasculares son la principal causa de muerte a nivel mundial, inlcuso en paises con ingresos altos solo por debajo del cancer. El electrocardiograma es una herramienta útil para evaluar el estado clínico cardíaco del paciente.

# 2. Datos

Los datos fueron obtenidos de PTB-XL, el cual es el mayor conjunto de datos clínicos de ECG disponible públicamente hasta la fecha. Estos datos fueron desarrollados con el objetivo de ser utiizados en métodos de aprendizaje de máquina con el fin de generar un sistema de decisiones automatizado para la interpretación de ECG. Este conjunto de datos fue desarollado con contrarrestar dos grandes obstáculos que se tenían con datos de ECG.

* El primer obstáculo que se tenía era que no existía un conjunto de datos para entrenamiento y validación público que puidera utilizarse para entrenar los modelos.
* El segundo obstáclulo fue la falta de procedimientos definidos para evaluar los algoritmos.

Los datos de *PTB-XL* fue registrada mediante dispositivos *Schiller AG* de octubre de 1989 a junio de 1966.

El conjunto de datos tiene las siguientes características:

* tiene un volumen de 21,837 registros de 12 señales, cada una de 10 segundos, provenientes de 18,885 pacientes.
* El conjunto de datos está balanceado respecto al género: 52 % hombres y 48 % mujeres. Presenta un amplio rango de edades, desde 0 hasta 95 años, con una mediana de 62 y un rango intercuartílico de 22.
* Los electrocardiogramas fueron validados por hasta 2 cardiologos, los registros incluyen información sobre ritmo, forma y diagnóstico del ECG.
* Los diagnósticos se clasificaron en formato de múltiples etiquetas, organizados en 5 superclases y 24 subclases.

Para los datos se realizó un proceso de adquisición de datos y procesamiento de datos.

### Adquisicón de datos

1. Las señales se recortaron en segmentos de 10 segundos y guardado en un formato comprimido de 400 Hz. Para todas las señales, se usó el estandar de las 12 caras (I, II, III, aVL, aVR, aVF, V1, V2, V3, V4, V5 y V6) con referencia al brazo derecho.
2. La información fue registrada en la base de datos por una enfermera.
3. Cada registro fue interpretado en un 67.13% de manera manual por un cardiólogo, 31.2% de manera automática por un dispositivo de ECG con validaciones posteriores por un cardiólogo y un 1.67% sin reporte inicial.
4. Finalmente, todos los reportes volvieron a ser anotados de manera manual por un experto basado principalmente en características cuantitativas de las señales.


### Procesamiento de datos

Las señalres fueron converidas del formato original a un formato finario con 16bits de preseición a una resolución de 1 \mu V /LSB. Las señalres pasaron por un proceso en el que se eliminaron picos de encendido y apagado en los dispositivos, estos picos se encontraban al inicio y final de los registros, además, las señales fueron resampleadas a una señal de 500 Hz, también se generó una versión de 100 Hz.

dónde se obtuvieron los datos, organización de datos. origen de datos. clases de datos, etc.

# 3 Análisis exploratorio y Procesamiento

Para el análisis exploratorio de datos exploramos distintas opciones. 

En primer lugar analizamos las autocorrelaciones, ACF, y autocorrelaciones parciales, PACF. Para este primer análisis, se revisó la ACF de las series originales y obteniendo el agregado de estas; también se analizó el agregado de las series y posteriormente se obtuvo la ACF y PACF. Para el análisis de la ACF y PACF se dividió el análisis por clase y por señal.

El segundo análisis que se realizó fue el análisis de identificación de raices unitarios por medio de la prueba de Dickey-Fuller, esto para identificar si las series presentaban una raíz unitario que podría afectar la estacionalidad de las series. Este anális fue de vital importancia ya que logramos identificar que la mayoría de las series eran no estacionarias lo que afectaba la identificación de la ACF, posterior a aplicar diferencia pudimos encontrar patrones más claros en esta.

El tercer análisis que realizamos fue la descomposición de la serie, para este análisis nos enfocamos específicamente en el componente estacional, esto con el objetivo de encontrar patrones y periodicidades. Una vez encontrado estos patrones obtuvimos un patrón de picos en la serie el que utilizamos posteriormente para suavizar la serie, con el objetivo de encontrar posibles patrones en las series eliminando ruido en esta.

El cuatro análisis que realizamos fue el análisis de ´cross-correlation´, en este análisis ya no nos centramos en las señales particulares, sino que analizamos combinaciones de señales, con el objetivo de encontrar relaciones entre pares. En este apartado analizamos las series originales y el promedio de las series.

El quinto análisis fue el promedio de las series con intervalos de desviación estandar. Este análisis se realizó tanto para la serie original como a la serie suavizada con los resultados obtenidos en el análisis tres.

El sexto análisis que se realizó fue similar al análisis cuatro, a diferencia que para este análisis utilizamos un suavizamiento de las series.

Un sétimo análisis que se realizó fue un análisis de cluster KMeans con las primeras autocorrelaciones y autocorrelaciones parciales, con el objetivo de identificar si esto era suficiente para separar las clases.


Análisis de datos que se realizó, análisis de ACF, PACF, CCF, seasonal, etc. Hallazgos de este análisis exploratorio y oportunidades de ingeniería de variables.

Manera en que se trabajaron los datos, alineado al análisis exploratorio, si se realizó limpieza, transformaciones ect.

# 4. Ingenieria de variables

Variables que se crearon, origenes de variables.

En este apartado se trabajó, con base en los resultados del análisis exploratorio de datos, la creación de variables para el modelo. Se trabajaron tres bloques de variables:

* Variables de acf y pacf.
* variables estadísticas de la serie y descomposición de la serie.
* Variables de ccf.


Para las variables de acf y pacf, con base en el análisis, se encontró que las primeras 5 autocorrelaciones en promedio eran distintas de cero, por lo que se utilizaron estas autocorrelaciones como variables para el modelo. En total se tenían 12 señales y cada señal se utilizó las primeras 5 acf y pacf dando en total ´12x5x2 = 120´ variables de este bloque.

Para las variables de estádisticas de la serie se tomó en cuenta variables como amplitud de la serie, intensidad de la serie, ratio de la serie, promedio en donde se dan los picos en la serie en el componente estacional, desviación estandar de la serie en el componente estacional y número de picos en la serie en el componente estacional.

Para las variables de la CCF se tomaron en cuenta: el número de cruces por cero; el promedio, máximo, mínimo y desviación estándar de la CCF; el lag correspondiente al máximo y al mínimo de la CCF; la curtosis de la CCF; la media recortada de la CCF; y la norma de la matriz de la CCF.

# 5 Modelo

Para el modelado se utilizaron distintas arquitecturas de modelos de aprendizaje de máquina. En el modelado realizamos 3 pruebas: 

* Modelado utilzando feature engeeniering
* Modelado utilizando los valores de las series dividido en partes la serie (chunks).
* Modelado utilizando feature engeeniering diviendo la serie en partes (chunks). 

Los modelos que se utilizaron en estas pruebas fueron los siguientes:

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier
* Naive Bayes
* XGB Classifier

Para estos modelos se realizó una búsqueda de hiperparámetros con técnicas de ´grid seacrh´ a excepción del modelo de XGB Classifier. 

Para la evaluación de modelos tomanos en consideración las métricas de accuracy, recall_weighted, f1_weighted y roc_auc_ovr. Con base en estas métricas se evaluó que modelo que desempeñó de mejor manera. Para los modelos que se dividieron en chunks adicional de estas métricas, tomamos en consideración la moda de las predicciones de los chunks y así obtuvimos las métricas de accuracy, precision_weighted, recall_weighted y f1_weighted.


Modelos que se realizaron, tiempos de ejecución de modelos, comparación de modelos.

# 6 Resultados

Resultados obtenidos en los modelos, en relación al punto 5 y resultados de análisis exploratorio con respecto al objetivo el proyecto.

# 7 Conclusiones
