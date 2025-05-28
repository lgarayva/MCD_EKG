

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

Es importante mencionar que en este apartado analizaremos únicamente la señal II y clase MI ,Infarto de Miocardio, ya que sirve computacionalmente para analizar infarto inferior.

## 3.1 Análisis de ACF y PACF

![ACF PACF II](img/acf_pacf/acf_pacf_II.png)

![ACF PACF II 30 lags](img/acf_pacf/acf_pacf_II_2.png)

De las gráficas de de autocorrelación, podemos notar que estas no decienden a cero, mientras que las gráficas de autocorrelación parciales tienden a decrecer a cero confirme aumentan los lags. De esto podemos pensar que existe no estacionariedad en la serie o inlcuso que existe una raíz unitaria. Que la PACF tienda a cero podría indicar que existe un componente AR en la serie, es decir, la serie podría analizarse con un modelo ARIMA(p,d,0).

Para analizar más a fondo, estudiaremos la existencia de raíces unitarias y de no estacionariedad aplicando diferencias a la serie y aplicando la prueba de Dickey-Fuller.

## Raíces unitarias

Realizando la prueba de Dickey-Fuller por señal y por clase obtenemos los siguientes resultados:

| Señal   |       MI |   STTC MI |     STTC |    OTHER |
|:--------|---------:|----------:|---------:|---------:|
| AVL     | 0.843333 |  0.88     | 0.818333 | 0.735    |
| V3      | 0.916667 |  0.916667 | 0.881667 | 0.941667 |
| V1      | 0.856667 |  0.906667 | 0.86     | 0.853333 |
| V2      | 0.923333 |  0.941667 | 0.891667 | 0.941667 |
| II      | 0.841667 |  0.861667 | 0.823333 | 0.85     |
| V4      | 0.861667 |  0.891667 | 0.805    | 0.911667 |
| V5      | 0.795    |  0.876667 | 0.815    | 0.883333 |
| V6      | 0.708333 |  0.836667 | 0.753333 | 0.82     |
| III     | 0.808333 |  0.873333 | 0.733333 | 0.681667 |
| AVR     | 0.888333 |  0.908333 | 0.878333 | 0.91     |
| AVF     | 0.785    |  0.84     | 0.733333 | 0.733333 |
| I       | 0.891667 |  0.9      | 0.888333 | 0.906667 |

de la tabla podemos ver que existe una parte de señales y clases que tienen una raíz unitaria, sin embargo no es la mayoría (menos del 50%), sin embargo las series no son estacionarios, por lo que para volver estacionaria la serie es necesario aplicar una diferencia.

| Señal   |       MI |   STTC MI |     STTC |   OTHER |
|:--------|---------:|----------:|---------:|--------:|
| AVL     | 0.996667 |         1 | 0.998333 |       1 |
| V3      | 1        |         1 | 1        |       1 |
| V1      | 1        |         1 | 1        |       1 |
| V2      | 1        |         1 | 1        |       1 |
| II      | 1        |         1 | 1        |       1 |
| V4      | 1        |         1 | 1        |       1 |
| V5      | 1        |         1 | 1        |       1 |
| V6      | 0.998333 |         1 | 1        |       1 |
| III     | 0.996667 |         1 | 0.998333 |       1 |
| AVR     | 1        |         1 | 1        |       1 |
| AVF     | 0.998333 |         1 | 1        |       1 |
| I       | 0.998333 |         1 | 1        |       1 |

Realizando la prueba para analizar si es necesario aplicar una diferencia adicional a la serie, vemos que menos del 1% de las series presenta raíz unitaria, por lo que únicamente aplicaremos una diferencia.

![ACF PACF II diff](img/acf_pacf/acf_pacf_II_diff_MI.png)

Aplicando una diferencia a la serie y obteniendo la ACF y PACF, podemos observar que únicamente las primeras 5 autocorrelaciones de la ACF son significativamente distintas de cero, mientras que la PACF conserva el comportamiento decreciente a cero, lo que nos lleva a pensar que nuestra suposición fue correcta de que las series eran no estacionarias y presentan un componente autorregresivo.

## Descomposición de la serie

En el análisis de descomposición de la serie, analisamos únicamente la parte estacional de la serie. Para este analisis utilizamos el promedio de la serie para obtener esta descomposición. Una vez obtenido el componente estacional del promedio de las series obtuvimos los saltos dentro de la serie tomando en cuenta un salto por encima de dos desviaciones estándar. Con los saltos obtenemos una estadística de cada cuantos periodos se dan estos y obteuvimos los siguientes resultados:


| Señal   |       MI |   STTC MI |     STTC |    OTHER |   promedio |      std |
|:--------|---------:|----------:|---------:|---------:|-----------:|---------:|
| AVL     |  48.2105 |   33.4483 | 100      |  31.7931 |    53.363  | 31.9553  |
| V3      | 100      |   48.8421 |  25.2051 |  32.5517 |    51.6497 | 33.7129  |
| V1      | 100      |  100      |  25.0256 |  50      |    68.7564 | 37.49    |
| V2      |  33.3793 |  100      |  50      |  50      |    58.3448 | 28.8542  |
| II      |  50.2105 |   47.7895 |  32.2069 |  48.3684 |    44.6438 |  8.3553  |
| V4      |  25.359  |   19.898  | 100      | 100      |    61.3142 | 44.7261  |
| V5      |  24.4872 |   24.0256 |  19.9184 |  31.8276 |    25.0647 |  4.95428 |
| V6      |  25.3846 |   24.3077 |  33.3793 | 100      |    45.7679 | 36.3805  |
| III     |  19.6939 |   50.3684 |  24.359  | 100      |    48.6053 | 36.8252  |
| AVR     |  24.4872 |   31.5517 |  32.2069 |  32.2759 |    30.1304 |  3.77628 |
| AVF     |  32.069  |   50.2632 |  32.9655 |  48.3684 |    40.9165 |  9.73633 |
| I       |  32.9655 |   31.5517 |  33.6897 |  34.2143 |    33.1053 |  1.15534 |


Teniendo en promedio los picos cada $46.805166$ en todas las señales.

![ACF PACF II diff](img/acf_pacf/seasonal_trend_II.png)

De las gráficas de componente estacional, podemos observar que para las señales MI presenta menos variabilidad o desviación, y vemos que existen tendencias de picos más marcadas.

## Análisis de correlación cruzada

## Análisis de serie con intervalos de desviación estándar

## Análisis de serie con intervalos de desviación estándar con suavizamiento de series

## Análisis KMeans

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
