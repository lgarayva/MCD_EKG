

# 1. Introducción

Planteamiento del problema, que se busca realizar. Oigen de los datos. Importancia del proyecto.

# 2. Datos

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

# 5 Modelo

Modelos que se realizaron, tiempos de ejecución de modelos, comparación de modelos.

# 6 Resultados

Resultados obtenidos en los modelos, en relación al punto 5 y resultados de análisis exploratorio con respecto al objetivo el proyecto.

# 7 Conclusiones
