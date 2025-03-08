
## 01

### Análisis de autocorrelaciones y autocorrelaciones parciales

#### Análisis de autocorrelaciones y autocorrelaciones parciales series originales

En esta sección se analizó la autocorrelación y la autocorrelación parcial. En el análisis se analizó el promedio de las autocorrelaciones y las autocorrelaciones parciales, esto con el objetivo de analizar si existía algún patrón en las distintas autocorrelaciones y autocorrelaciones parciales, con el objetivo de identificar algún retraso significativo. Además, se analizó el promedio de las series y cómo se compararon la autocorrelación y la autocorrelación parcial.

Cada conjunto de gráficas muestran los siguientes gráficos:

* Agregado ACF means: autocorrelograma de los promedios de las autocorrelaciones de las series.
* Agregado PACF means: autocorrelograma de los promedios de las autocorrelaciones parciales de las series.
* PACF means: autocorrelograma de las autocorrelaciones de los promedios de las series.
* ACF means: sautocorrelograma de las autocorrelaciones parciales de los promedios de las series.

[Ir a Análisis autocorrelaciones y autocorrelaciones parciales](../notebooks/01_analisis_autocorrelaciones.ipynb#3d1)

De la primera serie de gráficas podemos observar que en general las autocorrelaciones no decrecen a cero, mientras que las autocorrelaciones parciales tienen una tendencia decreciente a cero, lo que nos podría indicar la existe de raíces unitarios o que la serie es no estacionaria.

#### Pruebas raíces unitarias

#### Resultados serie original

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

De la prueba podemos ver que la mayoría de las series no presentan una raíz unitaria, sin embargo existe una señal de no estacionariedad.

#### Resultados posterior a aplicar una diferencia

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

De la prueba aplicando una diferencia podemos ver que más del 99% de las series no presentan raíz unitaria, por lo que no tendríamos evidencia para aplicar una segunda diferencia.

#### Análisis de autocorrelaciones y autocorrelaciones parciales series diferenciadas

[Ir a Análisis autocorrelaciones y autocorrelaciones parciales con diferencias](../notebooks/01_analisis_autocorrelaciones.ipynb#3d3)

En general, podemos observar que las primeras `q` autocorrelaciones son distintas de cero, mientras que las autocorrelaciones parciales tienen una sucesión infinita a cero. Con esto podemos ver que al aplicar una diferencia se corrige el problema de no estacionariedad.

### Análisis de descomposición de series

Para el análisis de descomposición de series se tomó en cuenta únicamente el componente estacional, con el objetivo de encontrar tendencias de estación. Para esta evaluación se tomó un período como el conjunto de `100` observaciones. 

Para cada clase y cada señal tomamos el promedio de todos los pacientes, una vez que obtuvimos esto descomponemos la serie y se extrajo el componente estacional. 

A partir del componente estacional obtenemos los máximos y mínimos con el objetivo de encontrar picos que se repitan en la serie para poder realizar un suavizamiento en la serie.


|    | Señal   |       MI |   STTC MI |     STTC |    OTHER |   promedio |      std |
|---:|:--------|---------:|----------:|---------:|---------:|-----------:|---------:|
|  0 | AVL     |  48.2105 |   33.4483 | 100      |  31.7931 |    53.363  | 31.9553  |
|  1 | V3      | 100      |   48.8421 |  25.2051 |  32.5517 |    51.6497 | 33.7129  |
|  2 | V1      | 100      |  100      |  25.0256 |  50      |    68.7564 | 37.49    |
|  3 | V2      |  33.3793 |  100      |  50      |  50      |    58.3448 | 28.8542  |
|  4 | II      |  50.2105 |   47.7895 |  32.2069 |  48.3684 |    44.6438 |  8.3553  |
|  5 | V4      |  25.359  |   19.898  | 100      | 100      |    61.3142 | 44.7261  |
|  6 | V5      |  24.4872 |   24.0256 |  19.9184 |  31.8276 |    25.0647 |  4.95428 |
|  7 | V6      |  25.3846 |   24.3077 |  33.3793 | 100      |    45.7679 | 36.3805  |
|  8 | III     |  19.6939 |   50.3684 |  24.359  | 100      |    48.6053 | 36.8252  |
|  9 | AVR     |  24.4872 |   31.5517 |  32.2069 |  32.2759 |    30.1304 |  3.77628 |
| 10 | AVF     |  32.069  |   50.2632 |  32.9655 |  48.3684 |    40.9165 |  9.73633 |
| 11 | I       |  32.9655 |   31.5517 |  33.6897 |  34.2143 |    33.1053 |  1.15534 |

De la tabla podemos observar que hay una notable diferencia en los saltos del componente estacional. Con este análisis, una de las posibles soluciones es tomar una sola ventana de suavizamiento por señal o incluso una ventana de suavizamiento por señal y por clase.

## 02

### Análisis Cross Correlation

[Ir a Análisis Cross Correlation](../notebooks/02_analisis_cc.ipynb#3)

De las gráficas originales podemos notar que existe un comportamiento notable en los retrazos `[-75, 0, 75]`, además de que en algunas señales podemos observar una acumulación en picos en el rango `(-75,75)`.

#### Análisis de autocorrelaciones y autocorrelaciones parciales series originales