
# 

## Análisis de autocorrelaciones y autocorrelaciones parciales

### Análisis de autocorrelaciones y autocorrelaciones parciales series originales

En esta sección se analizó la autocorrelación y la autocorrelación porcial. Del análisis de analizó el promedio de las autocorrelaciones y las autocorrelaciones parciales, esto con el objetivo de analizar si existía algún patrón en las distintas autocorrelaciones y autocorrelaciones parciales, con el objetivo de identificar algún retrazo significativo. Además, se analizó el promedio de las series y cómo se compartaron la autocorrelación y la autocorrelación parcial.

Cada conjunto de gráficas muestran los siguientes gráficos:

* Agregado ACF mean's: autocorrelograma de los promedios de las autocorrelaciones de las series.
* Agregado PACF mean'sautocorrelograma de los promedios de las autocorrelaciones parciales de las series.
* PACF mean's: autocorrelograma de las autocorrelaciones de los promedios de las series.
* ACF mean'sautocorrelograma de las autocorrelaciones parciales de los promedios de las series.

[Ir a Análisis autocorrelaciones y autocorrelaciones parcial](../notebooks/01_analisis_autocorrelaciones.ipynb#análisis-autocorrelaciones-y-autocorrelaciones-parcial)

De la primera serie de gráficas podemos observar que en general las autocorrelaciones no decrecen a cero, mientras que las autocorrelaciones parciales tienen una tendencia decreciente a cero, lo que nos podría indicar la existe de raíces unitarios o que la serie es no estacionaria.

### Pruebas raíces unitarias

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

De la prueba podemos ver que la mayoría de las series no presentan una raíz unitaria, sin embargo existe una presencia no estacionaria.

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

### Análisis de autocorrelaciones y autocorrelaciones parciales series diferenciadas

[Ir a Análisis autocorrelaciones y autocorrelaciones parcial con diferencias](../notebooks/01_analisis_autocorrelaciones.ipynb#análisis-autocorrelaciones-y-autocorrelaciones-parcial-con-diferencias)

En general, podemos obervar que las primeras `q` autocorrelaciones son distintas de cero, mientras que las autocorrelaciones parciales tienen una sucesión infinita a cero. Con esto podemos ver que al aplicar una diferencia se corrige el problema de no estacionaridad.

Otro punto a resaltar es que la autocorrelación del promedio de la serie parece tener algunas autocorrelaciones distintas de cero en retrazos futuros a los `q` primeros.


