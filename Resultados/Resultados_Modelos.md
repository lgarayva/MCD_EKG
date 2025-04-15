# Resultados modelos ML

## Primeros resultados
### Logistic Regression

#### Parámetros

| parámetro   | valor   |
|:------------|:--------|
| C           | 0.001   |
| penalty     | l2      |

#### Resumen modelo
|              |   precision |   recall |   f1-score |    support |
|:-------------|------------:|---------:|-----------:|-----------:|
| 0            |    0.487395 | 0.483333 |   0.485356 | 120        |
| 1            |    0.612613 | 0.566667 |   0.588745 | 120        |
| 2            |    0.475806 | 0.491667 |   0.483607 | 120        |
| 3            |    0.539683 | 0.566667 |   0.552846 | 120        |
| accuracy     |    0.527083 | 0.527083 |   0.527083 |   0.527083 |
| macro avg    |    0.528874 | 0.527083 |   0.527638 | 480        |
| weighted avg |    0.528874 | 0.527083 |   0.527638 | 480        |

#### Tabla resultados

| metric             |    value |
|:-------------------|---------:|
| accuracy           | 0.527083 |
| precision_weighted | 0.528874 |
| recall_weighted    | 0.527083 |
| f1_weighted        | 0.527638 |
| roc_auc_ovr        | 0.790475 |
| log_loss           | 1.06877  |
| gini_normalized    | 0.580949 |
| ks_test_clase_0    | 0.408333 |
| ks_test_clase_1    | 0.525    |
| ks_test_clase_2    | 0.388889 |
| ks_test_clase_3    | 0.494444 |

#### Matriz de confusión

|        |   Pred 0 |   Pred 1 |   Pred 2 |   Pred 3 |
|:-------|---------:|---------:|---------:|---------:|
| Real 0 |       58 |       20 |       17 |       25 |
| Real 1 |       20 |       68 |       25 |        7 |
| Real 2 |       18 |       17 |       59 |       26 |
| Real 3 |       23 |        6 |       23 |       68 |

### Random Forest

#### Parámetros

| parámetro         | valor   |
|:------------------|:--------|
| max_depth         | 10      |
| max_features      | sqrt    |
| min_samples_leaf  | 2       |
| min_samples_split | 2       |
| n_estimators      | 500     |

#### Resumen modelo

|              |   precision |   recall |   f1-score |    support |
|:-------------|------------:|---------:|-----------:|-----------:|
| 0            |    0.390244 | 0.4      |   0.395062 | 120        |
| 1            |    0.57377  | 0.583333 |   0.578512 | 120        |
| 2            |    0.445545 | 0.375    |   0.40724  | 120        |
| 3            |    0.514925 | 0.575    |   0.543307 | 120        |
| accuracy     |    0.483333 | 0.483333 |   0.483333 |   0.483333 |
| macro avg    |    0.481121 | 0.483333 |   0.48103  | 480        |
| weighted avg |    0.481121 | 0.483333 |   0.48103  | 480        |

#### Tabla resultados

|    | metric             |    value |
|---:|:-------------------|---------:|
|  0 | accuracy           | 0.483333 |
|  1 | precision_weighted | 0.481121 |
|  2 | recall_weighted    | 0.483333 |
|  3 | f1_weighted        | 0.48103  |
|  4 | roc_auc_ovr        | 0.751476 |
|  5 | log_loss           | 1.24126  |
|  6 | gini_normalized    | 0.502951 |
|  7 | ks_test_clase_0    | 0.302778 |
|  8 | ks_test_clase_1    | 0.522222 |
|  9 | ks_test_clase_2    | 0.327778 |
| 10 | ks_test_clase_3    | 0.45     |

#### Matriz de confusión

|        |   Pred 0 |   Pred 1 |   Pred 2 |   Pred 3 |
|:-------|---------:|---------:|---------:|---------:|
| Real 0 |       48 |       22 |       20 |       30 |
| Real 1 |       23 |       70 |       21 |        6 |
| Real 2 |       25 |       21 |       45 |       29 |
| Real 3 |       27 |        9 |       15 |       69 |

### Gradient Boosting Classifier

#### Parámetros

| parámetro         | valor   |
|:------------------|:--------|
| learning_rate     | 0.01    |
| max_depth         | 20      |
| max_features      | sqrt    |
| min_samples_leaf  | 30      |
| min_samples_split | 2       |
| n_estimators      | 100     |
| subsample         | 1.0     |

#### Resumen modelo

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| 0            |    0.431818 | 0.475    |   0.452381 |     120   |
| 1            |    0.585586 | 0.541667 |   0.562771 |     120   |
| 2            |    0.45045  | 0.416667 |   0.4329   |     120   |
| 3            |    0.539683 | 0.566667 |   0.552846 |     120   |
| accuracy     |    0.5      | 0.5      |   0.5      |       0.5 |
| macro avg    |    0.501884 | 0.5      |   0.500224 |     480   |
| weighted avg |    0.501884 | 0.5      |   0.500224 |     480   |

#### Tabla resultados

|    | metric             |    value |
|---:|:-------------------|---------:|
|  0 | accuracy           | 0.5      |
|  1 | precision_weighted | 0.501884 |
|  2 | recall_weighted    | 0.5      |
|  3 | f1_weighted        | 0.500224 |
|  4 | roc_auc_ovr        | 0.751817 |
|  5 | log_loss           | 1.25756  |
|  6 | gini_normalized    | 0.503634 |
|  7 | ks_test_clase_0    | 0.35     |
|  8 | ks_test_clase_1    | 0.480556 |
|  9 | ks_test_clase_2    | 0.344444 |
| 10 | ks_test_clase_3    | 0.480556 |


#### Matriz de confusión

|        |   Pred 0 |   Pred 1 |   Pred 2 |   Pred 3 |
|:-------|---------:|---------:|---------:|---------:|
| Real 0 |       57 |       20 |       20 |       23 |
| Real 1 |       26 |       65 |       24 |        5 |
| Real 2 |       21 |       19 |       50 |       30 |
| Real 3 |       28 |        7 |       17 |       68 |

### Naive Bayes

#### Parámetros

| parámetro     |   valor |
|:--------------|--------:|
| var_smoothing |   1e-09 |

#### Resumen modelo

|              |   precision |   recall |   f1-score |    support |
|:-------------|------------:|---------:|-----------:|-----------:|
| 0            |    0.338129 | 0.391667 |   0.362934 | 120        |
| 1            |    0.315412 | 0.733333 |   0.441103 | 120        |
| 2            |    0.363636 | 0.1      |   0.156863 | 120        |
| 3            |    0.482759 | 0.116667 |   0.187919 | 120        |
| accuracy     |    0.335417 | 0.335417 |   0.335417 |   0.335417 |
| macro avg    |    0.374984 | 0.335417 |   0.287205 | 480        |
| weighted avg |    0.374984 | 0.335417 |   0.287205 | 480        |

#### Tabla resultados

|    | metric             |    value |
|---:|:-------------------|---------:|
|  0 | accuracy           | 0.335417 |
|  1 | precision_weighted | 0.374984 |
|  2 | recall_weighted    | 0.335417 |
|  3 | f1_weighted        | 0.287205 |
|  4 | roc_auc_ovr        | 0.616916 |
|  5 | log_loss           | 6.40477  |
|  6 | gini_normalized    | 0.233831 |
|  7 | ks_test_clase_0    | 0.186111 |
|  8 | ks_test_clase_1    | 0.266667 |
|  9 | ks_test_clase_2    | 0.163889 |
| 10 | ks_test_clase_3    | 0.216667 |

#### Matriz de confusión

|        |   Pred 0 |   Pred 1 |   Pred 2 |   Pred 3 |
|:-------|---------:|---------:|---------:|---------:|
| Real 0 |       47 |       65 |        5 |        3 |
| Real 1 |       25 |       88 |        6 |        1 |
| Real 2 |       32 |       65 |       12 |       11 |
| Real 3 |       35 |       61 |       10 |       14 |

### XGB Classifier

#### Parámetros

| parámetro          | valor          |
|:-------------------|:---------------|
| objective          | multi:softprob |
| enable_categorical | False          |
| eval_metric        | merror         |
| learning_rate      | 0.1            |
| missing            | nan            |
| n_estimators       | 5000           |

#### Resumen modelo

|              |   precision |   recall |   f1-score |    support |
|:-------------|------------:|---------:|-----------:|-----------:|
| 0            |    0.445378 | 0.441667 |   0.443515 | 120        |
| 1            |    0.603604 | 0.558333 |   0.580087 | 120        |
| 2            |    0.48     | 0.5      |   0.489796 | 120        |
| 3            |    0.536    | 0.558333 |   0.546939 | 120        |
| accuracy     |    0.514583 | 0.514583 |   0.514583 |   0.514583 |
| macro avg    |    0.516245 | 0.514583 |   0.515084 | 480        |
| weighted avg |    0.516245 | 0.514583 |   0.515084 | 480        |

#### Tabla resultados

|    | metric             |    value |
|---:|:-------------------|---------:|
|  0 | accuracy           | 0.514583 |
|  1 | precision_weighted | 0.516245 |
|  2 | recall_weighted    | 0.514583 |
|  3 | f1_weighted        | 0.515084 |
|  4 | roc_auc_ovr        | 0.762303 |
|  5 | log_loss           | 1.4775   |
|  6 | gini_normalized    | 0.524606 |
|  7 | ks_test_clase_0    | 0.333333 |
|  8 | ks_test_clase_1    | 0.494444 |
|  9 | ks_test_clase_2    | 0.380556 |
| 10 | ks_test_clase_3    | 0.505556 |

#### Matriz de confusión

|        |   Pred 0 |   Pred 1 |   Pred 2 |   Pred 3 |
|:-------|---------:|---------:|---------:|---------:|
| Real 0 |       53 |       22 |       14 |       31 |
| Real 1 |       22 |       67 |       26 |        5 |
| Real 2 |       22 |       16 |       60 |       22 |
| Real 3 |       22 |        6 |       25 |       67 |

#### Resumen modelo

#### Tabla resultados

#### Matriz de confusión