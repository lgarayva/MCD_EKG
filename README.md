
# Clasificación de ECG con Machine Learning: Un Análisis de Series Temporales y Feature Engineering

**Maestría en Ciencia de Datos**  
**Estancia de Investigación**

Autor:
- León Garay | [lgarayva](https://github.com/lgarayva)

Este repositorio contiene el código, los datos y la documentación para el proyecto **Clasificación de ECG con Machine Learning: Un Análisis de Series Temporales y Feature Engineering** desarrollado como parte de la estancia de investigación de la Maestría en Ciencia de Datos.

## Table of contents
- [Clasificación de ECG con Machine Learning: Un Análisis de Series Temporales y Feature Engineering](#clasificación-de-ecg-con-machine-learning-un-análisis-de-series-temporales-y-feature-engineering).
- [Table of contents](#table-of-contents).
- [Objetivo](#objetivo).
- [Creación de entorno virtual](#creación-de-entorno-virtual).
- [Estructura del repositorio](#estructura-del-repositorio).

## Objetivo

Este proyecto busca desarrollar un modelo de clasificación multiclase de ECG para diferenciar distintas patologías cardíacas, usando ingeniería de variables, análisis de series temporales y algoritmos de machine learning, evaluando su desempeño y utilidad clínica.

## Creación de entorno virtual

Para crear un entorno virtual con **Python 3.12.7**, ejecuta los siguientes comandos en la terminal:

```bash

python -m venv mcd_venv

conda activate mcd_env

source mcd_venv/bin/activate

python -m pip install --upgrade --force-reinstall pip

pip install -r requerimientos_MCD.txt

```

## Estructura del repositorio

- **config**: archivos de configuración y constantes.
- **Datos-Leonel-PTBXL**: datos brutos y procesados utilizados en el proyecto. 
- **docs**: documentos generados.
- **img**: imágenes utilizadas en el proyecto o generadas como resultados.
- **notebooks**: notebooks ejecutados, ordenados por número o etapa.
- **resultados**: reportes y resultados de los modelos y análisis finales.
- **src**: código fuente y funciones utilizadas a lo largo del desarrollo.

```
├── config
├── Datos-Leonel-PTBXL
│   ├── mi
│   ├── other
│   ├── sttc
│   └── sttc_mi
├── docs
├── img
│   ├── acf_pacf
│   ├── ccf
│   └── general
├── notebooks
├── output
│   ├── chunk_data
│   │   ├── chunk_100
│   │   │   ├── features
│   │   │   ├── pre_model
│   │   │   └── train_test_val
│   │   ├── chunk_5
│   │   │   └── pre_model
│   │   └── chunk_5 
│   │       └── pre_model
│   └── split_train_test
├── Resultados
└── src
```

