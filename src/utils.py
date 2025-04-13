import os
import pandas as pd
import numpy as np
from typing import Union

from scipy.signal import find_peaks
from scipy.stats import ks_2samp, pearsonr, spearmanr, kurtosis, trim_mean



from statsmodels.tsa.stattools import acf, pacf, adfuller, coint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import math

def lectura_carpetas_dict(data_path : str) -> dict:
    """
    La función tiene como objetivo, dada una ruta específica, obtener las subcarpetas
    y leer los archivos CSV de estas subcarpetas.

    Args:
        data_path (str): Ruta del directorio que contiene las subcarpetas con archivos CSV.

    Returns:
        dict: Diccionario con las carpetas como claves y, dentro de cada una, otro diccionario
              con los archivos CSV y sus correspondientes DataFrames.
    """

    carpetas = [entrada.name for entrada in os.scandir(data_path) if entrada.is_dir()]
    dict_archivos = {}

    for folder in carpetas:
        lista_archivos = [f for f in os.scandir(data_path + "/" + folder) if f.is_file() and f.name.endswith(".csv")]
        dict_archivos[folder] = lista_archivos

    df_folder_dict = {}
 
    for j in dict_archivos.keys():
        df_dicts = {}

        for file_path in dict_archivos[j]:
            df = pd.read_csv(file_path.path)
            patient_id = file_path.name.replace(".csv", "")
            df["patient_id"] = patient_id
            df["label"] = j

            df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
            df_dicts[patient_id] = df
            
        df_folder_dict[j] = df_dicts.copy()

    return df_folder_dict

def get_acf_pacf(dict_series : dict, label : str = "II", **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    La función tiene como objetivo dado un diccionario de series obtener la autocorrelación y la
    autocorrelación parcial de la serie dado una clase.

    Args:
        dict_series (dict): Diccionario con las distintas series a analizar.
        label (str): Clase que se va a analizar de la serie.

    Returns:
        tuple: La salida de la función es una tupla conformada de dos pd.DataFrame que contiene los 
               valores de la autocorrelación y autocorrelación parcial de cada serie.
    """

    acf_dict = {}
    pacf_dict = {}

    for pacient in dict_series.keys():
        pd_series = dict_series[pacient][label]
        pd_serie_acf, pd_serie_pacf = acf_pacf(pd_series, **kwargs)
        acf_dict[pacient] = pd_serie_acf
        pacf_dict[pacient] = pd_serie_pacf

    acf_df = pd.DataFrame(acf_dict)
    pacf_df = pd.DataFrame(pacf_dict)

    return acf_df, pacf_df

def get_df_label(dict_series : dict, label : str = "II", **kwargs) -> pd.DataFrame:
    """
    La función tiene como objetivo dado un diccionario de series generar un dataframe con una clase dada.

    Args:
        dict_series (dict): Diccionario con las distintas series a analizar.
        label (str): Clase que se va a analizar de la serie.

    Returns:
        pd.DataFrame: La salida de la función es un pd.DataFrame que contiene los 
                      valores de cada serie.
    """

    df_dict = {}

    for pacient in dict_series.keys():
        pd_series = dict_series[pacient][label]
        df_dict[pacient] = pd_series

    df = pd.DataFrame(df_dict)
    
    return df

def get_dict_labels(dict_series : dict, signals : list, **kwargs) -> dict:
    """
    Genera un diccionario de DataFrames a partir de un diccionario de series y una lista de señales.

    Args:
        dict_series (dict): Un diccionario donde las claves son identificadores de pacientes 
                            y los valores son diccionarios con etiquetas de señales.
        signals (list): Una lista de etiquetas de señales que se desean extraer.

    Returns:
        dict: Un diccionario donde las claves son las etiquetas de señales y los valores son 
              DataFrames con los datos correspondientes a cada paciente.
    """

    df_dict = {label: pd.DataFrame({
        pacient : dict_series[pacient][label]
        for pacient in dict_series.keys()
        })
    for label in signals
        }
    
    return df_dict

def plot_scale_gray(df):
    """
    Grafica un DataFrame en escala de grises.

    Args:
        df (DataFrame): El DataFrame que se desea graficar.

    Returns:
        None
    """

    df.plot(color="gray", alpha=0.1, legend=False)
    plt.show()

def get_estadisticas(df : pd.DataFrame, **kargs) -> pd.DataFrame:
    """
    Dado un DataFrame, la función entrega un DataFrame con las estadísticas por fila.

    Args:
        df (pd.DataFrame): DataFrame con las series a analizar.

    Returns:
        pd.DataFrame: DataFrame con las estadísticas de las series por fila.
    """

    pd_mean = df.mean(axis=1)
    pd_min = df.min(axis=1)
    pd_max = df.max(axis=1)
    pd_std = df.std(axis=1)

    df_estadisticas = pd.DataFrame({"mean" : pd_mean, "std": pd_std, "min" : pd_min, "max" : pd_max})

    return df_estadisticas

def plot_acf_pact_df(df : pd.DataFrame, metric : str = "mean" ,label : str = "ACP", N : int = 1000):
    """
    Grafica la función de autocorrelación (ACF) de un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos a graficar.
        metric (str): La métrica a utilizar para calcular la autocorrelación. Por defecto es "mean".
        label (str): La etiqueta para el eje y de la gráfica. Por defecto es "ACP".
        N (int): El tamaño de la muestra para calcular el intervalo de confianza. Por defecto es 1000.

    Returns:
        None
    """

    lags = np.arange(df.shape[0])
    autocorr_values = df[metric]
    intervalo_confianza = 1.96/math.sqrt(N)
    plt.stem(lags, autocorr_values)
    plt.axhline(intervalo_confianza, color='red', linestyle='--')
    plt.axhline(-1*intervalo_confianza, color='red', linestyle='--')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Lag")
    plt.ylabel(label)
    plt.show()

def plot_acf_residuals(series : pd.Series):
    """
    Grafica la función de autocorrelación (ACF) de los residuos de una serie temporal.

    Args:
        series (pd.Series): La serie temporal que contiene los residuos a graficar.

    Returns:
        None
    """

    label = "ACF Residuals"
    N = series.shape[0]
    lags = np.arange(N)
    intervalo_confianza = 1.96/math.sqrt(N)
    plt.stem(lags, series)
    plt.axhline(intervalo_confianza, color='red', linestyle='--')
    plt.axhline(-1*intervalo_confianza, color='red', linestyle='--')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Lag")
    plt.ylabel(label)
    plt.show()

def CCF_lags(x : Union[pd.Series, np.ndarray]| np.ndarray, y : Union[pd.Series, np.ndarray]| np.ndarray, max_lag : int =20):
    """
    Calcula la función de correlación cruzada (CCF) entre dos series temporales para una cantidad 
    máxima de desfases (lags).

    Args:
        x (Union[pd.Series, np.ndarray]o np.ndarray): La primera serie temporal.
        y (Union[pd.Series, np.ndarray]o np.ndarray): La segunda serie temporal.
        max_lag (int): El número máximo de desfases (lags) a calcular. Por defecto es 20.

    Returns:
        pd.DataFrame: Un DataFrame con los valores de la CCF y los desfases correspondientes.
    """

    n = np.max([x.shape[0], y.shape[0]])
    lags = np.arange(-max_lag, max_lag + 1)
    ccf_values = [np.corrcoef(x[max(0, -lag):n - max(0, lag)], 
                              y[max(0, lag):n - max(0, -lag)])[0, 1] for lag in lags]
    # return ccf_values, lags
    return pd.DataFrame({'ccf' : ccf_values, 'lags' : lags})

def plot_CCF(ccf_values, lags, figsize = (10,10)):
    """
    Grafica la función de correlación cruzada (CCF).

    Args:
        ccf_values (array-like): Los valores de la función de correlación cruzada.
        lags (array-like): Los desfases (lags) correspondientes a los valores de la CCF.
        figsize (tuple): El tamaño de la figura de la gráfica. Por defecto es (10, 10).

    Returns:
        None
    """

    # Graficar función de correlación cruzada
    plt.figure(figsize=figsize)
    plt.stem(lags, ccf_values)
    plt.axhline(0, color='black', linestyle="--")
    plt.xlabel("Lag")
    plt.ylabel("Correlación Cruzada")
    plt.title("Cross-Correlation Function (CCF)")
    plt.show()

def diff_ts(series : pd.Series, differences : int = 1) -> pd.Series:
    """
    La función recibe como argumento una serie de pandas y el número de diferencias que se le aplicarán 
    a la serie. La función regresa la serie diferenciada n veces.

    Args:
        series (pd.Series): Recibe una pandas serie.
        differences (int, opcional): Número de diferencias que se le aplicará a la serie. 
                                     Por defecto es 1.

    Returns:
        pd.Series: Regresa la serie diferenciada.
    """

    diff_series = series
    for i in range(differences):
        diff_series = diff_series.diff().dropna()
    
    return diff_series

def plot_acf_pacf_serie(series : pd.Series, lags : int = None, **kwargs):
    """
    Grafica la función de autocorrelación (ACF) y la función de autocorrelación parcial (PACF) de 
    una serie temporal.

    Args:
        series (pd.Series): La serie temporal a graficar.
        lags (int, opcional): El número de desfases (lags) a calcular. Si no se especifica, se calcula 
                              como 10 * log(n), donde n es el tamaño de la serie.
        **kwargs: Argumentos adicionales opcionales.

    Returns:
        None
    """

    if lags == None:
        lags = 10*int(math.log((series.shape[0])))

    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (8,8))
    _ = plot_acf(series, lags=lags, zero = False, ax = ax1)
    _ = plot_pacf(series, lags=lags, zero = False, ax = ax2)

def acf_pacf(series : pd.Series, nlags : int = None, **kwargs) -> tuple[pd.Series, pd.Series]:
    """
    Calcula la función de autocorrelación (ACF) y la función de autocorrelación parcial (PACF) de 
    una serie temporal.

    Args:
        series (pd.Series): La serie temporal para la cual se calcularán las funciones.
        nlags (int, opcional): El número de desfases (lags) a calcular. Si no se especifica, se 
                               calcula como 10 * log(n), donde n es el tamaño de la serie.
        **kwargs: Argumentos adicionales opcionales.

    Returns:
        tuple: Una tupla que contiene dos arrays, uno con los valores de la ACF y otro con los 
               valores de la PACF.
    """

    if nlags == None:
        nlags = 10*int(math.log((series.shape[0])))

    acf_vals = acf(series, nlags=nlags)
    pacf_vals = pacf(series, nlags=nlags)

    return acf_vals, pacf_vals

def fit_arima(series: pd.Series, order : tuple[int, int, int], **kwargs) -> pd.Series:
    """
    Obtiene los residuales después de evaluar un modelo ARIMA.

    Args:
        series (pd.Series): Serie a ajustar modelo ARIMA.
        order (tuple[int, int, int]): Orden del modelo ARIMA (p, d, q).
            p (int): Grado autorregresivo.
            d (int): Número de diferencias.
            q (int): Grado de promedio móviles.

    Returns:
        pd.Series: Residuos posterior a ajustar el modelo.
    """

    model = ARIMA(series, order=order)
    result = model.fit()
    return result

def smooth_serie(serie: pd.Series, do_abs: bool = True, window_size: int = 50, metodo: str = 'mean') -> pd.Series:
    """
    La función suaviza la serie aplicando primero valor absoluto en caso de no definir lo contrario 
    y aplica un método de rolling window dependiendo del método seleccionado.

    Args:
        serie (pd.Series): Serie que se va a suavizar.
        do_abs (bool, opcional): Booleano que define si se aplicará valor absoluto. Por defecto es True.
        window_size (int, opcional): Número de ventana a aplicar el rolling window. En caso de definir 
                                     una ventana mayor al número de elementos de la serie, se asignará 
                                     la longitud de la serie entre 10. Por defecto es 50.
        metodo (str, opcional): Método que se aplicará al rolling window. Por defecto es 'mean'.

    Raises:
        ValueError: En caso de no utilizar un método definido generará un error.

    Returns:
        pd.Series: Regresa la serie suavizada.
    """

    if do_abs:
        serie = serie.abs()
    
    window_size = min(window_size, serie.shape[0])
    serie_rol = serie.rolling(window_size)
    
    if metodo:
        metodo_dict = {
            'mean': serie_rol.mean,
            'std': serie_rol.std,
            'min': serie_rol.min,
            'max': serie_rol.max
        }

        if metodo not in metodo_dict:
            raise ValueError(f"El método '{metodo}' no es válido")
        
        pd_smooth_serie = metodo_dict[metodo]()
    else:
        pd_smooth_serie = serie

    return pd_smooth_serie

def CCF_lags(x : Union[pd.Series, np.ndarray]| np.ndarray, y : Union[pd.Series, np.ndarray]| np.ndarray, max_lag : int =250):
    """
    Calcula la función de correlación cruzada (CCF) entre dos series temporales para una cantidad 
    máxima de desfases (lags).

    Args:
        x (Union[pd.Series, np.ndarray]o np.ndarray): La primera serie temporal.
        y (Union[pd.Series, np.ndarray]o np.ndarray): La segunda serie temporal.
        max_lag (int): El número máximo de desfases (lags) a calcular. Por defecto es 250.

    Returns:
        pd.DataFrame: Un DataFrame con los valores de la CCF y los desfases correspondientes.
    """

    n = np.max([x.shape[0], y.shape[0]])
    lags = np.arange(-max_lag, max_lag + 1)
    ccf_values = [np.corrcoef(x[max(0, -lag):n - max(0, lag)], 
                              y[max(0, lag):n - max(0, -lag)])[0, 1] for lag in lags]
    # return ccf_values, lags
    return pd.DataFrame({'ccf' : ccf_values, 'lags' : lags})

def genera_df_ccf(dict_df : dict, col1 : str = "II", col2 : str = "III") -> pd.DataFrame:
    """
    Genera un DataFrame que contiene las funciones de correlación cruzada (CCF) entre dos columnas 
    específicas de múltiples DataFrames.

    Args:
        dict_df (dict): Un diccionario donde las claves son identificadores de pacientes y los valores 
                        son DataFrames con las series temporales.
        col1 (str): El nombre de la primera columna para calcular la CCF. Por defecto es "II".
        col2 (str): El nombre de la segunda columna para calcular la CCF. Por defecto es "III".

    Returns:
        pd.DataFrame: Un DataFrame donde las filas son los desfases (lags) y las columnas son las CCF 
                      de cada paciente.
    """

    mi_patients = list(dict_df.keys())
    df_ccf = pd.concat(
    [CCF_lags(dict_df[i][col1], dict_df[i][col2]).set_index("lags").rename(columns={'ccf': i}) 
     for i in mi_patients], 
    axis=1, join="inner"
    )

    return df_ccf

def genera_dict_comb_ccf(df : dict, combinaciones : list) -> dict:
    """
    Genera un diccionario que contiene DataFrames con las funciones de correlación cruzada (CCF) para 
    múltiples combinaciones de señales.

    Args:
        df (dict): Un diccionario donde las claves son identificadores de pacientes y los valores son 
            DataFrames con las series temporales.
        combinaciones (list): Una lista de tuplas, donde cada tupla contiene dos nombres de columnas para 
            las cuales se calculará la CCF.

    Returns:
        dict: Un diccionario donde las claves son tuplas de nombres de señales y los valores son DataFrames 
            con las CCF correspondientes.
    """

    dict_ccf = {
        (signal1, signal2): genera_df_ccf(df, col1=signal1, col2=signal2)
        for signal1, signal2 in combinaciones
        }
    return dict_ccf

def plot_ccf_dict(df_dict : dict, x_lags : list = [0, -75, 75]):
    """
    Grafica las funciones de correlación cruzada (CCF) almacenadas en un diccionario.

    Args:
        df_dict (dict): Un diccionario donde las claves son tuplas de nombres de señales y los valores 
                        son DataFrames con las CCF correspondientes.
        x_lags (list): Una lista de desfases (lags) específicos para marcar en las gráficas. 
                       Por defecto es [0, -75, 75].

    Returns:
        None
    """

    for signal1, signal2 in df_dict.keys():
        df_dict[(signal1, signal2)].plot(color="gray", alpha=0.1, legend=False)
        plt.title(f"CCF señal {signal1} vs señal {signal2}, 250 lags")
        [plt.axvline(x, color='black') for x in x_lags]
        plt.show()

def plot_ccf_faces(dict_mi_ccf : dict, dict_sttc_mi_ccf : dict, dict_sttc_ccf : dict, dict_other_ccf : dict, 
                   list_signals : list,
                   x_lags : list = [0, -75, 75],
                   con_suavizamiento = None):
    """
    Grafica las funciones de correlación cruzada (CCF) para múltiples combinaciones de señales y clases, 
    organizadas en una cuadrícula de subgráficas.

    Args:
        dict_mi_ccf (dict): Diccionario con las CCF para la clase MI.
        dict_sttc_mi_ccf (dict): Diccionario con las CCF para la clase STTC MI.
        dict_sttc_ccf (dict): Diccionario con las CCF para la clase STTC.
        dict_other_ccf (dict): Diccionario con las CCF para la clase OTHER.
        list_signals (list): Lista de tuplas, donde cada tupla contiene dos nombres de señales para 
                             las cuales se graficarán las CCF.
        x_lags (list): Lista de desfases (lags) específicos para marcar en las gráficas. 
                       Por defecto es [0, -75, 75].
        con_suavizamiento (bool, opcional): Indica si se debe aplicar suavizamiento a las gráficas. 
                                            Por defecto es None.

    Returns:
        None
    """
    
    suavizamiento = "con suavizamiento" if con_suavizamiento else ""
    
    for signal1, signal2 in list_signals:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8)) 

        dict_mi_ccf[(signal1, signal2)].plot(ax=axs[0, 0], color="gray", alpha=0.1, legend=False)
        axs[0, 0].set_title(f"CCF señal {signal1} vs señal {signal2}, 250 lags. \nClase MI {suavizamiento}")
        [axs[0, 0].axvline(x, color='black') for x in x_lags]


        dict_sttc_mi_ccf[(signal1, signal2)].plot(ax=axs[0, 1], color="gray", alpha=0.1, legend=False)
        axs[0, 1].set_title(f"CCF señal {signal1} vs señal {signal2}, 250 lags. \nClase STTC MI {suavizamiento}")
        [axs[0, 1].axvline(x, color='black') for x in x_lags]

        dict_sttc_ccf[(signal1, signal2)].plot(ax=axs[1, 0], color="gray", alpha=0.1, legend=False)
        axs[1, 0].set_title(f"CCF señal {signal1} vs señal {signal2}, 250 lags. \nClase STTC {suavizamiento}")
        [axs[1, 0].axvline(x, color='black') for x in x_lags]

        dict_other_ccf[(signal1, signal2)].plot(ax=axs[1, 1], color="gray", alpha=0.1, legend=False)
        axs[1, 1].set_title(f"CCF señal {signal1} vs señal {signal2}, 250 lags. \nClase OTHER {suavizamiento}")
        [axs[1, 1].axvline(x, color='black') for x in x_lags]
        
        plt.tight_layout()
        plt.show()

def plot_ccf_faces_stats(dict_mi_ccf : dict,
                        dict_sttc_mi_ccf : dict,
                        dict_sttc_ccf : dict,
                        dict_other_ccf : dict,
                   list_signals : list,
                   status : bool = False,
                   series : bool = True,
                   x_lags : list = [0, -75, 75],
                   con_suavizamiento = None):
    """
    Grafica las funciones de correlación cruzada (CCF) para múltiples combinaciones de señales y clases, 
    organizadas en una cuadrícula de subgráficas, con opción de incluir estadísticas.

    Args:
        dict_mi_ccf (dict): Diccionario con las CCF para la clase MI.
        dict_sttc_mi_ccf (dict): Diccionario con las CCF para la clase STTC MI.
        dict_sttc_ccf (dict): Diccionario con las CCF para la clase STTC.
        dict_other_ccf (dict): Diccionario con las CCF para la clase OTHER.
        list_signals (list): Lista de tuplas, donde cada tupla contiene dos nombres de señales para 
                             las cuales se graficarán las CCF.
        status (bool): Indica si se deben calcular y graficar estadísticas (media y desviación estándar). 
                       Por defecto es False.
        series (bool): Indica si se deben graficar las series individuales. Por defecto es True.
        x_lags (list): Lista de desfases (lags) específicos para marcar en las gráficas. 
                       Por defecto es [0, -75, 75].
        con_suavizamiento (bool, opcional): Indica si se debe aplicar suavizamiento a las gráficas. 
                                            Por defecto es None.

    Returns:
        None
    """
    
    suavizamiento = "con suavizamiento" if con_suavizamiento else ""
    if status:
        dict_mi_ccf_stats = get_estadistica_label_dict(dict_mi_ccf, list_signals)
        dict_sttc_mi_ccf_stats = get_estadistica_label_dict(dict_sttc_mi_ccf, list_signals)
        dict_sttc_ccf_stats = get_estadistica_label_dict(dict_sttc_ccf, list_signals)
        dict_other_ccf_stats = get_estadistica_label_dict(dict_other_ccf, list_signals)
    
    for signal1, signal2 in list_signals:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8)) 
        if series:
            dict_mi_ccf[(signal1, signal2)].plot(ax=axs[0, 0], color="gray", alpha=0.1, legend=False)
        if dict_mi_ccf_stats:
            dict_mi_ccf_stats[(signal1, signal2)]["mean"].plot(ax=axs[0, 0], color="black", alpha=1, legend=False)
            dict_mi_ccf_stats[(signal1, signal2)]["top_std"].plot(ax=axs[0, 0], color="red", alpha=0.5, legend=False)
            dict_mi_ccf_stats[(signal1, signal2)]["buttom_std"].plot(ax=axs[0, 0], color="red", alpha=0.5, legend=False)
        axs[0, 0].set_title(f"CCF señal {signal1} vs señal {signal2}, 250 lags. \nClase MI {suavizamiento}")
        [axs[0, 0].axvline(x, color='black') for x in x_lags]

        if series:
            dict_sttc_mi_ccf[(signal1, signal2)].plot(ax=axs[0, 1], color="gray", alpha=0.1, legend=False)
        if dict_sttc_mi_ccf_stats:
            dict_sttc_mi_ccf_stats[(signal1, signal2)]["mean"].plot(ax=axs[0, 1], color="black", alpha=1, legend=False)
            dict_sttc_mi_ccf_stats[(signal1, signal2)]["top_std"].plot(ax=axs[0, 1], color="red", alpha=0.5, legend=False)
            dict_sttc_mi_ccf_stats[(signal1, signal2)]["buttom_std"].plot(ax=axs[0, 1], color="red", alpha=0.5, legend=False)
        
        axs[0, 1].set_title(f"CCF señal {signal1} vs señal {signal2}, 250 lags. \nClase STTC MI {suavizamiento}")
        [axs[0, 1].axvline(x, color='black') for x in x_lags]

        if series:
            dict_sttc_ccf[(signal1, signal2)].plot(ax=axs[1, 0], color="gray", alpha=0.1, legend=False)
        if dict_sttc_ccf_stats:
            dict_sttc_ccf_stats[(signal1, signal2)]["mean"].plot(ax=axs[1, 0], color="black", alpha=1, legend=False)
            dict_sttc_ccf_stats[(signal1, signal2)]["top_std"].plot(ax=axs[1, 0], color="red", alpha=0.5, legend=False)
            dict_sttc_ccf_stats[(signal1, signal2)]["buttom_std"].plot(ax=axs[1, 0], color="red", alpha=0.5, legend=False)

        axs[1, 0].set_title(f"CCF señal {signal1} vs señal {signal2}, 250 lags. \nClase STTC {suavizamiento}")
        [axs[1, 0].axvline(x, color='black') for x in x_lags]

        if series:
            dict_other_ccf[(signal1, signal2)].plot(ax=axs[1, 1], color="gray", alpha=0.1, legend=False)

        if dict_other_ccf_stats:
            dict_other_ccf_stats[(signal1, signal2)]["mean"].plot(ax=axs[1, 1], color="black", alpha=1, legend=False)
            dict_other_ccf_stats[(signal1, signal2)]["top_std"].plot(ax=axs[1, 1], color="red", alpha=0.5, legend=False)
            dict_other_ccf_stats[(signal1, signal2)]["buttom_std"].plot(ax=axs[1, 1], color="red", alpha=0.5, legend=False)

        axs[1, 1].set_title(f"CCF señal {signal1} vs señal {signal2}, 250 lags. \nClase OTHER {suavizamiento}")
        [axs[1, 1].axvline(x, color='black') for x in x_lags]
        
        plt.tight_layout()
        plt.show()

def dict_apply_function(dict_df : dict, col : str, funct, **kargs) -> pd.DataFrame:
    """
    Aplica una función a una columna específica de múltiples DataFrames almacenados en un diccionario 
    y concatena los resultados en un solo DataFrame.

    Args:
        dict_df (dict): Un diccionario donde las claves son identificadores de pacientes y 
                        los valores son DataFrames con las series temporales.
        col (str): El nombre de la columna a la cual se aplicará la función.
        funct (function): La función que se aplicará a la columna especificada.
        **kargs: Argumentos adicionales que se pasarán a la función.

    Returns:
        pd.DataFrame: Un DataFrame con los resultados de aplicar la función a la columna especificada de cada 
                      DataFrame en el diccionario.
    """
    mi_patients = list(dict_df.keys())
    df_func = pd.concat(
    [dict_df[i][col].apply(funct, **kargs) for i in mi_patients], 
    axis=1, join="inner"
    )
    return df_func

def dict_apply_smooth(dict_df : dict, cols : list, dict_window : dict = None, **kargs):
    """
    Aplica una función de suavizamiento a columnas específicas de múltiples DataFrames almacenados 
    en un diccionario.

    Args:
        dict_df (dict): Un diccionario donde las claves son identificadores de pacientes y 
                        los valores son DataFrames con las series temporales.
        cols (list): Una lista de nombres de columnas a las cuales se aplicará la función 
                     de suavizamiento.
        dict_window (dict, opcional): Un diccionario que especifica el tamaño de la ventana 
                                      de suavizamiento para cada columna. Si no se proporciona, se utilizará un tamaño de ventana por defecto.
        **kargs: Argumentos adicionales que se pasarán a la función de suavizamiento.

    Returns:
        dict: Un diccionario con los DataFrames suavizados.
    """

    if dict_window:
        dict_smooth_df = {
    i: dict_df[i][cols].
        apply(lambda col: smooth_serie(col, window_size=dict_window[col.name], **kargs)).
        dropna().assign(patient_id=i).assign(label=dict_df[i]["label"].values[0])
    for i in dict_df.keys()}
    else:
        dict_smooth_df = {
            i: dict_df[i][cols].apply(smooth_serie, **kargs).dropna().
            assign(patient_id=i).assign(label=dict_df[i]["label"].values[0])
            for i in dict_df.keys()}
    return dict_smooth_df

def get_estadistica_dict(df_dict : dict, signals : list) -> dict:
    """
    Calcula estadísticas descriptivas para múltiples señales en un diccionario de DataFrames.

    Args:
        df_dict (dict): Un diccionario donde las claves son identificadores de pacientes y 
                        los valores son DataFrames con las series temporales.
        signals (list): Una lista de nombres de señales para las cuales se calcularán las estadísticas.

    Returns:
        dict: Un diccionario donde las claves son los nombres de las señales y los valores son 
              DataFrames con las estadísticas calculadas (media, mínimo, máximo, desviación estándar, 
              límite superior e inferior de la desviación estándar).
    """

    df_stats = {
        label: pd.DataFrame({
            pacient: df_dict[pacient][label] 
            for pacient in df_dict.keys()
        }).iloc[:, 1:].agg(["mean", "min", "max", "std"], axis=1).
        assign(top_std=lambda df: df["mean"]+df["std"]).
        assign(buttom_std=lambda df: df["mean"]-df["std"])
        for label in signals
    }
    return df_stats

def get_estadistica_label_dict(df_dict : dict, signals : list) -> dict:
    """
    Calcula estadísticas descriptivas para múltiples señales en un diccionario de DataFrames.

    Args:
        df_dict (dict): Un diccionario donde las claves son identificadores de pacientes y 
                        los valores son DataFrames con las series temporales.
        signals (list): Una lista de nombres de señales para las cuales se calcularán las estadísticas.

    Returns:
        dict: Un diccionario donde las claves son los nombres de las señales y los valores son 
              DataFrames con las estadísticas calculadas (media, mínimo, máximo, desviación estándar, 
              límite superior e inferior de la desviación estándar).
    """

    df_stats = {
        label: pd.DataFrame({
            pacient: df_dict[label][pacient] 
            for pacient in df_dict[label].keys()
        }).iloc[:, 1:].agg(["mean", "min", "max", "std"], axis=1).
        assign(top_std=lambda df: df["mean"]+df["std"]).
        assign(buttom_std=lambda df: df["mean"]-df["std"])
        for label in signals
    }
    return df_stats

def plot_signal_stats(df_mi_stats : dict, df_sttc_mi_stats : dict, df_sttc_stats : dict, df_other_stats : dict,
          list_signals : list,
          stat : str = "mean",
          with_std : bool = False,
          con_suavizamiento = None):
    # list_signals = ['AVL', 'V3', 'V1', 'V2', 'II', 'V4', 'V5', 'V6', 'III', 'AVR', 'AVF', 'I']
    """
    Grafica estadísticas de señales para múltiples clases, organizadas en una cuadrícula de subgráficas.

    Args:
        df_mi_stats (dict): Diccionario con las estadísticas de las señales para la clase MI.
        df_sttc_mi_stats (dict): Diccionario con las estadísticas de las señales para la clase STTC MI.
        df_sttc_stats (dict): Diccionario con las estadísticas de las señales para la clase STTC.
        df_other_stats (dict): Diccionario con las estadísticas de las señales para la clase OTHER.
        list_signals (list): Lista de nombres de señales para las cuales se graficarán las estadísticas.
        stat (str): La estadística a graficar (por defecto es "mean").
        with_std (bool): Indica si se deben graficar las bandas de desviación estándar. 
                        Por defecto es False.
        con_suavizamiento (bool, opcional): Indica si se debe aplicar suavizamiento a las gráficas. 
                                            Por defecto es None.

    Returns:
        None
    """

    suavizamiento = "con suavizamiento" if con_suavizamiento else ""

    for i in list_signals:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8)) 

        df_mi_stats[i][stat].plot(ax=axs[0, 0], color="navy", alpha=1, legend=False)
        if with_std:
            df_mi_stats[i]["top_std"].plot(ax=axs[0, 0], color="red", alpha=0.5, legend=False)
            df_mi_stats[i]["buttom_std"].plot(ax=axs[0, 0], color="red", alpha=0.5, legend=False)
        axs[0, 0].set_title(f"Señal {i}. \nClase MI {suavizamiento} \n({stat})")

        df_sttc_mi_stats[i][stat].plot(ax=axs[0, 1], color="navy", alpha=1, legend=False)
        if with_std:
            df_sttc_mi_stats[i]["top_std"].plot(ax=axs[0, 1], color="red", alpha=0.5, legend=False)
            df_sttc_mi_stats[i]["buttom_std"].plot(ax=axs[0, 1], color="red", alpha=0.5, legend=False)
        axs[0, 1].set_title(f"Señal {i}. \nClase STTC MI {suavizamiento} \n({stat})")

        df_sttc_stats[i][stat].plot(ax=axs[1, 0], color="navy", alpha=1, legend=False)
        if with_std:
            df_sttc_stats[i]["top_std"].plot(ax=axs[1, 0], color="red", alpha=0.5, legend=False)
            df_sttc_stats[i]["buttom_std"].plot(ax=axs[1, 0], color="red", alpha=0.5, legend=False)
        axs[1, 0].set_title(f"Señal {i}.\nClase STTC {suavizamiento} \n({stat})")

        df_other_stats[i][stat].plot(ax=axs[1, 1], color="navy", alpha=1, legend=False)
        if with_std:
            df_other_stats[i]["top_std"].plot(ax=axs[1, 1], color="red", alpha=0.5, legend=False)
            df_other_stats[i]["buttom_std"].plot(ax=axs[1, 1], color="red", alpha=0.5, legend=False)
        axs[1, 1].set_title(f"Señal {i}. \nClase OTHER {suavizamiento} \n({stat})")
        
        plt.tight_layout()
        plt.show()

def prueba_ljung_box_labels(df_dict : dict, signals : list) -> pd.DataFrame:
    """
    Realiza la prueba de Ljung-Box para múltiples señales en un diccionario de DataFrames y 
    calcula el promedio de los resultados.

    Args:
        df_dict (dict): Un diccionario donde las claves son identificadores de pacientes y 
                        los valores son DataFrames con las series temporales.
        signals (list): Una lista de nombres de señales para las cuales se realizará la 
                        prueba de Ljung-Box.

    Returns:
        pd.Series: Una serie con el promedio de los resultados de la prueba de Ljung-Box para cada señal.
    """
    plb = pd.DataFrame({label :[ 1 if (acorr_ljungbox(df_dict[pacient][label], 
                                                lags=[100], return_df=True)['lb_pvalue'].iloc[0]) < 0.05 else 0
        for pacient in df_dict.keys()]
    for label in signals
    }).mean()

    return plb

def prueba_dickey_fuller(df_dict : dict, signals : list, apply_diff : bool = False):
    """
    Realiza la prueba de Dickey-Fuller aumentada para múltiples señales en un diccionario de 
    DataFrames y calcula el promedio de los resultados.

    Args:
        df_dict (dict): Un diccionario donde las claves son identificadores de pacientes y los 
                        valores son DataFrames con las series temporales.
        signals (list): Una lista de nombres de señales para las cuales se realizará la prueba
                        de Dickey-Fuller.
        apply_diff (bool): Indica si se debe aplicar la diferenciación a las series temporales 
                           antes de realizar la prueba. Por defecto es False.

    Returns:
        pd.Series: Una serie con el promedio de los resultados de la prueba de Dickey-Fuller 
                   para cada señal.
    """

    if apply_diff:
        pdf = pd.DataFrame({label :[ 1 if (adfuller(df_dict[pacient][label].diff().dropna())[1]) < 0.05 else 0
            for pacient in df_dict.keys()]
        for label in signals
        }).mean()
    else:
        pdf = pd.DataFrame({label :[ 1 if (adfuller(df_dict[pacient][label])[1]) < 0.05 else 0
            for pacient in df_dict.keys()]
        for label in signals
        }).mean()

    return pdf

def evalua_ks(serie1 : Union[pd.Series, np.ndarray], serie2 : Union[pd.Series, np.ndarray], alpha : float = 0.05) -> int:
    """
    Evalúa la prueba de Kolmogorov-Smirnov (KS) para dos series temporales.

    Args:
        serie1 (Union[pd.Series, np.ndarray]o np.ndarray): La primera serie temporal.
        serie2 (Union[pd.Series, np.ndarray]o np.ndarray): La segunda serie temporal.
        alpha (float): El nivel de significancia para la prueba. Por defecto es 0.05.

    Returns:
        int: Retorna 1 si el p-valor es mayor que alpha (no se rechaza la hipótesis nula), 
             de lo contrario retorna 0.
    """
    _, p_value = ks_2samp(serie1, serie2)
    return 1 if p_value > alpha else 0

def eval_corr(serie1 : Union[pd.Series, np.ndarray]|Union[pd.Series, np.ndarray], serie2 : Union[pd.Series, np.ndarray]|Union[pd.Series, np.ndarray]) -> tuple:
    """
    Calcula la correlación de Pearson y Spearman entre dos series temporales.

    Args:
        serie1 (Union[pd.Series, np.ndarray]o np.ndarray): La primera serie temporal.
        serie2 (Union[pd.Series, np.ndarray]o np.ndarray): La segunda serie temporal.

    Returns:
        tuple: Una tupla que contiene la correlación de Pearson y la correlación de Spearman.
    """
    pearson_corr, _ = pearsonr(serie1, serie2)
    spearman_corr, _ = spearmanr(serie1, serie2)
    return pearson_corr, spearman_corr

def max_lag_corr(serie1 : Union[pd.Series, np.ndarray]|Union[pd.Series, np.ndarray], serie2 : Union[pd.Series, np.ndarray]|Union[pd.Series, np.ndarray]) -> int:
    """
    Calcula el desfase (lag) máximo de correlación cruzada entre dos series temporales.

    Args:
        serie1 (Union[pd.Series, np.ndarray]o np.ndarray): La primera serie temporal.
        serie2 (Union[pd.Series, np.ndarray]o np.ndarray): La segunda serie temporal.

    Returns:
        int: El desfase (lag) en el que se encuentra la máxima correlación cruzada.
    """
    cross_corr = np.correlate(serie1, serie2, mode="full")
    max_len = np.max(len(serie1), len(serie2))
    max_lag = np.argmax(cross_corr) - (max_len - 1)
    return max_lag

def distancia_euclidiana(serie1 : Union[pd.Series, np.ndarray]|Union[pd.Series, np.ndarray], serie2 : Union[pd.Series, np.ndarray]|Union[pd.Series, np.ndarray]) -> float:
    """
    Calcula la distancia euclidiana entre dos series temporales.

    Args:
        serie1 (Union[pd.Series, np.ndarray]o np.ndarray): La primera serie temporal.
        serie2 (Union[pd.Series, np.ndarray]o np.ndarray): La segunda serie temporal.

    Returns:
        float: La distancia euclidiana entre las dos series temporales.
    """

    distancia = np.linalg.norm(serie1 - serie2)
    return distancia

def eval_coint(serie1 : Union[pd.Series, np.ndarray]|Union[pd.Series, np.ndarray], serie2 : Union[pd.Series, np.ndarray]|Union[pd.Series, np.ndarray], alpha : float = 0.05) -> int:
    """
    Evalúa la prueba de cointegración entre dos series temporales.

    Args:
        serie1 (Union[pd.Series, np.ndarray]o np.ndarray): La primera serie temporal.
        serie2 (Union[pd.Series, np.ndarray]o np.ndarray): La segunda serie temporal.
        alpha (float): El nivel de significancia para la prueba. Por defecto es 0.05.

    Returns:
        int: Retorna 1 si el p-valor es menor que alpha (se rechaza la hipótesis nula de no cointegración), 
             de lo contrario retorna 0.
    """

    stat, p_value, _ = coint(serie1, serie2)
    stat, p_value
    return 1 if p_value < alpha else 0
    
def plot_acf_pact_analysis(df : pd.DataFrame, 
                           label : str, 
                           metric : str = "mean", 
                           apply_diff : bool = False, 
                           method = None, 
                           clase : str = "", 
                           intervalo_confianza : bool = False, 
                           **kwargs):
    """
    Grafica el análisis de la función de autocorrelación (ACF) y la función de autocorrelación 
    parcial (PACF) para una serie temporal.

    Args:
        df (pd.DataFrame): El DataFrame que contiene las series temporales.
        label (str): El nombre de la columna que contiene la serie temporal a analizar.
        metric (str): La métrica a utilizar para calcular la ACF y PACF. Por defecto es "mean".
        apply_diff (bool): Indica si se debe aplicar la diferenciación a la serie temporal 
                           antes de realizar el análisis. Por defecto es False.
        method (str, opcional): El método a utilizar para calcular la PACF. Por defecto es None.
        clase (str, opcional): La clase a la que pertenece la serie temporal. Por defecto es 
                               una cadena vacía.
        intervalo_confianza (bool): Indica si se debe calcular y graficar el intervalo de confianza. 
                                    Por defecto es False.
        **kwargs: Argumentos adicionales que se pasarán a las funciones de ACF y PACF.

    Returns:
        None
    """
    
    if apply_diff:
        df_series = df[label].diff().dropna()
    else: 
        df_series = df[label]
    
    if intervalo_confianza:
        N = df[label].shape[0]
        ic = 1.96/math.sqrt(N)
    else:
        ic = 0
    
    agg_acf = get_estadisticas(df_series.apply(acf, **kwargs))[metric]
    agg_pacf = get_estadisticas(df_series.apply(pacf, method=method, **kwargs))[metric]
    acf_agg = get_estadisticas(df_series).apply(acf, **kwargs)[metric]
    pacf_agg = get_estadisticas(df_series).apply(pacf, method=method, **kwargs)[metric]

    lags = np.arange(agg_acf.shape[0])

    fig, axs = plt.subplots(2, 2, figsize=(10, 8)) 

    fig.suptitle(f"ACT y FACT {clase} {label}")

    axs[0,0].stem(lags, agg_acf)
    axs[0,0].set_xlabel("Lag")
    axs[0,0].set_ylabel(label)
    axs[0,0].set_title(f"Agregado ACF {metric}'s {label}")
    axs[0,0].axhline(ic, color='red', linestyle='--')
    axs[0,0].axhline(-1*ic, color='red', linestyle='--')

    axs[0,1].stem(lags, agg_pacf)
    axs[0,1].set_xlabel("Lag")
    axs[0,1].set_ylabel(label)
    axs[0,1].set_title(f"Agregado PACF {metric}'s {label}")
    axs[0,1].axhline(ic, color='red', linestyle='--')
    axs[0,1].axhline(-1*ic, color='red', linestyle='--')

    axs[1,0].stem(lags, acf_agg)
    axs[1,0].set_xlabel("Lag")
    axs[1,0].set_ylabel(label)
    axs[1,0].set_title(f"ACF {metric}'s {label}")
    axs[1,0].axhline(ic, color='red', linestyle='--')
    axs[1,0].axhline(-1*ic, color='red', linestyle='--')

    axs[1,1].stem(lags, pacf_agg)
    axs[1,1].set_xlabel("Lag")
    axs[1,1].set_ylabel(label)
    axs[1,1].set_title(f"PACF {metric}'s {label}")
    axs[1,1].axhline(ic, color='red', linestyle='--')
    axs[1,1].axhline(-1*ic, color='red', linestyle='--')

    plt.tight_layout()
    plt.show()

def get_peaks_seasonal(df_seasonal : Union[pd.Series, np.ndarray]|Union[pd.Series, np.ndarray], n_std : float = 2) -> list:
    """
    Encuentra los picos estacionales en una serie temporal estacional.

    Args:
        df_seasonal (Union[pd.Series, np.ndarray]o np.ndarray): La serie temporal estacional.
        n_std (float): El número de desviaciones estándar para determinar los picos. Por defecto es 2.

    Returns:
        list: Una lista de índices donde se encuentran los picos y valles en la serie temporal estacional.
    """

    up = find_peaks(df_seasonal, height=df_seasonal.std()*n_std)[0]
    low = find_peaks(-1*df_seasonal, height=df_seasonal.std()*n_std)[0]
    up_low = list(up)+list(low)
    up_low.sort()
    return up_low

def get_list_jumps(up_low : list) -> list:
    """
    Calcula las diferencias entre los índices de picos y valles en una lista.

    Args:
        up_low (list): Una lista de índices que representan los picos y valles en una serie temporal.

    Returns:
        list: Una lista de diferencias entre los índices de picos y valles consecutivos.
    """
    
    diferencias = [up_low[i+1] - up_low[i] for i in range(len(up_low)-1)]
    return diferencias

def get_jumps_signal(df_signal : pd.DataFrame, metric : str = "mean", period : int = 100) -> dict:
    """
    Calcula los saltos promedio en la señal estacional para múltiples señales en un DataFrame.

    Args:
        df_signal (pd.DataFrame): El DataFrame que contiene las series temporales.
        metric (str): La métrica a utilizar para calcular las estadísticas. Por defecto es "mean".
        period (int): El periodo para la descomposición estacional. Por defecto es 100.

    Returns:
        dict: Un diccionario donde las claves son los nombres de las señales y los valores son los 
              saltos promedio en la señal estacional.
    """

    labels = df_signal.keys()
    df_dict_seasonal = {
        label : 
            # [
                np.mean(
                    get_list_jumps(
                        get_peaks_seasonal(
                            seasonal_decompose(
                                pd.Series(
                                    get_estadisticas(df_signal[label])[metric]),
                                        period = period).seasonal)))
                                    # ]
        for label in labels}
    return df_dict_seasonal

def genera_df_jumps_signals(mi_dict : dict, sttc_mi_dict : dict, sttc_dict : dict, other_dict : dict) -> pd.DataFrame:
    """
    Genera un DataFrame que contiene los saltos promedio en la señal estacional para múltiples 
    clases de señales.

    Args:
        mi_dict (dict): Diccionario con los saltos promedio para la clase MI.
        sttc_mi_dict (dict): Diccionario con los saltos promedio para la clase STTC MI.
        sttc_dict (dict): Diccionario con los saltos promedio para la clase STTC.
        other_dict (dict): Diccionario con los saltos promedio para la clase OTHER.

    Returns:
        pd.DataFrame: Un DataFrame que contiene los saltos promedio y la desviación estándar para 
                      cada señal y clase.
    """
    
    df_mi = pd.DataFrame(list(mi_dict.items()), columns=['Señal', 'MI'])
    df_sttc_mi = pd.DataFrame(list(sttc_mi_dict.items()), columns=['Señal', 'STTC MI'])
    df_sttc = pd.DataFrame(list(sttc_dict.items()), columns=['Señal', 'STTC'])
    df_other = pd.DataFrame(list(other_dict.items()), columns=['Señal', 'OTHER'])

    df = pd.merge(df_mi, df_sttc_mi, on='Señal')
    df = pd.merge(df, df_sttc, on='Señal')
    df = pd.merge(df, df_other, on='Señal')

    df["promedio"] = df[['MI', 'STTC MI', 'STTC', 'OTHER']].mean(axis=1)
    df["std"] = df[['MI', 'STTC MI', 'STTC', 'OTHER']].std(axis=1)

    return df

def get_seasonal_trend(df_signal : pd.DataFrame, metric : str = "mean", period : int = 100) -> dict:
    """
    Calcula los saltos promedio en la señal estacional para múltiples señales en un DataFrame.

    Args:
        df_signal (pd.DataFrame): El DataFrame que contiene las series temporales.
        metric (str): La métrica a utilizar para calcular las estadísticas. Por defecto es "mean".
        period (int): El periodo para la descomposición estacional. Por defecto es 100.

    Returns:
        dict: Un diccionario donde las claves son los nombres de las señales y los valores son 
              los saltos promedio en la señal estacional.
    """

    labels = df_signal.keys()
    df_dict_seasonal = {
        label : 
            # [
                np.mean(
                    get_list_jumps(
                        get_peaks_seasonal(
                            seasonal_decompose(
                                pd.Series(
                                    get_estadisticas(df_signal[label])[metric]),
                                        period = period).seasonal)))
                                    # ]
        for label in labels}
    return df_dict_seasonal

def plot_seasonal_analysis(df_mi_signals : pd.DataFrame,
                            df_sttc_mi_signals : pd.DataFrame,
                            df_sttc_signals : pd.DataFrame,
                            df_other_signals : pd.DataFrame,
                            label : str, metric : str = "mean", period : int = 100, 
                            aug_figsize : int = 1,**kwargs):
    """
    Grafica el análisis estacional para múltiples clases de señales.

    Args:
        df_mi_signals (pd.DataFrame): DataFrame con las señales de la clase MI.
        df_sttc_mi_signals (pd.DataFrame): DataFrame con las señales de la clase STTC MI.
        df_sttc_signals (pd.DataFrame): DataFrame con las señales de la clase STTC.
        df_other_signals (pd.DataFrame): DataFrame con las señales de la clase OTHER.
        label (str): El nombre de la columna que contiene la serie temporal a analizar.
        metric (str): La métrica a utilizar para calcular las estadísticas. Por defecto es "mean".
        period (int): El periodo para la descomposición estacional. Por defecto es 100.
        aug_figsize (int): Factor de aumento para el tamaño de la figura. Por defecto es 1.
        **kwargs: Argumentos adicionales opcionales.

    Returns:
        None
    """

    df_mi_seasonal = seasonal_decompose(
                    pd.Series(
                        get_estadisticas(df_mi_signals[label])[metric]),
                            period = period).seasonal
    df_sttc_mi_seasonal = seasonal_decompose(
                    pd.Series(
                        get_estadisticas(df_sttc_mi_signals[label])[metric]),
                            period = period).seasonal
    df_sttc_seasonal = seasonal_decompose(
                    pd.Series(
                        get_estadisticas(df_sttc_signals[label])[metric]),
                            period = period).seasonal
    df_other_seasonal = seasonal_decompose(
                    pd.Series(
                        get_estadisticas(df_other_signals[label])[metric]),
                            period = period).seasonal


    fig, axs = plt.subplots(2, 2, figsize=(aug_figsize*10, aug_figsize*8)) 

    fig.suptitle(f"Seasonal Trends {label}")

    axs[0,0].plot(df_mi_seasonal)
    axs[0,0].set_title(f"Seasonal trend MI {label}")

    axs[0,1].plot(df_sttc_mi_seasonal)
    axs[0,1].set_title(f"Seasonal trend STTC MI {label}")

    axs[1,0].plot(df_sttc_seasonal)
    axs[1,0].set_title(f"Seasonal trend STTC {label}")

    axs[1,1].plot(df_other_seasonal)
    axs[1,1].set_title(f"Seasonal trend OTHER {label}")

    plt.tight_layout()
    plt.show()

def apply_plot_ccf_faces(df_mi : dict, df_sttc_mi : dict, df_sttc : dict, df_other : dict, 
                   list_signals : list, **kwargs):
    """
    Aplica la generación y graficación de funciones de correlación cruzada (CCF) para múltiples clases de señales.

    Args:
        df_mi (dict): Diccionario con las señales de la clase MI.
        df_sttc_mi (dict): Diccionario con las señales de la clase STTC MI.
        df_sttc (dict): Diccionario con las señales de la clase STTC.
        df_other (dict): Diccionario con las señales de la clase OTHER.
        list_signals (list): Lista de tuplas, donde cada tupla contiene dos nombres de señales para las cuales se calcularán y 
                             graficarán las CCF.
        **kwargs: Argumentos adicionales opcionales que se pasarán a la función de graficación.

    Returns:
        None
    """
    
    df_mi_ccf = genera_dict_comb_ccf(df_mi, list_signals)
    df_sttc_mi_ccf = genera_dict_comb_ccf(df_sttc_mi, list_signals)
    df_sttc_ccf = genera_dict_comb_ccf(df_sttc, list_signals)
    df_other_ccf = genera_dict_comb_ccf(df_other, list_signals)
    

    plot_ccf_faces(df_mi_ccf, df_sttc_mi_ccf, df_sttc_ccf, df_other_ccf, 
                   list_signals, **kwargs)
    
def apply_plot_ccf_faces_stats(df_mi : dict,
                        df_sttc_mi : dict,
                        df_sttc : dict,
                        df_other : dict,
                   list_signals : list,
                   status : bool = True,
                   series : bool = True,
                   x_lags : list = [],
                   con_suavizamiento = None):
    """
    Aplica la generación y graficación de funciones de correlación cruzada (CCF) con estadísticas 
    para múltiples clases de señales.

    Args:
        df_mi (dict): Diccionario con las señales de la clase MI.
        df_sttc_mi (dict): Diccionario con las señales de la clase STTC MI.
        df_sttc (dict): Diccionario con las señales de la clase STTC.
        df_other (dict): Diccionario con las señales de la clase OTHER.
        list_signals (list): Lista de tuplas, donde cada tupla contiene dos nombres de señales 
                             para las cuales se calcularán y graficarán las CCF.
        status (bool): Indica si se deben calcular y graficar estadísticas (media y desviación estándar). 
                       Por defecto es True.
        series (bool): Indica si se deben graficar las series individuales. Por defecto es True.
        x_lags (list): Lista de desfases (lags) específicos para marcar en las gráficas. 
                       Por defecto es una lista vacía.
        con_suavizamiento (bool, opcional): Indica si se debe aplicar suavizamiento a las gráficas. 
                                            Por defecto es None.

    Returns:
        None
    """
    
    df_mi_ccf = genera_dict_comb_ccf(df_mi, list_signals)
    df_sttc_mi_ccf = genera_dict_comb_ccf(df_sttc_mi, list_signals)
    df_sttc_ccf = genera_dict_comb_ccf(df_sttc, list_signals)
    df_other_ccf = genera_dict_comb_ccf(df_other, list_signals)
    
    plot_ccf_faces_stats(df_mi_ccf,
                        df_sttc_mi_ccf,
                        df_sttc_ccf,
                        df_other_ccf,
                   list_signals,
                   status = status,
                   series = series,
                   x_lags = x_lags,
                   con_suavizamiento = con_suavizamiento)
    
def apply_plot_signal_stats(df_mi : dict, df_sttc_mi : dict, df_sttc : dict, df_other : dict,
      list_signals : list, con_suavizamiento = None, dict_window : dict = None, window_size : int = None, **kwargs):
    """
    Aplica la generación y graficación de estadísticas de señales para múltiples clases de señales, 
    con opción de suavizamiento.

    Args:
        df_mi (dict): Diccionario con las señales de la clase MI.
        df_sttc_mi (dict): Diccionario con las señales de la clase STTC MI.
        df_sttc (dict): Diccionario con las señales de la clase STTC.
        df_other (dict): Diccionario con las señales de la clase OTHER.
        list_signals (list): Lista de nombres de señales para las cuales se calcularán y graficarán 
                             las estadísticas.
        con_suavizamiento (bool, opcional): Indica si se debe aplicar suavizamiento a las señales. 
                                            Por defecto es None.
        dict_window (dict, opcional): Diccionario que especifica el tamaño de la ventana de 
                                      suavizamiento para cada señal. Por defecto es None.
        window_size (int, opcional): Tamaño de la ventana de suavizamiento. Por defecto es None.
        **kwargs: Argumentos adicionales opcionales que se pasarán a la función de graficación.

    Returns:
        None
    """

    if con_suavizamiento or dict_window or window_size:
        if window_size:
            df_mi = dict_apply_smooth(df_mi, list_signals, window_size = window_size)
            df_sttc_mi = dict_apply_smooth(df_sttc_mi, list_signals, window_size = window_size)
            df_sttc = dict_apply_smooth(df_sttc, list_signals, window_size = window_size)
            df_other = dict_apply_smooth(df_other, list_signals, window_size = window_size)
        else:
            df_mi = dict_apply_smooth(df_mi, list_signals, dict_window = dict_window)
            df_sttc_mi = dict_apply_smooth(df_sttc_mi, list_signals, dict_window = dict_window)
            df_sttc = dict_apply_smooth(df_sttc, list_signals, dict_window = dict_window)
            df_other = dict_apply_smooth(df_other, list_signals, dict_window = dict_window)
        
        con_suavizamiento = True
    
    df_mi_stats = get_estadistica_dict(df_mi, list_signals)
    df_sttc_mi_stats = get_estadistica_dict(df_sttc_mi, list_signals)
    df_sttc_stats = get_estadistica_dict(df_sttc, list_signals)
    df_other_stats = get_estadistica_dict(df_other, list_signals)
        
    plot_signal_stats(df_mi_stats, df_sttc_mi_stats, df_sttc_stats, df_other_stats,
            list_signals, con_suavizamiento = con_suavizamiento, **kwargs)

def split_train_test_val(X,y, sizes = [0.10, 0.20], random_state = 42, stratify = None):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sizes[1], random_state=random_state, stratify=stratify)
    if stratify.any():
        stratify = y_train
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=sizes[0], random_state=random_state, stratify=stratify)
    
    return X_train, X_test, X_val, y_train, y_test, y_val

def get_df_acf_pacf(df, list_signals, apply_diff = False):
    acf_pacf_dict = {}
    patient_id = df["patient_id"].values[0]
    label = df["label"].values[0]

    for signal in list_signals:

        df_series = df[signal].diff().dropna() if apply_diff else df[signal]
        
        serie_acf, serie_pacf = acf_pacf(df_series, lags = 5)
        acf_pacf_dict["acf_" + signal] = (pd.DataFrame(
                    [serie_acf[1:6]], 
                    columns=["acf_" + signal + "_lag_" + str(i) for i in range(1,len(serie_acf[1:6])+1)]))
        acf_pacf_dict["pacf_" + signal] = (pd.DataFrame(
                    [serie_pacf[1:6]], 
                    columns=["pacf_" + signal + "_lag_" + str(i) for i in range(1,len(serie_pacf[1:6])+1)]))

    
    result_df = pd.concat(acf_pacf_dict.values(), axis=1)
    result_df["patient"] = patient_id
    result_df["label"] = label

    return result_df

def genera_df_acf_pacf(df, list_signals, apply_diff = False):
    dict_acf_pacf = {
        pacient: get_df_acf_pacf(df[pacient], list_signals, apply_diff = apply_diff)
        for pacient in df.keys()}
    return dict_acf_pacf

def dict_to_dataframe(dict):
    return pd.concat(dict.values(), axis=0, ignore_index=True)

def get_jumps_patient(df_patients : pd.DataFrame, metric : str = "mean", period : int = 100) -> dict:
    """
    Calcula los saltos promedio en la señal estacional para múltiples señales en un DataFrame.

    Args:
        df_signal (pd.DataFrame): El DataFrame que contiene las series temporales.
        metric (str): La métrica a utilizar para calcular las estadísticas. Por defecto es "mean".
        period (int): El periodo para la descomposición estacional. Por defecto es 100.

    Returns:
        dict: Un diccionario donde las claves son los nombres de las señales y los valores son los 
              saltos promedio en la señal estacional.
    """

    patients = df_patients.keys()
    df_dict_seasonal = {
        patient : 
            # [
                np.mean(
                    get_list_jumps(
                        get_peaks_seasonal(
                            seasonal_decompose(
                                pd.Series(
                                    get_estadisticas(df_patients[patient])[metric]),
                                        period = period).seasonal)))
                                    # ]
        for patient in patients}
    return df_dict_seasonal

class SerieAnalisis:
    def __init__(self, serie):
        self.serie = serie
    def amplitud(self):
        return self.serie.max() - self.serie.min()
    def intensidad(self):
        return self.serie.std()
    def seasonal_serie(self, period = 100):
        return seasonal_decompose(self.serie, period = period).seasonal
    def ratio(self, period=100):
        seasonal = self.seasonal_serie(period)
        return seasonal.std() / self.serie.std()
    def mean_peaks(self, n_std = 2):
        diff_peaks = get_list_jumps(get_peaks_seasonal(self.serie, n_std = n_std))
        if len(diff_peaks) < 2:
            return np.nan
        return np.mean(diff_peaks)
    def std_peaks(self, n_std = 2):
        diff_peaks = get_list_jumps(get_peaks_seasonal(self.serie, n_std = n_std))
        if len(diff_peaks) < 2:
            return np.nan
        return np.std(diff_peaks)
    def n_peaks(self, n_std = 2):
        peaks = get_peaks_seasonal(self.serie, n_std = n_std)
        return len(peaks)

def get_serie_summary(serie, cara, period = 100):
    """
    Genera un resumen de la serie temporal
    """
    class_serie = SerieAnalisis(serie)
    
    return {
        "amplitud" + "_" + cara : class_serie.amplitud(),
        "intensidad" + "_" + cara : class_serie.intensidad(),
        "ratio" + "_" + cara : class_serie.ratio(),
        "mean_peaks" + "_" + cara : class_serie.mean_peaks(),
        "std_peaks" + "_" + cara : class_serie.std_peaks(),
        "n_peaks" + "_" + cara : class_serie.n_peaks()
    }

def get_dict_serie_summary(dict, caras, period,) -> pd.DataFrame:
    """Genera un resumen de la serie temporal

    Args:
        dict (_type_): _description_
        caras (_type_): _description_
        period (_type_): _description_

    Returns:
        pd.DataFrame: _description_
    """

    df_summary = pd.DataFrame()
    
    for patient in dict.keys():
        df_aux = pd.DataFrame()
        for cara in caras:
            if cara in dict[patient].keys():
                serie = dict[patient][cara]
                summary = get_serie_summary(serie, cara, period)
                df_aux = pd.concat([df_aux, pd.DataFrame(summary, index=[0])], axis=1)
        df_aux["patient"] = patient
        # df_aux["cara"] = cara
        df_summary = pd.concat([df_summary, df_aux], axis=0)
                
    return df_summary.reset_index(drop=True)

def patients_dict_ccf(dict, combinaciones):
    patients_dict = {patient: pd.concat(
    [CCF_lags(dict[patient][col1], dict[patient][col2])
        .set_index("lags")
        .rename(columns={'ccf': col1 if col1 == col2 else col1 + "_" + col2}) 
        for col1, col2 in combinaciones], 
    axis=1, join="inner"
    )# .assign(fixed_column="fixed_value")
    for patient in dict.keys()}

    return patients_dict

class CCFAnalisis:
    def __init__(self, serie):
        self.serie = serie
    def cruces_cero(self):
        signos = np.sign(self.serie)
        cruces = signos[:-1].values * signos[1:].values < 0
        indices_cruce = np.where(cruces)[0]
        return indices_cruce
    def n_cruces_cero(self):
        return len(self.cruces_cero())
    def promedio(self):
        return np.mean(self.serie)
    def std(self):
        return np.std(self.serie)
    def max(self):
        return self.serie.max()
    def min(self):
        return self.serie.min()
    def maxlag(self):
        return self.serie.idxmax()
    def minlag(self):
        return self.serie.idxmin()
    def kurtosis(self):
        return kurtosis(self.serie)
    def trim_mean(self, proportion_to_cut=0.05):
        return trim_mean(self.serie, proportion_to_cut)
    

def matrix_norm(df, dict_combinaciones):
    """
    Calcula la norma de una matriz.

    Args:
        df (pd.DataFrame): DataFrame que contiene las ccf.
        dict_combinaciones (dict): Diccionario con las combinaciones de columnas, 
                                   con claves "uni_combinacion" y "bi_combinacion".

    Returns:
        float: La norma calculada.
    """
    norma = (sum((np.mean(df[col]) ** 2) for col in dict_combinaciones["uni_combinacion"]) +
            sum(2 * (np.mean(df[col]) ** 2) for col in dict_combinaciones["bi_combinacion"]))
    return np.sqrt(norma)

def get_ccf_summary(ccf, combinacion, proportion_to_cut=0.05):
    """
    Genera un resumen de la serie temporal
    """
    class_ccf = CCFAnalisis(ccf)
    
    return {
        "n_cruces_cero" + "_" + combinacion : class_ccf.n_cruces_cero(),
        "promedio" + "_" + combinacion : class_ccf.promedio(),
        "std" + "_" + combinacion : class_ccf.std(),
        "max" + "_" + combinacion : class_ccf.max(),
        "min" + "_" + combinacion : class_ccf.min(),
        "maxlag" + "_" + combinacion : class_ccf.maxlag(),
        "minlag" + "_" + combinacion : class_ccf.minlag(),
        "kurtosis" + "_" + combinacion : class_ccf.kurtosis(),
        "trim_mean" + "_" + combinacion : class_ccf.trim_mean(proportion_to_cut = proportion_to_cut),
    }

def get_dict_ccf_summary(dict, dict_combinaciones, proportion_to_cut=0.05,) -> pd.DataFrame:
    """Genera un resumen de la ccf

    Args:
        dict (_type_): _description_
        caras (_type_): _description_
        period (_type_): _description_

    Returns:
        pd.DataFrame: _description_
    """

    df_summary = pd.DataFrame()
    
    for patient in dict.keys():
        df_aux = pd.DataFrame()
        for combinacion in dict_combinaciones["uni_combinacion"] + dict_combinaciones["bi_combinacion"]:
            if combinacion in dict[patient].keys():
                serie = dict[patient][combinacion]
                summary = get_ccf_summary(serie, combinacion, proportion_to_cut)
                df_aux = pd.concat([df_aux, pd.DataFrame(summary, index=[0])], axis=1)
        df_aux["patient"] = patient
        # df_aux["combinacion"] = combinacion
        df_aux["norm_ccf"] = matrix_norm(dict[patient], dict_combinaciones)
        df_summary = pd.concat([df_summary, df_aux], axis=0)
                
    return df_summary.reset_index(drop=True)

def genera_acf_pacf_features(df, patients, clase, list_signals, muestra):
    patients_clase = patients[patients["class"] == clase]["patient"].values
    df_clase = {patient : df[patient] for patient in patients_clase}
    df_acf_pacf = dict_to_dataframe(genera_df_acf_pacf(df_clase, list_signals, apply_diff= True))
    df_acf_pacf.to_csv(f"output/features/{muestra}/{clase}_acf_pacf.csv", index = False)

def genera_peak_features(df, patients, clase, list_signals, muestra, period = 100):
    patients_clase = patients[patients["class"] == clase]["patient"].values
    df_clase = {patient : df[patient] for patient in patients_clase}
    df_peak = get_dict_serie_summary(df_clase, list_signals, period)
    df_peak.to_csv(f"output/features/{muestra}/{clase}_peak.csv", index = False)


def genera_ccf_features(df, patients, clase, dict_combinaciones, muestra):
    patients_clase = patients[patients["class"] == clase]["patient"].values
    df_clase = {patient : df[patient] for patient in patients_clase}
    df_ccf = patients_dict_ccf(df_clase, dict_combinaciones["uni_combinacion"] + dict_combinaciones["bi_combinacion"])
    df_cff_stats = get_dict_ccf_summary(df_ccf, dict_combinaciones,)
    df_cff_stats.to_csv(f"output/features/{muestra}/{clase}_cff_stats.csv", index = False)


def get_features(muestra, clase, proyect_path = os.getcwd(), gen_csv = True):
    df_acf_pacf = pd.read_csv(f"{proyect_path}/output/features/{muestra}/{clase}_acf_pacf.csv")
    df_peak = pd.read_csv(f"{proyect_path}/output/features/{muestra}/{clase}_peak.csv").drop('cara', axis=1, errors='ignore')
    df_ccf_stats = pd.read_csv(f"{proyect_path}/output/features/{muestra}/{clase}_ccf_stats.csv").drop('combinacion', axis=1, errors='ignore')

    df_features = pd.merge(pd.merge(df_acf_pacf, df_peak, on=["patient"], how='inner'),
                     df_ccf_stats, on=["patient"], how='inner')
    
    if gen_csv:
        df_features.to_csv(f"{proyect_path}/output/features/{muestra}/{clase}_features.csv", index=False)
        return
    
    return df_features