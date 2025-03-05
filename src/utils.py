import os
import pandas as pd
import numpy as np

from scipy.signal import find_peaks
from scipy.stats import ks_2samp, pearsonr, spearmanr


from statsmodels.tsa.stattools import acf, pacf, adfuller, coint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

import matplotlib.pyplot as plt

import math
# from config import data_folder

def lectura_carpetas_dict(data_path : str) -> dict:

    """
    La función tiene como objetivo, dada una ruta específica, obtener las subcarpetas
    y leer los archivos CSV de estas subcarpetas.

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

    """La función tiene como objetivo dado un diccionario de series obtener la autocorrelación y la
        autocorrelación parcial de la serie dado una clase.

    Args:
        dict_series (dict): diccionario con las distintas series a analizar
        label (str) : clase que se va a analizar de la serie

    Returns:
        tuple: la salida de la función es una tupla conformada de dos pd.DataFrame que contiene los 
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

    """La función tiene como objetivo dado un diccionario de series generar un dataframe
        con una clase dada.

    Args:
        dict_series (dict): diccionario con las distintas series a analizar
        label (str) : clase que se va a analizar de la serie

    Returns:
        pd.DataFrame: la salida de la función es un pd.DataFrame que contiene los 
        valores de cada serie.
    """
    df_dict = {}

    for pacient in dict_series.keys():
        pd_series = dict_series[pacient][label]
        df_dict[pacient] = pd_series

    df = pd.DataFrame(df_dict)
    
    return df

def get_estadisticas(df : pd.DataFrame, **kargs) -> pd.DataFrame:
    """Dado un DataFrame la función entrega un dataframe con las estadísticas por fila.

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

def CCF_lags(x, y, max_lag=20):
    n = np.max([x.shape[0], y.shape[0]])
    lags = np.arange(-max_lag, max_lag + 1)
    ccf_values = [np.corrcoef(x[max(0, -lag):n - max(0, lag)], 
                              y[max(0, lag):n - max(0, -lag)])[0, 1] for lag in lags]
    # return ccf_values, lags
    return pd.DataFrame({'ccf' : ccf_values, 'lags' : lags})

def plot_CCF(ccf_values, lags, figsize = (10,10)):

    # Graficar función de correlación cruzada
    plt.figure(figsize=figsize)
    plt.stem(lags, ccf_values)
    plt.axhline(0, color='black', linestyle="--")
    plt.xlabel("Lag")
    plt.ylabel("Correlación Cruzada")
    plt.title("Cross-Correlation Function (CCF)")
    plt.show()

def diff_ts(series : pd.Series, differences : int = 1) -> pd.Series:
    """La función recive como argumento una serie de pandas y el número de 
    diferencias que se le aplicarán a la serie. La función regresa la 
    serie diferenciada n veces.

    Args:
        series (pd.Series): recibe una pandas serie
        differences (int, optional): número de diferencias que se le aplicará a la serie. Defaults to 1.

    Returns:
        pd.Series: regresa la serie diferenciada.
    """
    diff_series = series
    for i in range(differences):
        diff_series = diff_series.diff().dropna()
    
    return diff_series

def plot_acf_pacf_serie(series : pd.Series, lags : int = None, **kwargs):
    if lags == None:
        lags = 10*int(math.log((series.shape[0])))

    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (8,8))
    _ = plot_acf(series, lags=lags, zero = False, ax = ax1)
    _ = plot_pacf(series, lags=lags, zero = False, ax = ax2)

def acf_pacf(series : pd.Series, nlags : int = None, **kwargs):
    if nlags == None:
        nlags = 10*int(math.log((series.shape[0])))

    acf_vals = acf(series, nlags=nlags)
    pacf_vals = pacf(series, nlags=nlags)

    return acf_vals, pacf_vals

def fit_arima(series: pd.Series, order : tuple[int, int, int], **kwargs) -> pd.Series:
    """Obtiene los residuales después de evaluar un modelo arima

    Args:
        series (pd.Series): erie a ajustar modelo arima
        order (tuple[int, int, int]): orden del modelo ARIMA(p,d,q)
            p (int): grado autorregresivo
            d (int): número de diferencias
            q (int): grado de promedio móviles

    Returns:
        pd.Series: residuos posterior a ajustar el modelo
    """
    

    model = ARIMA(series, order=order)
    result = model.fit()
    return result



def smooth_serie(serie: pd.Series, do_abs: bool = True, window_size: int = 50, metodo: str = 'mean') -> pd.Series:
    """La función suaviza la serie aplicado primero valor absoluto en caso de no definir lo contratio y 
    aplica un método de rolling windows dependiendo del método seleccionado.

    Args:
        serie (pd.Series): serie que se va a suavizar
        do_abs (bool, optional): booleano que define si se aplicará valor absoluto. Defaults to True.
        window_size (int, optional): número de ventana a aplicar el rolling window, en caso de definir 
        una ventana mayor al número de elementos de la serie, se asignará la longitud de la serie entre 10. Defaults to 50.
        metodo (str, optional): método que se aplicará al rolling window. Defaults to 'mean'.

    Raises:
        ValueError: en caso de no utilizar un método definido generará un error.

    Returns:
        pd.Series: regresa la serie suavizada
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

def CCF_lags(x, y, max_lag=250):

    n = np.max([x.shape[0], y.shape[0]])
    lags = np.arange(-max_lag, max_lag + 1)
    ccf_values = [np.corrcoef(x[max(0, -lag):n - max(0, lag)], 
                              y[max(0, lag):n - max(0, -lag)])[0, 1] for lag in lags]
    # return ccf_values, lags
    return pd.DataFrame({'ccf' : ccf_values, 'lags' : lags})

def genera_df_ccf(dict_df, col1 : str = "II", col2 : str = "III"):

    mi_patients = list(dict_df.keys())
    df_ccf = pd.concat(
    [CCF_lags(dict_df[i][col1], dict_df[i][col2]).set_index("lags").rename(columns={'ccf': i}) 
     for i in mi_patients], 
    axis=1, join="inner"
    )

    return df_ccf

def genera_dict_comb_ccf(df, combinaciones):
    dict_ccf = {
        (signal1, signal2): genera_df_ccf(df, col1=signal1, col2=signal2)
        for signal1, signal2 in combinaciones
        }
    return dict_ccf


def plot_ccf_dict(df_dict, x_lags = [0, -75, 75]):
    for signal1, signal2 in df_dict.keys():
        df_dict[(signal1, signal2)].plot(color="gray", alpha=0.1, legend=False)
        plt.title(f"CCF señal {signal1} vs señal {signal2}, 250 lags")
        [plt.axvline(x, color='black') for x in x_lags]
        plt.show()

def plot_ccf_faces(dict_mi_ccf, dict_sttc_mi_ccf, dict_sttc_ccf, dict_other_ccf, 
                   list_signals,
                   x_lags = [0, -75, 75],
                   con_suavizamiento = None):
    
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

def plot_ccf_faces_stats(dict_mi_ccf,
                        dict_sttc_mi_ccf,
                        dict_sttc_ccf,
                        dict_other_ccf,
                   list_signals,
                   status = False,
                   series = True,
                   x_lags = [0, -75, 75],
                   con_suavizamiento = None):
    
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

def dict_apply_function(dict_df, col, funct, **kargs):
    mi_patients = list(dict_df.keys())
    df_func = pd.concat(
    [dict_df[i][col].apply(funct, **kargs) for i in mi_patients], 
    axis=1, join="inner"
    )
    return df_func

def dict_apply_smooth(dict_df, cols, **kargs):

    dict_smooth_df = {
        i: dict_df[i][cols].apply(smooth_serie, **kargs).dropna()
        for i in dict_df.keys()}
    return dict_smooth_df

def get_estadistica_dict(df_dict, signals):
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

def get_estadistica_label_dict(df_dict, signals):
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

def del_get_estadistica_label_dict(df_dict, signals):

    dict_label_stats = {}
    for i in signals:
        pd_mean = df_dict[i].mean(axis=1)
        pd_min = df_dict[i].min(axis=1)
        pd_max = df_dict[i].max(axis=1)
        pd_std = df_dict[i].std(axis=1)
        pd_top_std = pd_mean + pd_std
        pd_buttom_std = pd_mean - pd_std

        dict_label_stats[i] = pd.DataFrame({"mean" : pd_mean,
                                        "std": pd_std,
                                        "min" : pd_min,
                                        "max" : pd_max,
                                        "top_std" : pd_top_std,
                                        "buttom_std": pd_buttom_std})

    return dict_label_stats


def plot_signal_stats(df_mi_stats, df_sttc_mi_stats, df_sttc_stats, df_other_stats,
          list_signals,
          stat = "mean",
          with_std = False,
          con_suavizamiento = None):
    # list_signals = ['AVL', 'V3', 'V1', 'V2', 'II', 'V4', 'V5', 'V6', 'III', 'AVR', 'AVF', 'I']

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
        axs[0, 0].set_title(f"Señal {i}. \nClase MI {suavizamiento} \n({stat})")
        axs[0, 1].set_title(f"Señal {i}. \nClase STTC MI \n({stat})")

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

def prueba_ljung_box_labels(df_dict, signals):
    plb = pd.DataFrame({label :[ 1 if (acorr_ljungbox(df_dict[pacient][label], 
                                                lags=[100], return_df=True)['lb_pvalue'].iloc[0]) < 0.05 else 0
        for pacient in df_dict.keys()]
    for label in signals
    }).mean()

    return plb

def prueba_dickey_fuller(df_dict, signals):
    pdf = pd.DataFrame({label :[ 1 if (adfuller(df_dict[pacient][label])[1]) < 0.05 else 0
        for pacient in df_dict.keys()]
    for label in signals
    }).mean()

    return pdf

def evalua_ks(serie1, serie2, alpha = 0.05):
    _, p_value = ks_2samp(serie1, serie2)
    return 1 if p_value > alpha else 0

def eval_corr(serie1, serie2):
    pearson_corr, _ = pearsonr(serie1, serie2)
    spearman_corr, _ = spearmanr(serie1, serie2)
    return pearson_corr, spearman_corr

def max_lag_corr(serie1, serie2):
    cross_corr = np.correlate(serie1, serie2, mode="full")
    max_len = np.max(len(serie1), len(serie2))
    max_lag = np.argmax(cross_corr) - (max_len - 1)
    return max_lag

def distancia_euclidiana(serie1, serie2):
    distancia = np.linalg.norm(serie1 - serie2)
    return distancia

def eval_coint(serie1, serie2, alpha = 0.05):
    stat, p_value, _ = coint(serie1, serie2)
    stat, p_value
    return 1 if p_value < alpha else 0
    
#{key: value for key, value in iterable if condition}

def plot_acf_pact_analysis(df, label, metric : str = "mean", apply_diff = False, method = None, clase = "", **kwargs):
    
    if apply_diff:
        df_series = df[label].diff().dropna()
    else: 
        df_series = df[label]
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

    axs[0,1].stem(lags, agg_pacf)
    axs[0,1].set_xlabel("Lag")
    axs[0,1].set_ylabel(label)
    axs[0,1].set_title(f"Agregado PACF {metric}'s {label}")

    axs[1,0].stem(lags, acf_agg)
    axs[1,0].set_xlabel("Lag")
    axs[1,0].set_ylabel(label)
    axs[1,0].set_title(f"ACF {metric}'s {label}")

    axs[1,1].stem(lags, pacf_agg)
    axs[1,1].set_xlabel("Lag")
    axs[1,1].set_ylabel(label)
    axs[1,1].set_title(f"PACF {metric}'s {label}")

    plt.tight_layout()
    plt.show()

