# %%
# Librerías
import os
import pandas as pd
import numpy as np
import openpyxl
import missingno as msno
from colorama import Fore, Style
import seaborn as sns
import matplotlib.pyplot as plt
import sys  
import plotly.express as px
from sklearn.model_selection import train_test_split
import scipy.stats 


#%%
def relacion_nulos_target(df):
    """
    Esta función calcula el porcentaje de valores nulos en cada columna del DataFrame,
    dividido por las categorías de la variable 'TARGET'. 

    Args:
        df (DataFrame): El DataFrame de pandas que contiene los datos.

    Returns:
        dict: Un diccionario donde las claves son los nombres de las columnas con valores nulos
              y los valores son los porcentajes de nulos para cada categoría del 'TARGET'.
    """
    # Crear un diccionario para almacenar los resultados
    lista = {}
    
    # Filtrar columnas que tienen valores nulos
    columnas_con_nulos = df.columns[df.isnull().any()]
    
    # Calcular el porcentaje de nulos por TARGET para cada columna
    for col in columnas_con_nulos:
        porcentajes_nulos = df.groupby('TARGET')[col].apply(lambda x: (x.isnull().sum() / len(x)) * 100)
        lista[col] = porcentajes_nulos  # Asignar al diccionario

    return lista


#%%
def categorizar_columnas(df):
    """
    Esta función categoriza las columnas de un DataFrame en tres grupos: 
    columnas booleanas, columnas numéricas y columnas categóricas.
    
    Args:
        df (DataFrame): El DataFrame de pandas que contiene los datos.

    Returns:
        tuple: Un tuple con tres listas:
            - col_bool: Listado de columnas booleanas o de enteros con solo valores 0 y 1.
            - col_cat: Listado de columnas categóricas (tipo 'object' o 'category').
            - col_num: Listado de columnas numéricas (tipo 'int64' o 'float') que no son booleanas.
    """
    # Columnas booleanas puras (tipo bool) y tipo int con solo 1 y 0
    col_bool = [col for col in df.columns if df[col].dtype == 'bool' or
                     (df[col].dtype == 'int64' and set(df[col].dropna().unique()) <= {0, 1})]
    
    # Columnas numéricas (int y float) que no son booleanas en disguise
    col_num = [col for col in df.select_dtypes(include=['int64', 'float']).columns 
                    if col not in col_bool]
    
    # Columnas categóricas (objetos y categorías)
    col_cat = df.select_dtypes(include=['object', 'category']).columns.tolist()

    return col_bool, col_cat, col_num



# %%

YELLOW = Fore.YELLOW  # Numérico
BLUE = Fore.BLUE      # Categórico
MAGENTA = Fore.MAGENTA  # Booleano
RESET = Style.RESET_ALL  # Reset color

def data_summary(df):
    """
    Esta función genera un resumen detallado de las columnas de un DataFrame, 
    identificando su tipo de datos (booleano, numérico, categórico) y mostrando 
    información relevante para cada tipo de columna.
    
    Args:
        df (DataFrame): El DataFrame de pandas que contiene los datos a analizar.
        
    Imprime el nombre de la columna, su tipo de dato y estadísticas clave según su tipo:
        - Para columnas booleanas: valores únicos.
        - Para columnas numéricas: rango y media.
        - Para columnas categóricas: muestra los primeros valores únicos.
    """
    
    for col in df.columns:
        # Detectar si la columna tiene solo valores 0 y 1 (tratarlas como booleanas)
        if df[col].isin([0, 1]).all():
            column_type = f"{MAGENTA}boolean{RESET}"
        elif df[col].dtype in ['int64', 'float64']:  # Identificar columnas numéricas
            column_type = f"{YELLOW}numeric{RESET}"
        elif df[col].dtype == 'bool':  # Identificar columnas booleanas
            column_type = f"{MAGENTA}boolean{RESET}"
        elif df[col].dtype == 'object':  # Identificar columnas categóricas
            column_type = f"{BLUE}categoric{RESET}"
        else:
            column_type = df[col].dtype  # Para otros tipos, sin color

        # Nombre de la columna y tipo de dato con color
        print(f"{col} ({column_type}) :", end=" ")
        
        # Tipo de dato detallado sin color
        print(f"(Type: {df[col].dtype})", end=" ")

        # Mostrar valores según el tipo de columna
        if column_type == f"{MAGENTA}boolean{RESET}":
            unique_values = df[col].unique()
            if len(unique_values) == 1:  # Si la columna solo tiene un valor único
                print(f"Unique: [{unique_values[0]}]")
            else:
                print(f"Unique: {list(unique_values)}")
        
        elif column_type == f"{YELLOW}numeric{RESET}":
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            print(f"Range = [{min_val:.2f} to {max_val:.2f}], Mean = {mean_val:.2f}")
        
        elif column_type == f"{BLUE}categoric{RESET}":
            unique_values = df[col].unique()
            # Mostrar los primeros 5 valores únicos
            print(f"Values: {unique_values[:5]}{' ...' if len(unique_values) > 5 else ''}")

        print()  # Línea en blanco para separar columnas

#%%
def calcular_tabla_resumen(df, outlier_threshold=3):
    """
    Genera una tabla resumen con las proporciones de valores nulos, valores atípicos y otros datos relevantes.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        outlier_threshold (float): Umbral para definir valores atípicos (basado en desviaciones estándar).
        
    Returns:
        pd.DataFrame: Tabla resumen con las columnas especificadas.
    """
    resumen = []
    n_rows = len(df)
    
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):  # Solo para columnas numéricas
            # Cálculo de outliers usando la regla de 3 sigma
            mean = df[col].mean()
            std = df[col].std()
            lower_limit = mean - outlier_threshold * std
            upper_limit = mean + outlier_threshold * std
            outlier_count = df[col][(df[col] < lower_limit) | (df[col] > upper_limit)].count()
            
            # Cálculo de valores nulos
            null_count = df[col].isnull().sum()
            
            resumen.append({
                '0.0': 1 - (null_count / n_rows),  # Proporción no nula
                '1.0': null_count / n_rows,       # Proporción nula
                'variable': col,
                'sum_outlier_values': outlier_count,
                'porcentaje_sum_null_values': null_count / n_rows
            })
    
    # Convertir a DataFrame
    resumen_df = pd.DataFrame(resumen)
    resumen_df = resumen_df.sort_values(by='porcentaje_sum_null_values', ascending=False).reset_index(drop=True)
    
    return resumen_df

# %%
 
def cramers_v(confusion_matrix):
    """ 
    Calculate Cramer's V statistic for categorical-categorical association.
    Uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    """
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]  # Cambio aquí para usar scipy.stats directamente
    n = confusion_matrix.sum().sum()  # Total de observaciones
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

#%%
# Función para calcular WOE e IV para variables categóricas
def calculate_woe_iv_cat(df, feature, target):
    """
    Calcula el Weight of Evidence (WoE) y el Information Value (IV) para una variable categórica.
    
    Args:
        df (DataFrame): DataFrame que contiene los datos.
        feature (str): Nombre de la variable categórica.
        target (str): Nombre de la variable objetivo (debe ser binaria: 0/1).

    Returns:
        DataFrame: Tabla con los valores de WoE e IV para cada categoría y el IV total.
    """
    # Crear una tabla de contingencia para la variable actual
    grouped = df.groupby(feature)[target].value_counts().unstack(fill_value=0)
    
    # Calcular las proporciones de buenos y malos por cada categoría
    grouped['good_pct'] = grouped[1] / grouped[1].sum()
    grouped['bad_pct'] = grouped[0] / grouped[0].sum()

    # Agregar un pequeño valor (epsilon) para evitar división por 0
    epsilon = 1e-6
    grouped['good_pct'] += epsilon
    grouped['bad_pct'] += epsilon

    # Calcular el WOE
    grouped['WOE'] = np.log(grouped['bad_pct'] / grouped['good_pct'])

    # Calcular el IV para cada categoría
    grouped['IV'] = (grouped['bad_pct'] - grouped['good_pct']) * grouped['WOE']

    # Calcular el IV total
    iv_total = grouped['IV'].sum()

    # Agregar una columna con el nombre de la variable y el IV total
    grouped['Feature'] = feature
    grouped['IV_Total'] = iv_total

    return grouped[['WOE', 'IV', 'Feature', 'IV_Total']]

#%%
# Función para calcular WOE e IV para variables continuas
def calculate_woe_iv_continuous(df, feature, target, bins=10, epsilon=1e-6):
    """
    Calcula el Weight of Evidence (WoE) y el Information Value (IV) para una variable continua.
    
    Args:
        df (DataFrame): DataFrame que contiene los datos.
        feature (str): Nombre de la variable continua.
        target (str): Nombre de la variable objetivo (debe ser binaria: 0/1).
        bins (int): Número de intervalos (bins) en los que se dividirá la variable continua.
        epsilon (float): Valor pequeño para evitar divisiones por cero.

    Returns:
        DataFrame: Tabla con los valores de WoE e IV para cada intervalo y el IV total.
    """
    # Dividir la variable continua en bins
    df['bin'] = pd.cut(df[feature], bins=bins)

    # Agrupar por los bins y contar los eventos buenos y malos
    grouped = df.groupby('bin')[target].value_counts().unstack(fill_value=0)

    # Calcular las proporciones de buenos y malos por cada bin
    grouped['good_pct'] = grouped[1] / grouped[1].sum()
    grouped['bad_pct'] = grouped[0] / grouped[0].sum()

    # Agregar un pequeño valor (epsilon) para evitar división por 0
    grouped['good_pct'] += epsilon
    grouped['bad_pct'] += epsilon

    # Calcular el WOE
    grouped['WOE'] = np.log(grouped['bad_pct'] / grouped['good_pct'])

    # Calcular el IV para cada bin
    grouped['IV'] = (grouped['bad_pct'] - grouped['good_pct']) * grouped['WOE']

    # Calcular el IV total
    iv_total = grouped['IV'].sum()

    # Agregar una columna con el nombre de la variable y el IV total
    grouped['Feature'] = feature
    grouped['IV_Total'] = iv_total

    # Filtrar bins con IV pequeño para limpieza
    grouped = grouped[grouped['IV'] > 0]

    return grouped[['WOE', 'IV', 'Feature', 'IV_Total']]

#%%
def calcular_tabla_resumen(df, outlier_threshold=3):
    """
    Genera una tabla resumen con las proporciones de valores nulos, valores atípicos y otros datos relevantes.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        outlier_threshold (float): Umbral para definir valores atípicos (basado en desviaciones estándar).
        
    Returns:
        pd.DataFrame: Tabla resumen con las columnas especificadas.
    """
    resumen = []
    n_rows = len(df)
    
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):  # Solo para columnas numéricas
            # Cálculo de outliers usando la regla de 3 sigma
            mean = df[col].mean()
            std = df[col].std()
            lower_limit = mean - outlier_threshold * std
            upper_limit = mean + outlier_threshold * std
            outlier_count = df[col][(df[col] < lower_limit) | (df[col] > upper_limit)].count()
            
            # Cálculo de valores nulos
            null_count = df[col].isnull().sum()
            
            resumen.append({
                '0.0': 1 - (null_count / n_rows),  # Proporción no nula
                '1.0': null_count / n_rows,       # Proporción nula
                'variable': col,
                'sum_outlier_values': outlier_count,
                'porcentaje_sum_null_values': null_count / n_rows
            })
    
    # Convertir a DataFrame
    resumen_df = pd.DataFrame(resumen)
    resumen_df = resumen_df.sort_values(by='porcentaje_sum_null_values', ascending=False).reset_index(drop=True)
    
    return resumen_df

#%%
def reemplazar_nulos_por_desconocido(df):
    """
    Reemplaza los valores nulos en el DataFrame por "desconocido".
    
    :param df: DataFrame con los datos a procesar.
    :return: DataFrame con los valores nulos reemplazados por "desconocido".
    """
    return df.fillna("desconocido", inplace=True)


    #%%
def get_deviation_of_mean_perc(df, list_var_continuous, target, multiplier):
    """
    Devuelve un DataFrame que muestra el porcentaje de valores que exceden el intervalo de confianza,
    junto con la distribución del target para esos valores.
    
    :param df: DataFrame con los datos a analizar.
    :param list_var_continuous: Lista de variables continuas (e.g., columnas_numericas_sin_booleanas).
    :param target: Variable objetivo para analizar la distribución de categorías.
    :param multiplier: Factor multiplicativo para determinar el rango de confianza (media ± multiplier * std).
    :return: DataFrame con las proporciones del target para los valores fuera del rango de confianza.
    """


    result = []  # Lista para almacenar los resultados finales
    
    for var in list_var_continuous:
        # Calcular la media y desviación estándar de la variable
        mean = df[var].mean()
        std = df[var].std()
        
        # Calcular los límites de confianza
        lower_limit = mean - multiplier * std
        upper_limit = mean + multiplier * std
        
        # Filtrar valores fuera del rango
        outliers = df[(df[var] < lower_limit) | (df[var] > upper_limit)]
        
        # Si hay outliers, calcular las proporciones del target
        if not outliers.empty:
            proportions = outliers[target].value_counts(normalize=True)
            proportions = proportions.to_dict()  # Convertir a diccionario para facilitar su uso
            
            # Almacenar la información en una lista
            result.append({
                'variable': var,
                'sum_outlier_values': outliers.shape[0],
                'porcentaje_sum_null_values': outliers.shape[0] / len(df),
                **proportions  # Añadir las proporciones del target
            })
    
    # Si no se encontró ningún outlier, mostrar mensaje
    if not result:
        print('No existen variables con valores fuera del rango de confianza')
    
    # Convertir el resultado en un DataFrame
    result_df = type(df)(result)
    
    return result_df.sort_values(by='sum_outlier_values', ascending=False)
