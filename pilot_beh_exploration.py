#%%
import pandas as pd
import numpy as np
import os
import matplotlib
%matplotlib qt

import matplotlib.pyplot as plt
from scipy.stats import norm
#%% Funciones para después

# Todo acá abajo es para generar la columna "label"
def cod_valencia_rostro(val):
    """
    'Po' si es 'positiva', 'Ne' si es 'negativa'
    """
    if val == "positiva":
        return "Po"
    elif val == "negativa":
        return "Ne"
    else:
        return "DesconocidoVal"

def cod_arousal_palabra(val):
    """
    'Baj' si es 'bajo', 'Alt' si es 'alto'
    """
    if val == "bajo":
        return "Baj"
    elif val == "alto":
        return "Alt"
    else:
        return "DesconocidoArousal"

def cod_congruencia(val):
    """
    'Con' si es 'congruente', 'Inc' si es 'incongruente'
    """
    if val == "congruente":
        return "Con"
    elif val == "incongruente":
        return "Inc"
    else:
        return "DesconocidoCong"

def cod_acierto(val):
    """
    'Ok' si es acierto, 'Mal' si es error, 'Nan' si es omision
    """
    if val == "acierto":
        return "Ok"
    elif val == "error":
        return "Mal"
    elif val == "omision":
        return "Nan"
    else:
        return "DesconocidoAcierto"

def cod_sexo_rostro(val):
    """
    'M' si es mujer, 'H' si es hombre
    """
    if val == "mujer":
        return "M"
    elif val == "hombre":
        return "H"
    else:
        return "DesconocidoSexo"

def cod_bloque_experimental(val):
    """
    'Etq' si es 'etiqueta', 'Em' si es 'emocionalmente activante'
    """
    if val == "etiqueta":
        return "Etq"
    elif val == "emocionalmente_activantes":
        return "Em"
    else:
        return "DesconocidoBloque"

# Función para calcular métricas agrupadas
def agrupar_por_condicion(df_beh, condicion):
    """
    Realiza un groupby sobre 'condicion' (puede ser una columna o una lista de columnas),
    y devuelve las métricas definidas (rt medio, desv, % aciertos/errores/omisiones).
    """
    # Si 'condicion' es un string, lo convertimos en lista para que funcione de forma genérica
    if isinstance(condicion, str):
        condicion = [condicion]
        
    df_por_condicion = df_beh.groupby(condicion).agg(
        n=('palabra', 'count'),
        rt_medio=('rt_raw', 'mean'),
        rt_desv=('rt_raw', 'std'),
        pct_aciertos=('acierto', lambda x: (x == "acierto").mean() * 100),
        pct_errores=('acierto', lambda x: (x == "error").mean() * 100),
        pct_omisiones=('acierto', lambda x: (x == "omision").mean() * 100)).reset_index()

    df_por_condicion["label"] = df_por_condicion["exp"]+"_"+df_por_condicion["congruente"]+"_"+df_por_condicion["valencia_palabra"]

    last_col_name = df_por_condicion.columns[-1]
    last_col_data = df_por_condicion.pop(last_col_name)
    df_por_condicion.insert(0, last_col_name, last_col_data)

    df_por_condicion.drop(columns=['exp','congruente','valencia_palabra'], inplace=True)
    
    df_por_condicion["subject"] = subject

    return df_por_condicion

#%% Leer los datos comportamentales
carpeta = "data"
subjects = [archivo for archivo in os.listdir(carpeta) if not archivo.endswith(".csv")]

# Definir las condiciones para el análisis
condicion = ['exp','congruente','valencia_palabra']


for subject in subjects:
    archivos_excel = [archivo for archivo in os.listdir(f'data/{subject}') if archivo.endswith(".xlsx")]

    dfs = []

    # Leer todos los excels menos el de práctica, y ponerlos en uno solo
    for archivo in archivos_excel:
        if "practica" not in archivo:
            ruta_archivo = os.path.join(carpeta, subject, archivo)
            df = pd.read_excel(ruta_archivo, skipfooter=8)
            dfs.append(df)

    df_beh = pd.concat(dfs, ignore_index=True)  

    # Ordenar los estímulos como se fueron presentando (insumo para análisis EEG)
    if df_beh["orden_exp_raw"].unique() == np.array(["'2'"]):
        df_beh.sort_values(by=['exp', 'rep_raw', 'order'], 
        ascending=[True, True, True], inplace=True)

    else: 
        df_beh.sort_values(by=['exp', 'rep_raw', 'order'], 
        ascending=[False, True, True], inplace=True)

    # Preparación de las columnas faltantes de interés

    # Limpiar formato de algunas columnas
    df_beh['rt_raw'] = df_beh['rt_raw'].apply(lambda x: float(x.replace("'", "")) if isinstance(x, str) else x)
    df_beh['response_raw'] = df_beh['response_raw'].str.replace("'", "")

    # Crear una nueva columna de aciertos
    df_beh['acierto'] = df_beh.apply(
        lambda row: "acierto" if ((row['response_raw'] == 'right' and row['valencia_rostro'] == 'positiva') or
                                (row['response_raw'] == 'left'  and row['valencia_rostro'] == 'negativa'))
        else ("omision" if pd.isna(row['response_raw']) else "error"),
        axis=1)

    # Crear una nueva columna de sexo del rostro
    df_beh['rostro_sexo'] = df_beh.apply(
        lambda row: 'mujer' if 'mujer' in row['rostro'] else 'hombre', axis=1)

    # Modificar columna congruente (en vez de 1/0, congruente/incongruente)
    df_beh['congruente'] = df_beh.apply(
        lambda row: 'congruente' if row['congruente'] == 1 else 'incongruente', axis=1)

    df_beh["label"] = (
        df_beh["valencia_rostro"].apply(cod_valencia_rostro) +
        df_beh["arousal_palabra"].apply(cod_arousal_palabra) +
        df_beh["congruente"].apply(cod_congruencia) +
        df_beh["acierto"].apply(cod_acierto) +
        df_beh["rostro_sexo"].apply(cod_sexo_rostro) +
        df_beh["exp"].apply(cod_bloque_experimental)
    )

    df_beh["subject"] = subject

    df_beh.to_csv(f'data/{subject}/df_beh_{subject}.csv')

    df_por_condicion = agrupar_por_condicion(df_beh,condicion)

    df_por_condicion.to_csv(f'data/{subject}/df_por_condicion_{subject}.csv')


#%% Juntar todos los df_por_condicion para hacer análisis

carpeta = "data"
subjects = [archivo for archivo in os.listdir(carpeta) if not archivo.endswith(".csv")]

list_dfs_all_subjects = []

for subject in subjects:
    df_por_condicion = pd.read_csv(f'data/{subject}/df_por_condicion_{subject}.csv')

    list_dfs_all_subjects.append(df_por_condicion.copy())

df_all_subjects = pd.concat(list_dfs_all_subjects).drop("Unnamed: 0",axis=1)

df_all_subjects.to_csv("data/df_all_subjects_por_condicion.csv")

#%% Juntar todos los trials de los sujetos para hacer análisis

carpeta = "data"
subjects = [archivo for archivo in os.listdir(carpeta) if not archivo.endswith(".csv")]

list_dfs_all_subjects = []

for subject in subjects:
    df_full_trials = pd.read_csv(f'data/{subject}/df_beh_{subject}.csv')

    list_dfs_all_subjects.append(df_full_trials.copy())

df_all_subjects = pd.concat(list_dfs_all_subjects).drop("Unnamed: 0",axis=1)

df_all_subjects.to_csv("data/df_all_subjects_full_trials.csv")

#%% Funcion para hacer descriptivo por condiciones

def filtrate_and_mean(df, condiciones):
    """
    Devuelve un DataFrame que solo incluye las filas 
    donde la columna 'label' contiene el texto de condición.

    df: El df_all_subjects
    condicion: La condición única por la que quieras hacer un promedio

    """
    # Aseguramos que 'condiciones' sea una lista
    if isinstance(condiciones, str):
        condiciones = [condiciones]
    
    # Creamos una máscara booleana que inicialmente sea True para todas las filas
    mask = pd.Series([True] * len(df), index=df.index)
    
    # Para cada condición, ajustamos la máscara para que solo queden las filas
    # que contengan esa condición (subcadena) en la columna 'label'
    for cond in condiciones:
        mask = mask & df['label'].str.contains(cond, case=False, na=False)
    
    # Filtramos con la máscara final y calculamos la media (descartando ciertas columnas)
    df_filtrado = df[mask].copy()
    return df_filtrado.drop(columns=['label', 'subject', 'n']).mean()

print("-"*5+"Efecto Priming General"+"-"*5)
print("*Congruente")
print(filtrate_and_mean(df_all_subjects, "congruente"))
print("\n"+"*Incongruente")
print(filtrate_and_mean(df_all_subjects, "incongruente"))

print("\n"+"-"*5+"Efecto Priming Emocionalmente Act."+"-"*5)
print("*Congruente")
print(filtrate_and_mean(df_all_subjects, ["congruente","emocionalmente_activante"]))
print("\n"+"*Incongruente")
print(filtrate_and_mean(df_all_subjects, ["incongruente","emocionalmente_activante"]))

print("\n"+"-"*5+"Efecto Priming Etiqueta"+"-"*5)
print("*Congruente")
print(filtrate_and_mean(df_all_subjects, ["congruente","etiqueta"]))
print("\n"+"*Incongruente")
print(filtrate_and_mean(df_all_subjects, ["incongruente","etiqueta"]))

#%%


# %%
def compute_dprime_for_positive(df, 
                                col_emotion='valencia_rostro', 
                                col_response='response_raw',
                                val_positive='positiva',
                                val_right='right'):
    """
    Calcula d' y c considerando que:
      - 'rostro positivo' = 'señal presente'
      - 'rostro no positivo' (negativo o neutro) = 'señal ausente'
      - 'right' = "sí, detecto positivo"

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con una fila por ensayo. Debe incluir:
          - col_emotion: emoción real del estímulo en cada ensayo
          - col_response: respuesta dada por el participante
    col_emotion : str
        Nombre de la columna del DF que contiene la emoción real
    col_response : str
        Nombre de la columna del DF que contiene la respuesta del participante
    val_positive : str
        Valor en `col_emotion` que define "rostro positivo" (señal presente)
    val_right : str
        Valor en `col_response` que define "respuesta detectando positivo"

    Retorna:
    --------
    dprime : float
        Sensibilidad (d') para detectar "rostro positivo" frente a "no positivo"
    c_criterion : float
        Criterio de respuesta (c), indicador del sesgo a responder "positivo"
    """

    # 1. Filtrar ensayos con señal presente (rostros positivos) y ausente
    df_signal_present = df[df[col_emotion] == val_positive]
    df_signal_absent  = df[df[col_emotion] != val_positive]

    # 2. Contar hits y falsas alarmas
    # Hit = rostro positivo y respuesta = "right"
    hits = np.sum(df_signal_present[col_response] == val_right)
    # False Alarm = rostro no positivo y respuesta = "right"
    fa   = np.sum(df_signal_absent[col_response] == val_right)

    # 3. Calcular N totales
    n_signal_present = len(df_signal_present)  # total señales presentes
    n_signal_absent  = len(df_signal_absent)   # total señales ausentes

    # 4. Proporción de Hits y FAs
    #   Se aplica corrección de 0.5 y N+1 para evitar 0 o 1 exactos.
    #   p(Hit) = (hits + 0.5) / (n_signal_present + 1), etc.
    p_hit = (hits + 0.5) / (n_signal_present + 1) if n_signal_present > 0 else 0.0
    p_fa  = (fa + 0.5)   / (n_signal_absent + 1)  if n_signal_absent  > 0 else 0.0

    # 5. Convertir a valores z
    z_hit = norm.ppf(p_hit)  # inverse CDF
    z_fa  = norm.ppf(p_fa)

    # 6. d' = Z(H) - Z(FA)
    dprime = z_hit - z_fa

    # 7. Criterio c = -0.5 * (Z(H) + Z(FA))
    c_criterion = -0.5 * (z_hit + z_fa)

    return dprime, c_criterion

# Cambiar acá la condicion para ver comparación 
# de d' y c según el valor de esa condición
condicion = "congruente"

for valor in df_beh[f'{condicion}'].unique():
    dprime, ccrit = compute_dprime_for_positive(df_beh[df_beh[f'{condicion}']==valor])
    print(f"d' {valor} = {dprime:.2f}")
    print(f"criterio c {valor} = {ccrit:.2f}")

# %%
