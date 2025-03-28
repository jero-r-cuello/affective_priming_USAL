#%%
import pandas as pd
import numpy as np
import os
import matplotlib
%matplotlib qt

import matplotlib.pyplot as plt
import neurokit2 as nk

#%% Leer los datos comportamentales

carpeta = "data/S2_piloto"

archivos_excel = [archivo for archivo in os.listdir(carpeta) if archivo.endswith(".xlsx")]

dfs = []

# Leer todos los excels menos el de práctica, y ponerlos en uno solo
for archivo in archivos_excel:
    if "practica" not in archivo:
        ruta_archivo = os.path.join(carpeta, archivo)
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

#%% Exploración del df
print(df_beh.head())

for columna in df_beh.columns:
    if columna != "palabra":
        print(df_beh[f'{columna}'].value_counts())
        print(" ")

# %%

df_beh = pd.read_csv("df_beh_S2.csv").drop("Unnamed: 0",axis=1)

df_beh['rt_raw'] = df_beh['rt_raw'].str.replace("'", "").astype(float)
df_beh['response_raw'] = df_beh['response_raw'].str.replace("'", "")

# Crear una nueva columna de aciertos (1 si la respuesta fue correcta, 0 si no)
df_beh['acierto'] = df_beh.apply(
    lambda row: "acierto" if ((row['response_raw'] == 'right' and row['valencia_rostro'] == 'positiva') or
                              (row['response_raw'] == 'left'  and row['valencia_rostro'] == 'negativa'))
    else ("omision" if pd.isna(row['response_raw']) else "error"),
    axis=1)

# Crear una nueva columna de sexo (para comprobar efectos por sexo del rostro)
df_beh['rostro_sexo'] = df_beh.apply(
    lambda row: 'mujer' if 'mujer' in row['rostro'] else 'hombre', axis=1)

# Modificar columna congruente (en vez de 1/0, congruente/incongruente)
df_beh['congruente'] = df_beh.apply(
    lambda row: 'congruente' if row['congruente'] == 1 else 'incongruente', axis=1)

df_beh.to_csv("df_beh_S2.csv")
#%%
# Definir las condiciones para el análisis
condiciones = ['congruente', 'valencia_rostro', 'valencia_palabra', 'arousal_palabra', 'rostro_sexo']

# Función para calcular métricas agrupadas
def calcular_metricas(df, agrupadores):
    return df.groupby(agrupadores).agg(
        rt_medio=('rt_raw', 'mean'),
        rt_desv=('rt_raw', 'std'),
        porcentaje_aciertos=('acierto', 'mean')
    ).reset_index()


resultados_exp = []

for condicion in condiciones:
    resultados_exp.append(calcular_metricas(df_beh, ['exp'] + [condicion]))

resultados_bloque = []

for condicion in condiciones:
    resultados_bloque.append(calcular_metricas(df_beh, ['bloque'] + [condicion]))

# %% Generar los códigos que pidio Lolo para Matlab

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

df_beh["label"] = (
    df_beh["valencia_rostro"].apply(cod_valencia_rostro) +
    df_beh["arousal_palabra"].apply(cod_arousal_palabra) +
    df_beh["congruente"].apply(cod_congruencia) +
    df_beh["acierto"].apply(cod_acierto) +
    df_beh["rostro_sexo"].apply(cod_sexo_rostro) +
    df_beh["exp"].apply(cod_bloque_experimental)
)
# %%
