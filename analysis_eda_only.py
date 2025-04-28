#%%
import pandas as pd
import numpy as np
import os
import matplotlib
%matplotlib qt
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk

#%% Configuración de MNE
info_canales = pd.read_excel("Informacion canales eeg (mapa canales).xlsx")

ch_names = list(info_canales["labels"])

ch_types = ['eeg'] * 35 # = data.shape[0]
ch_types[-2] = 'eog'
ch_types[-1] = 'eog'
ch_types[21] = 'gsr'
ch_names[21] = 'eda'
ch_types[20] = 'stim'
ch_names[20] = 'estimulo'
ch_types[19] = 'ecg'

#%% Leer los datos 
carpeta = "datos_physio"
subjects = [archivo for archivo in os.listdir(carpeta) if not archivo.endswith(".csv")]

list_dfs = []

for subject in subjects:
    if subject == "S4":
        continue
    
    # Cargar archivos fisiológicos
    archivo_txt = [archivo for archivo in os.listdir(f'datos_physio/{subject}') if archivo.endswith(".TXT") and "resting" not in archivo.lower()][0]
    data = np.loadtxt(f'{carpeta}/{subject}/{archivo_txt}')
    df_physio = pd.DataFrame(data).T
    df_physio.columns = ch_names

    # Cargar archivos comportamentales y crear eventos
    df_beh = pd.read_csv(f"data/{subject}/df_beh_{subject}.csv").drop("Unnamed: 0",axis=1)
    canal_stim = df_physio["EXT1"]
    event_conditions = ["estimulo"]*(524)
    events_dict = nk.events_find(canal_stim,
                             inter_min=440,
                             event_conditions=event_conditions)
    df_beh["onset"] = events_dict["onset"][12:]

    list_dfs.append((df_physio, df_beh))

#%% Solo para verificar que los eventos están bien
# Ploteo de los eventos

for i, (df_physio, df_beh) in enumerate(list_dfs):
    plt.figure()
    plt.plot(df_physio.eda, label=df_beh["subject"][0])
    for n in range(len(df_beh["onset"])):
        plt.axvline(x=df_beh["onset"][n], color='red', linestyle='--', linewidth=2)
    plt.title(df_beh["subject"][0])
    plt.legend()
    plt.show()


#%%


# %% Probando eventos a partir de marcadores



# Ploteo entero
plt.figure()
plt.plot(canal_stim, label="Estimulo")
for x in events_dict["onset"]:
    plt.axvline(x=x, color='red', linestyle='--', linewidth=2)
plt.show()

#%% Crear las anotaciones
onset_times = events_dict["onset"]/raw.info["sfreq"]

annotations = mne.Annotations(onset=onset_times, 
                              duration=[0] * len(events_dict["onset"]),
                              description=["estímulo"] * len(events_dict["onset"]))

raw.set_annotations(annotations)

raw.plot(scalings=scalings)

# Crear un nuevo objeto de anotaciones excluyendo las primeras 12
new_annotations = mne.Annotations(
    onset=annotations.onset[12:],  # Omitir los primeros 12 onsets
    duration=annotations.duration[12:],  # Omitir las primeras 12 duraciones
    description=annotations.description[12:]  # Omitir las primeras 12 descripciones
)

# Asignar las nuevas anotaciones al raw
raw.set_annotations(new_annotations)

raw.plot(scalings=scalings)

