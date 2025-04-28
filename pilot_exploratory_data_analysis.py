#%%
import pandas as pd
import numpy as np
import os
import matplotlib
%matplotlib qt
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk

#%% Seleccionar el archivo que queres (WIP)
# Leer los datos    
data = np.loadtxt('datos_physio/S2/.txt')
df = pd.DataFrame(data)
df_beh = pd.read_csv("df_beh_S2.csv").drop("Unnamed: 0",axis=1)

# %% Crear un MNE.raw

info_canales = pd.read_excel("Informacion canales eeg (mapa canales).xlsx")

ch_names = list(info_canales["labels"])

ch_types = ['eeg'] * data.shape[0]
ch_types[-2] = 'eog'
ch_types[-1] = 'eog'
ch_types[21] = 'ecg'
ch_types[20] = 'stim'
ch_types[19] = 'ecg'

info = mne.create_info(ch_names=ch_names,
                       sfreq=256,
                       ch_types=ch_types)

raw = mne.io.RawArray(data, info)
raw.drop_channels(['EKG'])

raw_nk = nk.mne_to_df(raw)

scalings = {'eeg': 100, 'eog': 100}

raw.plot(scalings=scalings)

# %% Probando eventos a partir de marcadores

canal_stim = raw_nk["EXT1"]

# inter_min = 440 porque sino te toma la subida y la bajada
# como marcador

event_conditions = ["Prime","Target"]*(524//2)

events_dict = nk.events_find(canal_stim,
                             inter_min=440,
                             event_conditions=event_conditions)

# Ploteo entero
plt.figure()
plt.plot(canal_stim)
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

#%% Preprocesamiento (WIP)
# 2. Filtrado de la señal
raw.filter(l_freq=0.1, h_freq=40, fir_design='firwin')  # Filtrado pasa-banda (0.1-40 Hz)

# 3. Detección y eliminación de ruido
raw.notch_filter(freqs=[50, 60])  # Elimina ruido de la red eléctrica (50/60Hz)

raw.plot(scalings=scalings)
#%% Usar MNE para ver por epochs
event_dict, event_id = mne.events_from_annotations(raw)

tmin, tmax = -0.2, 0.8  # 200ms antes, 800ms después

epochs = mne.Epochs(
    raw, event_dict, event_id=event_id,
    tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True
)

# Verificar estructura
print(epochs)

evoked = epochs.average()
evoked.plot()


#%% Poner anotaciones según la condición
# Tiene que ser una de ['congruente', 'valencia_rostro',
#  'valencia_palabra', 'arousal_palabra', 'rostro_sexo']
condicion = "congruente"

onset_times = raw.annotations.onset

annotations = mne.Annotations(onset=onset_times, 
                              duration=[0] * len(onset_times),
                              description=df_beh[f'{condicion}'])

raw.set_annotations(annotations)

raw.plot(scalings=scalings)

# %% Dividir por experimento

split_time_exp_1 = annotations.onset[256]

# Crear los subsets
raw_subset1 = raw.copy().crop(tmin=0, tmax=split_time_exp_1-1)
raw_subset2 = raw.copy().crop(tmin=split_time_exp_1, tmax=raw.times[-1])

subsets = [raw_subset1, raw_subset2]

#%% Sacar ERP para el GFP (es decir todos los canales) por exp
gfp_congruente = []
gfp_incongruente = []

# Definir la ventana de análisis
tmin, tmax = -0.2, 0.8  # 200ms antes, 800ms después

# Dividir las etiquetas de comportamiento en dos partes
df_beh_subset1 = df_beh.iloc[:len(subsets[0].annotations)]
df_beh_subset2 = df_beh.iloc[len(subsets[0].annotations):]

df_beh_subsets = [df_beh_subset1, df_beh_subset2]

for i, (subset, df_beh_subset) in enumerate(zip(subsets, df_beh_subsets)):
    print(f"Procesando subset {i+1}")

    # Extraer eventos y anotaciones
    event_dict, event_id = mne.events_from_annotations(subset)
    
    # Crear las epochs
    epochs = mne.Epochs(
        subset, event_dict, event_id=event_id,
        tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True
    )

    # Separar las condiciones asegurando que la longitud coincida
    congruente_mask = df_beh_subset["congruente"].values.astype(bool)[:len(epochs)]
    incongruente_mask = ~congruente_mask  # Complementario

    epochs_cong = epochs[congruente_mask]
    epochs_incong = epochs[incongruente_mask]

    # Calcular GFP (Global Field Power) como la desviación estándar en cada tiempo
    gfp_cong = np.std(epochs_cong.get_data(), axis=1).mean(axis=0)
    gfp_incong = np.std(epochs_incong.get_data(), axis=1).mean(axis=0)

    gfp_congruente.append(gfp_cong)
    gfp_incongruente.append(gfp_incong)

    # Graficar GFP para cada condición
    times = epochs.times

    plt.figure(figsize=(8, 5))
    plt.plot(times, gfp_cong, label="Congruente", color='blue')
    plt.plot(times, gfp_incong, label="Incongruente", color='red')
    plt.axvline(x=0, color='k', linestyle='--', label="Estímulo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Global Field Power (GFP)")
    plt.title(f"GFP para subset {i+1}")
    plt.legend()
    plt.show()

#%% Sacar ERP para un canal en específico por exp
canal_interes = "cz"

erp_congruente = []
erp_incongruente = []

# Definir la ventana de análisis
tmin, tmax = -0.2, 0.8  # 200ms antes, 800ms después

# Dividir las etiquetas de comportamiento en dos partes
df_beh_subset1 = df_beh.iloc[:len(subsets[0].annotations)]
df_beh_subset2 = df_beh.iloc[len(subsets[0].annotations):]

df_beh_subsets = [df_beh_subset1, df_beh_subset2]

for i, (subset, df_beh_subset) in enumerate(zip(subsets, df_beh_subsets)):
    print(f"Procesando subset {i+1}")

    # Extraer eventos y anotaciones
    event_dict, event_id = mne.events_from_annotations(subset)
    
    # Crear las epochs
    epochs = mne.Epochs(
        subset, event_dict, event_id=event_id,
        tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True
    )

    # Seleccionar solo el canal de interés
    epochs = epochs.pick_channels([canal_interes])

    # Separar las condiciones asegurando que la longitud coincida
    congruente_mask = df_beh_subset["congruente"].values.astype(bool)[:len(epochs)]
    incongruente_mask = ~congruente_mask  # Complementario

    epochs_cong = epochs[congruente_mask]
    epochs_incong = epochs[incongruente_mask]

    # Calcular ERP promediado en el canal de interés
    erp_cong = epochs_cong.average().data.flatten()
    erp_incong = epochs_incong.average().data.flatten()

    erp_congruente.append(erp_cong)
    erp_incongruente.append(erp_incong)

    # Graficar ERP para cada condición
    times = epochs.times

    plt.figure(figsize=(8, 5))
    plt.plot(times, erp_cong, label="Congruente", color='blue')
    plt.plot(times, erp_incong, label="Incongruente", color='red')
    plt.axvline(x=0, color='k', linestyle='--', label="Estímulo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Voltaje (µV)")
    plt.title(f"ERP en {canal_interes} para subset {i+1}")
    plt.legend()
    plt.show()

# %%
