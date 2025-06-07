#%%
"""
Validation test for GSR device
This script validates the GSR device by segmenting the EDA signal into 
blocks based on different solutions applied to the skin.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
%matplotlib qt
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import cvxopt as cv
from termcolor import colored

# Configuración de los canales
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

if not os.path.exists("plots/testing_device"):
    os.makedirs("plots/testing_device")

# Cargar los datos de prueba
data = np.loadtxt(f'pruebaSCR3.TXT')
df_physio = pd.DataFrame(data).T
df_physio.columns = ch_names

dict_blocks = {
    0: "Agua destilada",
    1: "Solución fisiológica",
    2: "Gel conductor",
    3: "Dedos control"
}

# Segmentar los datos en bloques según solución
block_1 = df_physio.eda[0:14000]
block_2 = df_physio.eda[28500:50400]
block_3 = df_physio.eda[57500:78750]
block_4 = df_physio.eda[110000:127200]

# Plot señal y ver bloques
plt.figure()
plt.plot(df_physio.eda)
plt.plot(block_1, label=dict_blocks[0])
plt.plot(block_2, label=dict_blocks[1])
plt.plot(block_3, label=dict_blocks[2])
plt.plot(block_4, label=dict_blocks[3])
plt.legend()
plt.title("Señal de prueba con segmentación")
plt.savefig("plots/testing_device/signal_and_blocks.png")
plt.show()

# Plot raw vs clean por bloque
blocks = [block_1, block_2, block_3, block_4]
for n, block in enumerate(blocks):
    x = np.arange(len(block))
    eda_clean = nk.eda_clean(block, sampling_rate=256, method="BioSPPy")
    plt.figure()
    plt.plot(x,block, label="EDA Raw")
    plt.plot(x,eda_clean, label="EDA Clean")
    plt.title(dict_blocks[n])
    plt.legend()
    plt.savefig(f'plots/testing_device/raw_vs_clean_{dict_blocks[n]}.png')
    plt.show()
# %%
