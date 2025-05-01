#%%
import pandas as pd
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
#%% Prueba de comparar EDA_Clean_normalized promedio, entre exps

# Cargar datos
df = pd.read_csv("datos_physio/eda_all_subjects_full_exps.csv")

# Verificar columnas disponibles
print(df.columns)

# Asegurar que 'EDA_Clean', 'subject' y 'exp' existen
assert 'EDA_Clean' in df.columns and 'subject' in df.columns and 'exp' in df.columns

# Agrupar por sujeto y experimento
grouped = df.groupby(['subject', 'exp'])

# Separar por experimento
exp_data = {}
for exp in df['exp'].unique():
    exp_signals = []
    for (subject, exp_i), df_sub in grouped:
        if exp_i == exp:
            exp_signals.append(df_sub['EDA_Clean_normalized'].values)
    
    # Alinear señales por padding con NaNs
    max_len = max(len(sig) for sig in exp_signals)
    aligned_signals = np.array([np.pad(sig, (0, max_len - len(sig)), constant_values=np.nan) for sig in exp_signals])
    
    # Calcular promedio ignorando NaNs
    avg_signal = np.nanmean(aligned_signals, axis=0)
    exp_data[exp] = avg_signal

    # Plot opcional
    plt.plot(avg_signal, label=f'Promedio {exp}')

plt.title("Promedio de EDA por experimento")
plt.xlabel("Tiempo (muestras)")
plt.ylabel("EDA Clean")
plt.legend()
plt.show()

# %%
