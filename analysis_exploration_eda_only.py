#%%
import pandas as pd
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
#%% Prueba de comparar EDA_Clean_normalized promedio, entre exps

def plot_avg_exp(df, señal, save=False):
    # Agrupar por sujeto y experimento
    grouped = df.groupby(['subject', 'exp'])

    # Separar por experimento
    exp_data = {}
    for exp in df['exp'].unique():
        exp_signals = []
        for (subject, exp_i), df_sub in grouped:
            if exp_i == exp:
                exp_signals.append(df_sub[señal].values)
        
        # Alinear señales por padding con NaNs
        max_len = max(len(sig) for sig in exp_signals)
        aligned_signals = np.array([np.pad(sig, (0, max_len - len(sig)), constant_values=np.nan) for sig in exp_signals])
        
        # Calcular promedio ignorando NaNs
        avg_signal = np.nanmean(aligned_signals, axis=0)
        exp_data[exp] = avg_signal

        # Plot opcional
        plt.plot(avg_signal, label=f'Promedio {exp}')

    plt.title(f"Promedio de {señal} por experimento")
    plt.xlabel("Tiempo (muestras)")
    plt.legend()
    if save:
        plt.savefig(f"plots/{señal}_avg_experiment.png")

    plt.show()

    
def plot_exps_by_subject(df, señal, save=False):
    # Agrupar por sujeto y experimento
    grouped = df.groupby(['subject', 'exp'])

    # Crear figura por sujeto
    for subject in df['subject'].unique():
        plt.figure(figsize=(10, 4))
        plt.title(f"{señal} - Sujeto {subject}")
        plt.xlabel("Tiempo (muestras)")

        for exp in df['exp'].unique():
            try:
                signal = grouped.get_group((subject, exp))[señal].values
                plt.plot(signal, label=f"{exp}")
            except KeyError:
                # Este sujeto no tiene datos para este experimento
                continue

        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f"plots/{señal}_subject_{subject}.png")
        
        plt.show()

#%%
# Definir señal a graficar y cargar datos
señal = 'EDA_Clean_normalized'
df = pd.read_csv("datos_physio/eda_all_subjects_full_exps.csv")

plot_exps_by_subject(df, señal)
plot_avg_exp(df, señal)

# %%
