#%%
import pandas as pd
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
import scipy.stats as stats
import os
import seaborn as sns

#%% Prueba de comparar EDA_Clean_normalized promedio, entre exps

def test_frequency_before_cleaning(df, max_freq=0.3):
    figures = {}
    for subject, df_sub in df.groupby("subject"):
        # Concatenar la señal cruda de todos los experimentos del sujeto
        eda_raw = df_sub["EDA_Raw"].to_numpy(dtype=float)

        # Calcular el espectro simpático con NeuroKit2
        # show=True genera el gráfico automáticamente
        _, info = nk.eda_sympathetic(
            eda_raw,
            sampling_rate=256,
            frequency_band=[0.001, max_freq],
            method="posada",
            show=True
        )

        # Capturar la figura actual
        fig = plt.gcf()
        figures[subject] = fig

    return figures

def check_errors_qqplot(df):
    grouped = df.groupby(['subject', 'exp', 'block'])

    for (subject, exp, block), df_sub in grouped:
        plt.figure(figsize=(6, 6))
        stats.probplot(df_sub['error'], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot - Sujeto: {subject}, Exp: {exp}, Bloque: {block+1}')
        plt.tight_layout()
        plt.show()

def check_error_boxplot(df):
    grouped = df.groupby(['subject', 'exp', 'block'])

    for (subject, exp, block), df_sub in grouped:
        plt.figure(figsize=(6, 4))
        sns.violinplot(y=df_sub['error'], color="skyblue")
        plt.title(f'Violin Plot - Sujeto: {subject}, Exp: {exp}, Bloque: {block+1}')
        plt.ylabel('Error')
        plt.tight_layout()
        plt.show()

def check_error_histogram(df):
    grouped = df.groupby(['subject', 'exp', 'block'])

    for (subject, exp, block), df_sub in grouped:
        plt.figure(figsize=(8, 4))
        sns.histplot(df_sub['error'], kde=True, color='blue', alpha=0.5, bins=30)
        plt.title(f'Distribución de Error - Sujeto: {subject}, Exp: {exp}, Bloque: {block+1}')
        plt.xlabel('Error')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        plt.show()

# Función para formatear el p-valor
def format_p_value(p):
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    elif p < 0.1:
        return f"{p:.2f}"
    else:
        return f"{p:.2f}"

def plot_error_distributions(df):
    grouped = df.groupby(['subject', 'exp', 'block'])

    for (subject, exp, block), df_sub in grouped:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        data = df_sub['error'].dropna()
        mu, sigma = data.mean(), data.std()

        # Histograma + KDE + curva normal teórica
        sns.histplot(data, kde=True, color='steelblue', ax=axes[0], bins=30, alpha=0.6, stat='density')
        x = np.linspace(data.min(), data.max(), 1000)
        y = stats.norm.pdf(x, mu, sigma)
        axes[0].plot(x, y, color='black', linestyle='--', alpha=0.6, label='Distribución normal')
        axes[0].legend()
        axes[0].set_title('Histograma + KDE + Normal teórica')
        axes[0].set_xlabel('Error')

        # Q-Q Plot
        stats.probplot(data, dist="norm", plot=axes[1])
        axes[1].get_lines()[1].set_color('red')
        axes[1].set_title('Q-Q Plot')

        # Violinplot + Boxplot sin puntos
        sns.violinplot(y=data, ax=axes[2], color='lightblue', inner=None)
        sns.boxplot(y=data, ax=axes[2], width=0.15, showcaps=True,
                    boxprops={'facecolor':'none', 'linewidth':2},
                    whiskerprops={'linewidth':2},
                    medianprops={'color':'black', 'linewidth':2},
                    showfliers=False)
        axes[2].set_title('Violin + Boxplot')
        axes[2].set_ylabel('Error')

        # Test de normalidad: Kolmogorov-Smirnov
        if len(data) >= 3:
            ks_stat, p_val = stats.kstest(data, 'norm', args=(mu, sigma))
            p_formatted = format_p_value(p_val)

            # Color del p según resultado
            color_p = 'red' if p_val < 0.05 else 'green'

            # Texto separado por partes para formatear color del p
            fig.text(0.82, 0.95, 'Kolmogorov-Smirnov:', fontsize=12, ha='left', va='top',
                     bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))
            fig.text(0.82, 0.91, f'D = {ks_stat:.3f}', fontsize=12, ha='left', va='top')
            fig.text(0.82, 0.87, f'p = {p_formatted}', fontsize=12, ha='left', va='top', color=color_p)
        else:
            fig.text(0.82, 0.95, 'Kolmogorov-Smirnov:\nNo aplica (n<3)', fontsize=12,
                     ha='left', va='top', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))

        # Título general
        fig.suptitle(f'Sujeto: {subject} | Exp: {exp} | Bloque: {block+1}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()

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
señal = 'EDA_Tonic'
df = pd.read_csv("datos_physio/eda_all_subjects_full_exps.csv")

plot_exps_by_subject(df, señal)
plot_avg_exp(df, señal)

# %%