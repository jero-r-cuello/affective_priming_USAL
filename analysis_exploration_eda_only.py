#%%
import pandas as pd
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
import scipy.stats as stats
import os
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats import weightstats as st
import pingouin as pg
from scipy.stats import levene
from scipy.stats import wilcoxon
from termcolor import colored


#%% Definición de funciones

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

# Función para formatear el p-valor, para gráficos más que nada
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

def plot_avg_exp(df, señales, save=False):
    """
    Plotea el promedio de la señal por experimento.
    """
    # Agrupar por sujeto y experimento
    grouped = df.groupby(['subject', 'exp'])

    for sig in señales:
        plt.figure(figsize=(10, 4))
        for exp in df['exp'].unique():
            # Reunir todas las señales de ese experimento
            exp_signals = [
                df_sub[sig].values
                for (subject, exp_i), df_sub in grouped
                if exp_i == exp
            ]
            if not exp_signals:
                continue

            max_len = max(map(len, exp_signals))
            aligned = np.array([
                np.pad(sig_v, (0, max_len - len(sig_v)),
                       constant_values=np.nan)
                for sig_v in exp_signals
            ])
            avg_signal = np.nanmean(aligned, axis=0)
            plt.plot(avg_signal, label=f'Promedio {exp}')

        plt.title(f"Promedio de {sig} por experimento")
        plt.xlabel("Tiempo (muestras)")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f"plots/{sig}_avg_experiment.png")
        plt.show()

def plot_each_block_by_subject(df, señales, save=False):
    """
    Plotea la señal de cada bloque, por sujeto
    (n plots = n sujetos * n bloques).
    """
    # Agrupar por sujeto y bloque
    grouped = df.groupby(['subject', 'exp', 'block'])

    for subject in df['subject'].unique():
        for exp in df['exp'].unique():
            for block in df['block'].unique():
                plt.figure(figsize=(10, 4))
                plt.title(f"Señales {', '.join(señales)} - Sujeto {subject} - Exp {exp}, B.{block+1}/4")
                plt.xlabel("Tiempo (muestras)")

                for sig in señales:
                    try:
                        signal = grouped.get_group((subject, exp, block))[sig].values
                        plt.plot(signal, label=f"{sig}")
                    except KeyError:
                        # No hay datos para esa combinación
                        continue

                plt.legend()
                plt.tight_layout()
                if save:
                    plt.savefig(
                        f"plots/{'_'.join(señales)}_subject_{subject}_exp_{exp}_block_{block+1}.png"
                    )
                plt.show()


def plot_exps_by_subject(df, señales, save=False):
    """
    Plotea la señal de ambos experimentos, por sujeto (n suj = n plots).
    """
    # Agrupar por sujeto y experimento
    grouped = df.groupby(['subject', 'exp'])

    for subject in df['subject'].unique():
        plt.figure(figsize=(10, 4))
        plt.title(f"Señales {', '.join(señales)} - Sujeto {subject}")
        plt.xlabel("Tiempo (muestras)")

        for exp in df['exp'].unique():
            for sig in señales:
                try:
                    signal = grouped.get_group((subject, exp))[sig].values
                    plt.plot(signal, label=f"{sig} - {exp}")
                except KeyError:
                    # No hay datos para esa combinación
                    continue

        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(
                f"plots/{'_'.join(señales)}_subject_{subject}.png"
            )
        plt.show()

def check_markers_by_block(df, stimuli_channel='stimuli'):
    """
    Ver que todos los bloques tengan la cantidad de marcadores correcta.
    """
    # Agrupar por sujeto y bloque
    grouped = df.groupby(['subject', 'exp', 'block'])

    for subject in df['subject'].unique():
        for exp in df['exp'].unique():
            for block in df['block'].unique():
                # Obtener el número de marcadores en el bloque
                stim_signal = grouped.get_group((subject, exp, block))[stimuli_channel]
                eventos = nk.events_find(stim_signal)
                num_markers = len(eventos['onset'])

                print(f"Sujeto {subject}, Exp {exp}, Bloque {block+1}: {num_markers} marcadores")
                
#WIP 
def scr_event_analysis(df, df_beh, condicion, stimuli_channel='stimuli'):
    """
    Realiza un análisis de eventos en la señal especificada.
    Primero se extraen las características de SCR por evento,
    luego se realizan pruebas estadísticas para comparar.

    OJO: 
    Funciona solo para SCR_Peak_Amplitude y con una sola condición.
    Actualizar en un futuro.
    """

    # Extraer features relacionadas a SCR por evento
    all_features = pd.DataFrame()

    grouped = df.groupby(['subject', 'exp', 'block'])

    for subject in df['subject'].unique():
        for exp in df['exp'].unique():
            for i, block in enumerate(df['block'].unique()):
                df_physio = grouped.get_group((subject, exp, block))
                df_beh_exp = df_beh[(df_beh['subject'] == subject) & (df_beh['exp'] == exp)]
                
                # Dividir el df de comportamiento en bloques
                # (asumiendo que cada bloque tiene 64 muestras)
                if i == 0:
                    df_beh_block = df_beh_exp[:64]
                elif i == 1:
                    df_beh_block = df_beh_exp[64:128]
                elif i == 2:
                    df_beh_block = df_beh_exp[128:192]
                elif i == 3:
                    df_beh_block = df_beh_exp[192:]

                # Obtener los eventos de la señal de estímulo
                stim_signal = grouped.get_group((subject, exp, block))[stimuli_channel]
                eventos = nk.events_find(stim_signal,
                                         event_conditions=df_beh_block[condicion].to_list())
                # Generar las épocas y obtener features de scr
                epochs = nk.epochs_create(df_physio,
                                          events=eventos,
                                          sampling_rate=256,
                                          epochs_start=-2,
                                          epochs_end=6)
        
                features = nk.eda_eventrelated(epochs, silent=True)
                #!! Handleo los nans como 0, porque sino después no puedo hacer pruebas de supuestos
                features['SCR_Peak_Amplitude'] = features['SCR_Peak_Amplitude'].fillna(0)
                features['SCR_Peak_Amplitude_Time'] = features['SCR_Peak_Amplitude_Time'].fillna(0)
                features['SCR_RiseTime'] = features['SCR_RiseTime'].fillna(0)
                features['SCR_RecoveryTime'] = features['SCR_RecoveryTime'].fillna(0)

                all_features = pd.concat([all_features, features], axis=0)

    #!! ACÁ SE VIENE LA PARTE ESTADÍSTICA
    #!! HABRÍA QUE MODIFICAR PARA QUE SE PUEDA HACER CON DIFERENTES
    #!! FEATURES, PORQUE ESTÁ HECHO NO MÁS PARA SCR_PEAK_AMPLITUDE
    #!! Las muestras serían pareadas??? o independientes???
    #!! OJO QUE NO ESTÁ SEPARADO POR EXPERIMENTO, JUNTA TODO
    # Separo por condición
    features_by_condition = all_features.groupby('Condition')
    features_by_condition_list = []

    # Agarro los eventos de cada condición
    for condicion in all_features['Condition'].unique():
        features_by_condition_list.append(features_by_condition.get_group(condicion))
    condition_1 = features_by_condition_list[0]["SCR_Peak_Amplitude"]
    condition_2 = features_by_condition_list[1]["SCR_Peak_Amplitude"]

    # Chequeamos normalidad de cada condición
    #!! Quizás no sea necesario, por el Teorema del Límite Central
    normality = []

    cond_1_normality = pg.normality(condition_1).normal.values
    cond_2_normality = pg.normality(condition_2).normal.values
    bool_normality = cond_1_normality and cond_2_normality
    
    # Chequeamos homogeneidad de varianzas
    p_var = levene(condition_1, condition_2).pvalue
    bool_var = p_var >= 0.05

    # Chequeamos si hay diferencias entre condiciones
    # Elegimos el test según los resultados de normalidad y homogeneidad
    if bool_normality and bool_var:
        # Test t de Student
        t_stat, p_val = stats.ttest_ind(condition_1, condition_2)
        test_name = "t-test"
    elif not bool_normality and bool_var:
        # Test de Welch
        t_stat, p_val = stats.ttest_ind(condition_1, condition_2, equal_var=False)
        test_name = "Welch's t-test"
    elif not bool_normality and not bool_var:
        # Test de Wilcoxon
        t_stat, p_val = wilcoxon(condition_1, condition_2)
        test_name = "Wilcoxon signed-rank test"
    else:
        raise ValueError("No se puede determinar el test a utilizar.")

    print(colored(f"Test: {test_name}", "yellow"))
    print(colored(f"t-statistic: {t_stat}", "yellow"))
    print(colored(f"p-value: {p_val}", "yellow"))

    return test_name, t_stat, p_val, bool_normality, bool_var
                      
#%%
# Definir señal a graficar y cargar datos
señal = ['EDA_Phasic']
df = pd.read_csv("datos_physio/eda_all_subjects_full_exps.csv")
df_beh = pd.read_csv("data/df_all_subjects_full_trials.csv")
condicion = 'congruente' #['valencia_palabra', 'valencia_rostro', 'arousal_palabra', 'rostro_sexo']

plot_exps_by_subject(df, señal)
plot_avg_exp(df, señal)

plot_each_block_by_subject(df, señal)

scr_event_analysis(df, df_beh, condicion)
# %%