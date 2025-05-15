#%%
import pandas as pd
import numpy as np
import os
import matplotlib
%matplotlib qt
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import cvxopt as cv

#%% Configuración de los canales
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

#%% Definición de funciones

# Función cvxEDA de pyEDA, no la pude importar
def cvxEDA_pyEDA(y, delta, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2,
           solver=None, options={'reltol':1e-9}):
    """CVXEDA Convex optimization approach to electrodermal activity processing
    This function implements the cvxEDA algorithm described in "cvxEDA: a
    Convex Optimization Approach to Electrodermal Activity Processing"
    (http://dx.doi.org/10.1109/TBME.2015.2474131, also available from the
    authors' homepages).
    Arguments:
       y: observed EDA signal (we recommend normalizing it: y = zscore(y))
       delta: sampling interval (in seconds) of y
       tau0: slow time constant of the Bateman function
       tau1: fast time constant of the Bateman function
       delta_knot: time between knots of the tonic spline function
       alpha: penalization for the sparse SMNA driver
       gamma: penalization for the tonic spline coefficients
       solver: sparse QP solver to be used, see cvxopt.solvers.qp
       options: solver options, see:
                http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
    Returns (see paper for details):
       r: phasic component
       p: sparse SMNA driver of phasic component
       t: tonic component
       l: coefficients of tonic spline
       d: offset and slope of the linear drift term
       e: model residuals
       obj: value of objective function being minimized (eq 15 of paper)
    """
    
    n = len(y)
    y = cv.matrix(y)

    # bateman ARMA model
    a1 = 1./min(tau1, tau0) # a1 > a0
    a0 = 1./max(tau1, tau0)
    ar = np.array([(a1*delta + 2.) * (a0*delta + 2.), 2.*a1*a0*delta**2 - 8.,
        (a1*delta - 2.) * (a0*delta - 2.)]) / ((a1 - a0) * delta**2)
    ma = np.array([1., 2., 1.])

    # matrices for ARMA model
    i = np.arange(2, n)
    A = cv.spmatrix(np.tile(ar, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))
    M = cv.spmatrix(np.tile(ma, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))

    # spline
    delta_knot_s = int(round(delta_knot / delta))
    spl = np.r_[np.arange(1.,delta_knot_s), np.arange(delta_knot_s, 0., -1.)] # order 1
    spl = np.convolve(spl, spl, 'full')
    spl /= max(spl)
    # matrix of spline regressors
    i = np.c_[np.arange(-(len(spl)//2), (len(spl)+1)//2)] + np.r_[np.arange(0, n, delta_knot_s)]
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl),1))
    p = np.tile(spl, (nB,1)).T
    valid = (i >= 0) & (i < n)
    B = cv.spmatrix(p[valid], i[valid], j[valid])

    # trend
    C = cv.matrix(np.c_[np.ones(n), np.arange(1., n+1.)/n])
    nC = C.size[1]

    # Solve the problem:
    # .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*p + .5*gamma*l'*l
    # s.t. A*q >= 0

    old_options = cv.solvers.options.copy()
    cv.solvers.options.clear()
    cv.solvers.options.update(options)
    if solver == 'conelp':
        # Use conelp
        z = lambda m,n: cv.spmatrix([],[],[],(m,n))
        G = cv.sparse([[-A,z(2,n),M,z(nB+2,n)],[z(n+2,nC),C,z(nB+2,nC)],
                    [z(n,1),-1,1,z(n+nB+2,1)],[z(2*n+2,1),-1,1,z(nB,1)],
                    [z(n+2,nB),B,z(2,nB),cv.spmatrix(1.0, range(nB), range(nB))]])
        h = cv.matrix([z(n,1),.5,.5,y,.5,.5,z(nB,1)])
        c = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T,z(nC,1),1,gamma,z(nB,1)])
        res = cv.solvers.conelp(c, G, h, dims={'l':n,'q':[n+2,nB+2],'s':[]})
        obj = res['primal objective']
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cv.sparse([[Mt*M, Ct*M, Bt*M], [Mt*C, Ct*C, Bt*C], 
                    [Mt*B, Ct*B, Bt*B+gamma*cv.spmatrix(1.0, range(nB), range(nB))]])
        f = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T - Mt*y,  -(Ct*y), -(Bt*y)])
        res = cv.solvers.qp(H, f, cv.spmatrix(-A.V, A.I, A.J, (n,len(f))),
                            cv.matrix(0., (n,1)), solver=solver)
        obj = res['primal objective'] + .5 * (y.T * y)
    cv.solvers.options.clear()
    cv.solvers.options.update(old_options)

    l = res['x'][-nB:]
    d = res['x'][n:n+nC]
    t = B*l + C*d
    q = res['x'][:n]
    p = A * q
    r = M * q
    e = y - r - t

    return (np.array(a).ravel() for a in (r, p, t, l, d, e, obj))

#%% Leer los datos 

carpeta = "datos_physio"
subjects = [archivo for archivo in os.listdir(carpeta) if not archivo.endswith(".csv") and not archivo.endswith(".TXT")]

# Sujetos excluídos junto con el motivo
excluded_subjects = {"S4": "El canal de estímulo está roto",
                     "S2": "Se sacó el scr a mitad del exp 1",
                     "S1": "No hay datos, creo que este solo fue una prueba"}

list_dfs = []

for subject in subjects:
    if subject in excluded_subjects.keys():
        continue

#    if os.path.exists(f"datos_physio/{subject}/df_eda_{subject}_emocionalmente_activantes.csv") and os.path.exists(f"datos_physio/{subject}/df_eda_{subject}_etiqueta.csv"):
#        print(f"Ya se procesó el sujeto {subject}")
#        continue

    
    # Cargar archivos fisiológicos
    archivo_txt = [archivo for archivo in os.listdir(f'datos_physio/{subject}') if archivo.endswith(".TXT") and "resting" not in archivo.lower()][0]
    data = np.loadtxt(f'{carpeta}/{subject}/{archivo_txt}')
    df_physio = pd.DataFrame(data).T
    df_physio.columns = ch_names

    # Cargar archivos comportamentales y crear eventos
    df_beh = pd.read_csv(f"data/{subject}/df_beh_{subject}.csv").drop("Unnamed: 0",axis=1)
    canal_stim = df_physio["estimulo"]
    event_conditions = ["estimulo"]*(524)
    events_dict = nk.events_find(canal_stim,
                             inter_min=440,
                             event_conditions=event_conditions)
    df_beh["onset"] = events_dict["onset"][12:]

    list_dfs.append((df_physio, df_beh))

#%% Solo para verificar que los eventos están bien
# Ploteo de los eventos
"""
for i, (df_physio, df_beh) in enumerate(list_dfs):
    plt.figure()
    plt.plot(df_physio.eda, label=df_beh["subject"][0])
    for n in range(len(df_beh["onset"])):
        plt.axvline(x=df_beh["onset"][n], color='red', linestyle='--', linewidth=2)
    plt.title(df_beh["subject"][0])
    plt.legend()
    plt.show()
"""
# EDA - Extracción de features

list_dfs_eda = []
list_coefs_ajuste = []

for i, (df_physio, df_beh) in enumerate(list_dfs):
    # Verificamos que el sujeto no haya sido ya preprocesado
    if os.path.exists(f"datos_physio/{df_beh['subject'][0]}/df_eda_{df_beh['subject'][0]}_emocionalmente_activantes.csv") and os.path.exists(f"datos_physio/{df_beh['subject'][0]}/df_eda_{df_beh['subject'][0]}_etiqueta.csv"):
        print(f"Ya se procesó el sujeto {df_beh['subject'][0]}")
        continue

    # Dividir el dataframe según df_beh.exp
    for j, exp in enumerate(df_beh["exp"].unique()):
        print(f"Procesando sujeto {df_beh['subject'][0]} - {exp}")
        
        if j == 0:
            onset_exp = df_beh[df_beh["exp"] == exp].onset.max()
            df_exp = df_physio.iloc[:int(onset_exp)+512]
            blocks_onsets = df_beh[df_beh["order"] == 0].onset.values[0:4]
            blocks_finish = df_beh[df_beh["order"] == 63].onset.values[0:4]

        else:
            onset_exp = df_beh[df_beh["exp"] == exp].onset.min()
            df_exp = df_physio.iloc[int(onset_exp)-512:]
            blocks_onsets = df_beh[df_beh["order"] == 0].onset.values[4:]
            blocks_finish = df_beh[df_beh["order"] == 63].onset.values[4:]
            
        list_dfs_blocks = []
        for n in range(4):
            df_eda = pd.DataFrame()

            print(f'Procesando bloque {n+1}/4')
            # Extraemos el bloque
            bloque = n
            block_onset = blocks_onsets[n]
            block_finish = blocks_finish[n]

            df_bloque = df_exp.loc[int(block_onset)-512:int(block_finish)+512]

            # Extraemos features de EDA
            eda_clean = nk.eda_clean(df_bloque["eda"], sampling_rate=256, method="BioSPPy")
            df_eda["EDA_Clean"] = eda_clean

            df_eda["EDA_Phasic"], df_eda["SMNA"], df_eda["EDA_Tonic"],_,_, df_eda["error"], coef_ajuste = cvxEDA_pyEDA(eda_clean, delta=1/256)
            df_eda["EDA_Phasic_normalized"] = nk.standardize(df_eda["EDA_Phasic"])
            df_eda["EDA_Tonic_normalized"] = nk.standardize(df_eda["EDA_Tonic"])

            df_eda["EDA_Raw"] = df_bloque["eda"].reset_index().drop("index",axis=1)
            df_eda["subject"] = df_beh["subject"][0]
            df_eda["exp"] = exp
            df_eda["bloque"] = bloque
            
            plt.figure()
            plt.plot(df_eda["EDA_Raw"], label="EDA Raw")
            plt.plot(df_eda["EDA_Clean"], label="EDA Clean")
            plt.legend()
            plt.title(f'{df_beh["subject"][0]} - {exp}, B. {n+1}/4')
            plt.show()
            
            print(f'Coef. de ajuste: {coef_ajuste}')

            list_coefs_ajuste.append(coef_ajuste)
            list_dfs_blocks.append(df_eda)

        df_eda_exp = pd.concat(list_dfs_blocks)
        list_dfs_eda.append(df_eda_exp)
        df_eda_exp.to_csv(f"datos_physio/{df_beh['subject'][0]}/df_eda_{df_beh['subject'][0]}_{exp}.csv", index=False)

pd.concat(list_dfs_eda).to_csv(f"datos_physio/eda_all_subjects_full_exps.csv", index=False)



# %%
