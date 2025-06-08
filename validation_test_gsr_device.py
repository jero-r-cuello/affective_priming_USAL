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

if not os.path.exists("plots/testing_device"):
    os.makedirs("plots/testing_device")
#%%
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


peaks_per_block = []
heigts_per_block = []
amplitudes_per_block = []
rise_times_per_block = []
recovery_times_per_block = []
for n, block in enumerate(blocks):
    df_eda = pd.DataFrame()
    eda_clean = nk.eda_clean(block, sampling_rate=256, method="BioSPPy")
    x = np.arange(len(block))

    # Plot raw vs clean por bloque
    plt.figure()
    plt.plot(x,block, label="EDA Raw")
    plt.plot(x,eda_clean, label="EDA Clean")
    plt.title(dict_blocks[n])
    plt.legend()
    plt.savefig(f'plots/testing_device/raw_vs_clean_{dict_blocks[n]}.png')
    plt.show()

    df_eda["EDA_Phasic"], df_eda["SMNA"], df_eda["EDA_Tonic"],_,_, df_eda["error"], coef_ajuste = cvxEDA_pyEDA(eda_clean, delta=1/256)
    peaks, info = nk.eda_peaks(df_eda["EDA_Phasic"],
                            sampling_rate=256,
                            method="kim2004",
                            amplitude_min=0.2)
    
    peaks_per_block.append(peaks["SCR_Peaks"])
    heigts_per_block.append(peaks["SCR_Height"])
    amplitudes_per_block.append(peaks["SCR_Amplitude"])
    rise_times_per_block.append(peaks["SCR_RiseTime"])
    recovery_times_per_block.append(peaks["SCR_RecoveryTime"])

counts_peaks = [series.sum() for series in peaks_per_block]
mean_heights = [series.mean() for series in heigts_per_block]
mean_amplitudes = [series.mean() for series in amplitudes_per_block]
mean_rise_times = [series.mean() for series in rise_times_per_block]
mean_recovery_times = [series.mean() for series in recovery_times_per_block]

labels = [dict_blocks[i] for i in range(len(peaks_per_block))]

color_map = {
    "Agua destilada":      "orange",
    "Solución fisiológica":"green",
    "Gel conductor":       "red",
    "Dedos control":       "purple"
}
colors = [color_map[label] for label in labels]

# Gráficos de barras con features
plt.figure(figsize=(6, 4))
plt.bar(labels, counts_peaks, color=colors)
plt.ylabel("Número de picos")
plt.title("Picos de SCR por bloque")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plots/testing_device/scr_peaks_per_block.png")
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(labels, mean_heights, color=colors)
plt.ylabel("Media de alturas")
plt.title("Alturas de SCR por bloque")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plots/testing_device/scr_heights_per_block.png")
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(labels, mean_amplitudes, color=colors)
plt.ylabel("Media de amplitudes")
plt.title("Amplitudes de SCR por bloque")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plots/testing_device/scr_amplitude_per_block.png")
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(labels, mean_rise_times, color=colors)
plt.ylabel("Media de tiempos de subida")
plt.title("Tiempos de subida de SCR por bloque")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plots/testing_device/scr_rise_times_per_block.png")
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(labels, mean_recovery_times, color=colors)
plt.ylabel("Media de tiempos de bajada")
plt.title("Tiempos de bajada de SCR por bloque")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plots/testing_device/scr_recovery_times_per_block.png")
plt.show()
# %%
