# -*- coding: utf-8 -*-
"""
Experimento de priming afectivo. Parte del proyecto 2025 del laboratorio de
Neurociencias y Psicología Experimental de USAL
"""
from psychopy import visual, data, core, gui, event, monitors
from psychopy.hardware import keyboard
import serial #Import the serial library
import pandas as pd
import numpy as np
import random
import os

#port = serial.Serial("COM3")
def angle_to_pixels(angle_deg, monitor):
    D = monitor.getDistance()  # Distance to screen in cm
    W = monitor.getWidth()     # Width of the screen in cm
    R = monitor.getSizePix()[0]  # Horizontal resolution in pixels
    angle_rad = np.deg2rad(angle_deg)  # Convert degrees to radians
    return 2 * D * np.tan(angle_rad / 2) * (R / W)


def showExpInfoDlg(expInfo):
    # Crear un diálogo manualmente
    dlg = gui.Dlg(title=expInfo['expName'])
    dlg.addField('Participant:', expInfo['participant'])
    dlg.addField('Age:', expInfo['age'])
    dlg.addField('Gender:', expInfo['gender'])
    dlg.addField('Orden Exp:', choices=['1', '2'])  # Añadir un desplegable
    dlg.show()  # Mostrar el diálogo

    if dlg.OK:  # Si el usuario presiona OK
        # Actualizar expInfo con los valores ingresados por el usuario
        dlgData = dlg.data
        expInfo['participant'] = dlgData[0]
        expInfo['age'] = dlgData[1]
        expInfo['gender'] = dlgData[2]
        expInfo['orden_exp'] = dlgData[3]
    else:
        core.quit()  # El usuario presionó cancelar

    return expInfo

expName = "Experimento USAL"
expInfo = {'participant': "", 'age': "", 'gender': "", 'orden_exp': '', 'date': data.getDateStr(), 'expName': expName}

expInfo = showExpInfoDlg(expInfo)


# Leemos los archivos necesarios
supra_bloques = pd.read_csv("estimulos/sheet_maestro.csv")
trials_practica = "estimulos/trials_practica.csv"

#%% Crear los componentes
win = visual.Window([1024,768], color=(-1, -1, -1), fullscr=True, units="pix")
kb = keyboard.Keyboard()
respuestas = []
rts = []
trials = []

monitor = monitors.Monitor('testMonitor')
distancia_esperada = 100 # En cm
angulo = 5
monitor.setDistance(distancia_esperada)
face_size = angle_to_pixels(angulo,monitor)

instrucciones_pre_practica = visual.TextStim(win, text="En la pantalla verás una secuencia de estímulos: primero una cruz de fijación, luego una palabra, y finalmente un rostro.\n\nTu tarea es:\nIdentificar la emoción del rostro presionando la flecha IZQUIERDA si la emoción es NEGATIVA y la flecha DERECHA si la emoción es POSITIVA.\n\nIntenta responder con precisión y rapidez. Recuerda que la tarea está dividida en 4 bloques.\n\nPresiona cualquier tecla para comenzar el primer bloque. ¡Buena suerte!",wrapWidth=monitor.getSizePix()[0],font="Arial", pos=(0, 0), color="white", height=40)
cierre = visual.TextStim(win, text="¡Has completado todos los bloques!\n\nGracias por tu participación y esfuerzo en esta tarea.\n\nPor favor, informa al experimentador que has terminado ",wrapWidth=monitor.getSizePix()[0], font="Arial", pos=(0, 0), color="white", height=40)
fixation = visual.TextStim(win, text="+",font="Arial", pos=(0, 0), color="white", height=80, wrapWidth=None, alignText='center',  anchorHoriz='center')



#%% Dibujarlos y presentarlos
# draw the stimuli
instrucciones_pre_practica.draw()
win.callOnFlip(kb.clock.reset)
win.flip()

keys = kb.waitKeys(clear=True)
if keys:
    response = keys[0].name
    rt = keys[0].rt
    if response == "escape":  # Exit on "escape"
        core.quit()


# Trials de práctica
practiceList = data.importConditions(trials_practica)
trials = data.TrialHandler(trialList=practiceList, nReps=3, method='random', extraInfo=expInfo)

for trial in trials:
    fixation.draw()
    win.flip()
    core.wait(1) #
    
    # Mostrar prime
    prime_text = visual.TextStim(win, text=trial["palabra"].upper(),font="Arial", pos=(0, 0), color="white", height=80, wrapWidth=None, alignText='center',  anchorHoriz='center')
    prime_text.draw()
    win.flip()
    core.wait(1)
    
    # Pantalla vacía
    win.flip()
    core.wait(0.5)
    
    # Mostrar target
    target_image = visual.ImageStim(win, image=trial["rostro"], size=(face_size,None))
    target_image.draw()
    #win.callOnFlip(port = serial.Serial("COM4"))# open port
    #win.callOnFlip(ser.close())# close port
    win.callOnFlip(kb.clock.reset)
    win.flip()
    
    keys = kb.waitKeys(maxWait=1, keyList=["left", "right", "escape"], clear=True)
    response = None
    rt = None
    if keys:
        response = keys[0].name
        rt = keys[0].rt
        if response == "escape":  # Exit on "escape"
            core.quit()
    
    # Tiempo entre estímulos
    #win.callOnFlip(port = serial.Serial("COM4"))# open port
    #win.callOnFlip(ser.close())# close port
    win.flip()
    core.wait(random.uniform(0.6, 0.8)) # ISI random de entre 0.6 y 0.8
    
    trials.addData('response', response)
    trials.addData('rt', rt)
    trials.addData('rep', "practica")
    trials.addData('orden_exp', expInfo["orden_exp"])

trials.saveAsExcel(f'data/{expInfo["participant"]}_practica.xlsx',dataOut=["all_raw"])

instrucciones_post_practica = visual.TextStim(win, text="Perfecto! Ahora pasaremos al experimento en sí.\n\nÉxitos!.\n\nPresiona cualquier tecla para comenzar.", font="Arial", pos=(0, 0), color="white", height=40)

instrucciones_post_practica.draw()
win.callOnFlip(kb.clock.reset)
win.flip()

keys = kb.waitKeys(clear=True)
if keys:
    response = keys[0].name
    rt = keys[0].rt
    if response == "escape":  # Exit on "escape"
        core.quit()


# Trials posta
if expInfo["session"] == 2:
    supra_bloques["nombre_supra_bloques"] = supra_bloques["nombre_supra_bloques"].iloc[::-1].reset_index(drop=True)

for supra_bloque in supra_bloques["nombre_supra_bloques"]:
    df_supra_bloque = pd.read_csv(f'{supra_bloque}.csv')
    df_supra_bloque["nombre_bloques"] = df_supra_bloque["nombre_bloques"].sample(frac=1).reset_index(drop=True) # Randomizamos los bloques
    df_supra_bloque = pd.concat([df_supra_bloque,df_supra_bloque],axis=0)
    
    n_rep = 1
    
    for bloque in df_supra_bloque["nombre_bloques"]:
        trialList = data.importConditions(f'{bloque}.csv')
        
        trials = data.TrialHandler(trialList=trialList, nReps=1, method='random', extraInfo=expInfo)
        
        for trial in trials:
            fixation.draw()
            win.flip()
            core.wait(1) #
            
            # Mostrar prime
            prime_text = visual.TextStim(win, text=trial["palabra"].upper(),font="Arial", pos=(0, 0), color="white", height=80, wrapWidth=None, alignText='center',  anchorHoriz='center')
            prime_text.draw()
            win.flip()
            core.wait(1)
            
            # Pantalla vacía
            win.flip()
            core.wait(0.5)
            
            # Mostrar target
            target_image = visual.ImageStim(win, image=trial["rostro"], size=(face_size,None))
            target_image.draw()
            #win.callOnFlip(port = serial.Serial("COM4"))# open port
            #win.callOnFlip(ser.close())# close port
            win.callOnFlip(kb.clock.reset)
            win.flip()
            
            keys = kb.waitKeys(maxWait=1, keyList=["left", "right", "escape"], clear=True)
            response = None
            rt = None
            if keys:
                response = keys[0].name
                rt = keys[0].rt
                if response == "escape":  # Exit on "escape"
                    core.quit()
            
            # Tiempo entre estímulos
            #win.callOnFlip(port = serial.Serial("COM4"))# open port
            #win.callOnFlip(ser.close())# close port
            win.flip()
            core.wait(random.uniform(0.6, 0.8)) # ISI random de entre 0.6 y 0.8
            
            trials.addData('response', response)
            trials.addData('rt', rt)
            trials.addData('rep', n_rep)
            trials.addData('orden_exp', expInfo["orden_exp"])
        
        trials.saveAsExcel(f'data/{expInfo["participant"]}_{bloque[10:]}_rep_{n_rep}.xlsx',dataOut=["all_raw"])
        
        n_rep += 1
        
        instrucciones_fin_bloque = visual.TextStim(win, text="Fin del bloque actual\n\nEs importante que avises al experimentador que terminaste el bloque.\n\nTómate un momento para descansar si lo necesitas.",wrapWidth=monitor.getSizePix()[0], font="Arial", pos=(0, 0), color="white", height=40)
        instrucciones_fin_bloque.draw()
        win.callOnFlip(kb.clock.reset)
        win.flip()
        
        keys = kb.waitKeys(clear=True)
        if keys:
            response = keys[0].name
            rt = keys[0].rt
            if response == "escape":  # Exit on "escape"
                core.quit()

cierre.draw()
win.flip()
event.waitKeys()