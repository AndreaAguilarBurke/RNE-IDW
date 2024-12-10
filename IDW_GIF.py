# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:40:23 2024

@author: andre
"""

############################### LIBRERÍAS #####################################
from mpl_toolkits.basemap import Basemap
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import joblib
from keras.models import load_model
import pandas as pd
import imageio  # Librería para crear el GIF

########################### COLORES ESPECIALES ################################
# Definir los colores en formato hexadecimal y almacenarlos en una lista
colores = ['#46B2B5', '#8FD5D5', '#EDB700', '#EEA36B', '#F3906B', '#66BC33','#668865', '#55A38B']

####################### 1. LLAMAR MODELOS RNA #################################
# Ruta para modelos y escaladores
ruta = "C:/Users/andre/OneDrive/Escritorio/MCIA PYTHON/ESTACIONES INIFAP/KERAS INIFAP/"

# Modelos
red_Vel = load_model(ruta + 'RNA_VelViento.h5')
red_Dir = load_model(ruta + 'RNA_DirViento.h5')

# Escaladores
x_scaler_vel = joblib.load(ruta + 'scaler_x_VelViento.joblib')
y_scaler_vel = joblib.load(ruta + 'scaler_y_VelViento.joblib')
x_scaler_dir = joblib.load(ruta + 'scaler_x_DirViento.joblib')
y_scaler_dir = joblib.load(ruta + 'scaler_y_DirViento.joblib')

####################### 2. CONDICIONES INICIALES ##############################
HRS_DIA = 0
DIA_ANO = 255
d_d1 = 90
v_v1 = 0.5

####################### DATOS DE LAS ESTACIONES ###############################
estaciones = ['22', '13467', '13470', '13471', '13472', '13473', '13474', '13476', '15977', '18787', '18833', '22575', '22581', '24576', '24588', '25427', '48291']
coordenadas_estaciones_dict = {
    '22': [21.7853, -102.264], '13467': [22.1233616, -102.254776], '13470': [22.0542774, -102.29464],
    '13471': [22.2176666, -102.274918], '13472': [21.7551384, -102.315], '13473': [22.33206, -102.2231],
    '13474': [22.0025, -102.26667], '13476': [21.91565, -102.318832], '15977': [21.9025269, -102.069763],
    '18787': [21.8648, -102.259331], '18833': [21.9224339, -102.375549], '22575': [22.3637218, -102.291725],
    '22581': [22.1671257, -102.292976], '24576': [21.978, -102.183891], '24588': [21.97286, -102.362831],
    '25427': [21.77011, -102.459053], '48291': [21.942, -102.0603]
}

# Convertir a matrices
coordenadas_estaciones = np.array(list(coordenadas_estaciones_dict.values()))
etiquetas_estaciones = np.array(list(coordenadas_estaciones_dict.keys()))

########################### 4. CREAR MATRIZ DE CEROS ##########################
matriz_v_d = np.zeros((17, 6))

########################### 4. LLENAR MATRICES ################################
#  matriz_d.fill(d_d1)  # (sirve para llenar toda la matriz con el mismo valor de d_d1)

matriz_v_d[:, 0] = HRS_DIA                       # Primera columna con horas del día
matriz_v_d[:, 1] = DIA_ANO                       # Segunda columna con día del año
matriz_v_d[:, 2] = coordenadas_estaciones[:, 0]  # Columna de latitud
matriz_v_d[:, 3] = coordenadas_estaciones[:, 1]  # Columna de longitud
matriz_v_d[:, 4] = d_d1                          # Quinta columna con estado previo de dirección
matriz_v_d[:, 5] = v_v1                          # Sexta columna con Velocidad inicial del viento 

# Impresión de matrices en consola
print("matriz_v_d:")
print(matriz_v_d)

shape = 17
dir_pred = np.full(shape, d_d1)
vel_pred = np.full(shape, v_v1)

############################## 5. I D W #######################################
#Aplicar el IDW
def idw_interpolation(distancias, valores, p):
    # Evitar distancias cero dividiendo por una distancia muy pequeña
    distancias = np.where(distancias == 0, 1e-10, distancias)
    pesos = 1 / (distancias ** p)
    valores_interpolados = np.dot(pesos, valores) / np.sum(pesos, axis=1)
    return valores_interpolados

######################## 6. MALLA DE PUNTOS Y DISTANCIAS ######################
num_estaciones = 17
p = 1

lat_front = [21.7596 - 0.025, 22.3512 + 0.025]
lon_front = [-102.4605 - 0.025, -102.0535 + 0.025]

latitudes = np.linspace(lat_front[0], lat_front[1], 10)
longitudes = np.linspace(lon_front[0], lon_front[1], 10)

puntos_malla = np.array([[lat, lon] for lat in latitudes for lon in longitudes])
distancias = distance_matrix(puntos_malla, coordenadas_estaciones)

x_malla, y_malla = np.meshgrid(longitudes, latitudes)

############################ 7. INTERPOLACIÓN IDW #############################

############################## 8. GENERAR GIF #################################
# Lista para almacenar las imágenes de cada iteración
imagenes = []

# Bucle de 24 horas
for t in range(24):
    # Predicción IDW para velocidad y dirección
    valores_velocidad_interpolados = idw_interpolation(distancias, vel_pred, p)
    valores_direccion_interpolados = idw_interpolation(distancias, dir_pred, p)

    valores_velocidad_reshape = valores_velocidad_interpolados.reshape((10, 10))
    valores_direccion_reshape = valores_direccion_interpolados.reshape((10, 10))

    u = valores_velocidad_reshape * np.cos(np.radians(valores_direccion_reshape))
    v = valores_velocidad_reshape * np.sin(np.radians(valores_direccion_reshape))
    
    # Realizar predicciones de la RNA para velocidad y dirección
    X_dir = np.column_stack([matriz_v_d[:, 0], matriz_v_d[:, 1], matriz_v_d[:, 2], matriz_v_d[:, 3], matriz_v_d[:, 4]])
    X_vel = np.column_stack([matriz_v_d[:, 0], matriz_v_d[:, 1], matriz_v_d[:, 2], matriz_v_d[:, 3], matriz_v_d[:, 5]])

    # Escalar los datos de entrada para la predicción de la RNA
    X_scaled_vel = x_scaler_vel.transform(X_vel)
    X_scaled_dir = x_scaler_dir.transform(X_dir)

    # Realizar las predicciones usando la red neuronal cargada para cada caso
    vel_pred = y_scaler_vel.inverse_transform(red_Vel.predict(X_scaled_vel)).flatten()
    dir_pred = y_scaler_dir.inverse_transform(red_Dir.predict(X_scaled_dir)).flatten()

    # Actualizar la matriz con los valores predichos
    matriz_v_d[:, 5] = vel_pred  # Actualizar la matriz con la velocidad predicha
    matriz_v_d[:, 4] = dir_pred  # Actualizar la matriz con la dirección predicha
    matriz_v_d[:, 0] = matriz_v_d[:, 0] + 1  # Incrementar el índice de tiempo
    
    marcadores = ['s', 'o', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', '|', '_', '.']

    # Generar el gráfico para esta hora
    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Centramos mejor el gráfico
    mapa_v = Basemap(llcrnrlon=np.min(puntos_malla[:, 1]), llcrnrlat=np.min(puntos_malla[:, 0]),
                     urcrnrlon=np.max(puntos_malla[:, 1]), urcrnrlat=np.max(puntos_malla[:, 0]), resolution='h')
    mapa_v.arcgisimage(service='World_Street_Map', verbose=False)
    
    contourf_plot_v = plt.contourf(x_malla, y_malla, valores_velocidad_reshape, shading='nearest', cmap='Blues', alpha=0.5, vmax=5, vmin=0, extend='max', levels=np.linspace(0, 15, 100))
    cbar_v = plt.colorbar(contourf_plot_v, label='Velocidad del Viento (m/s)')
    quiver_plot = plt.quiver(x_malla, y_malla, u, v, scale=50, color='green')
    
    for estacion, marcador in zip(estaciones, marcadores):
        plt.plot(coordenadas_estaciones_dict[estacion][1], coordenadas_estaciones_dict[estacion][0], marker = marcador, markersize = '8',
                 linestyle = 'None', markerfacecolor = 'None', label = estacion, color = 'black')
            
        #plt.plot(coordenadas_estaciones[:, 0], coordenadas_estaciones[:, 1], '', c='black', label='Estaciones')
        plt.xlabel('Latitud')
        plt.ylabel('Longitud')
        titulo_v = plt.title(f'Interpolación de Velocidad del Viento    t = {t}') 

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=6, fontsize = 'small')  # ncol según el número de columnas deseadas
        #plt.colorbar(label='Velocidad del Viento (m/s)')
        plt.pause(0.1)
        
    # Añadir un título al gráfico
    plt.title(f"IDW del viento  t = {t+1} ", fontsize=14)
    
    
    # Guardar imagen en el buffer para GIF
    nombre_imagen = f"frame_{t}.png"
    plt.savefig(nombre_imagen)
    imagenes.append(imageio.imread(nombre_imagen))
    plt.close()

# Crear el GIF a partir de las imágenes
imageio.mimsave('velocidad_viento.gif', imagenes, duration=0.5)