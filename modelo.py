import numpy as np
import pandas as pd
import datetime
import joblib
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from itertools import chain
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.callbacks import Callback


def train(folder_path_g, sequence_length, funct_act, optimizador):

    #Inicializar arrays
    scores_mse = []
    scores_mae = []
    archivos_csv_train = [file for file in os.listdir(folder_path_g) if file.endswith('.csv')]
    
    # Función para normalizar los datos entre -1 y 1
    def normalize_data(data):
        min_val = np.min(data)
        max_val = np.max(data)
        return 2 * (data - min_val) / (max_val - min_val) - 1

    # Función para cargar series temporales desde archivos CSV y normalizarlas
    def cargar_series_temporales(archivos):
        series_temporales = []
        for archivo in archivos:
            df = pd.read_csv(folder_path_g+"/"+archivo, header=None)
            serie_temporal = df.values.reshape(-1)  # Asegurar que la serie temporal sea un array unidimensional
            serie_temporal_norm = normalize_data(serie_temporal)
            series_temporales.append(serie_temporal_norm)
        return series_temporales

    # Cargar las series temporales desde los archivos CSV
    series_temporales = cargar_series_temporales(archivos_csv_train)

    #long
    long = sum(len(serie) for serie in series_temporales)

    # Crea secuencias de entrada y salida
    X, y = [], []
    for serie_temporal in series_temporales:
        for i in range(len(serie_temporal) - sequence_length):
            X.append(serie_temporal[i:i + sequence_length])
            y.append(serie_temporal[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Divide los datos en conjuntos de entrenamiento y prueba
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Crea el modelo LSTM
    model = Sequential()
    model.add(LSTM(100, activation=funct_act, input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer=optimizador, loss='mse')

    epochs=50
    batch_size=21
    # Entrena el modelo
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1) 

    # Evaluar el modelo con MSE
    mse_score = mean_squared_error(y_test, model.predict(X_test))
    scores_mse.append(mse_score)

    # Evaluar el modelo con MAE
    mae_score = mean_absolute_error(y_test, model.predict(X_test))
    scores_mae.append(mae_score)

    # Crear carpeta "models" si no existe
    if not os.path.exists("models"):
        os.makedirs("models")

    # Obtener fecha y hora actual
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Nombre del modelo y archivo de parámetros
    model_folder_name = f"modelo_{current_datetime}"
    model_folder_path = os.path.join("models", model_folder_name)

    # Crear subcarpeta para el modelo
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    model_name = f"{model_folder_name}.joblib"
    params_name = f"{model_folder_name}_params.txt"

    # Guardar el modelo en la subcarpeta
    joblib.dump(model, os.path.join(model_folder_path, model_name))

    # Guardar los parámetros en un archivo de texto en la subcarpeta
    params_path = os.path.join(model_folder_path, params_name)
    with open(params_path, 'w') as file:
        file.write(f"name: {model_folder_name}\n")
        file.write(f"ndata_train: {long}\n")
        file.write(f"epochs: {epochs}\n")
        file.write(f"sequence_length: {sequence_length}\n")
        file.write(f"funct_act: {funct_act}\n")
        file.write(f"optimizador: {optimizador}\n")
        file.write(f"mse: {mse_score}\n")
        file.write(f"mae: {mae_score}\n")

        

######################################################################################################## TEST ####################
def test(model,sequence_length, folder_path_g_test, mejora):
    # Función para normalizar los datos entre -1 y 1
    def normalize_data(data):
        min_val = np.min(data)
        max_val = np.max(data)
        return 2 * (data - min_val) / (max_val - min_val) - 1

    # Función para desnormalizar los datos a su escala original
    def denormalize_data(data_norm, data):
        min_val = np.min(data)
        max_val = np.max(data)
        return 0.5 * (data_norm + 1) * (max_val - min_val) + min_val

    # Ejemplo de carga series temporales desde archivos CSV
    test_series = []
    test_series_ORIGINAL = []

    # Iterar sobre los archivos CSV en el directorio
    for filename in os.listdir(folder_path_g_test):
        if filename.endswith('.csv'):  # Solo considerar archivos CSV
            # Leer la serie temporal desde el archivo CSV
            serie_temporal = pd.read_csv(os.path.join(folder_path_g_test, filename), header=None).to_numpy().reshape(-1)
            test_series_ORIGINAL.append(serie_temporal)
            
            # Normalizar la serie temporal
            serie_temporal_norm = normalize_data(serie_temporal)
            test_series.append(serie_temporal_norm)

    series_predichas=[]
    series_superres=[]

    # Hacer predicciones para cada serie temporal
    for i, serie_temporal_norm in enumerate(test_series):
        
        for _ in range(mejora):
        # Crea secuencias de entrada y salida
            test_X = []
            for j in range(len(serie_temporal_norm) - sequence_length):
                test_X.append(serie_temporal_norm[j:j + sequence_length])

            test_X = np.array(test_X)

            # Predice valores intermedios
            # Aquí deberías tener definido tu modelo 'model' para hacer predicciones
            predicted_values_norm_1 = model.predict(test_X)
            predicted_values_norm = [subarray[0] for subarray in predicted_values_norm_1]

            serie_temporal_norm = np.array(list(chain.from_iterable(zip(serie_temporal_norm[sequence_length-1:], predicted_values_norm))))

        # Desnormalizar las predicciones
        predicted_values = denormalize_data(predicted_values_norm_1, test_series_ORIGINAL[i])
        serie_temporal_norm = denormalize_data(serie_temporal_norm, test_series_ORIGINAL[i])
        
        series_predichas.append(predicted_values)
        series_superres.append(serie_temporal_norm)

    return series_predichas, test_series_ORIGINAL, series_superres