import sys
import os
import csv
import numpy as np
import joblib
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QSizePolicy, QVBoxLayout, QLabel, QMessageBox, QApplication
from PyQt6.uic import loadUi
from PyQt6.QtGui import QPixmap, QPainter, QPalette, QColor, QFont, QMovie
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtWidgets import QGraphicsScene
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from modelo import train, test
from itertools import chain
import re
import multiprocessing
import datetime


# Establecer la configuración de OpenGL antes de crear la instancia de QApplication
from PyQt6.QtWidgets import QApplication
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

#Datos globales
archivos_csv_train = []
archivos_csv_test = []
mejora = 2
folder_path = ""
folder_path_g_train = ""
folder_path_g_test = ""
sequence_length = 7
funct_act = "tanh"
optimizer = "adam"
model = ""
mae = 0
mse = 0
idioma = "Spanish"
model_name = ""
long_train = 0
epochs = 50



class PrimeraInterfaz(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("Inter1.ui", self)

        #Aplicar paleta
        self.apply_Palette_white()
        self.frame.setStyleSheet("background-color: rgb(220, 233, 235)")
        self.Titulo.setStyleSheet("color: rgb(0,0,0)")

        #Crear un icono con la imagen que desees
        pixmap =QPixmap("data/Logo_UCO.jpeg")

        #Cargar engranaje de carga
        self.loading_movie = QMovie("data/carga.gif")
        self.loading_movie.setScaledSize(QSize(42, 42))
        self.loading_label_2.setMovie(self.loading_movie)
        self.loading_label_2.setVisible(False)
        
        # Escalar el pixmap al tamaño del botón
        pixmap = pixmap.scaled(180, 200)
        
        #Asignar el icono al botón
        self.UCO.setPixmap(pixmap)
        
        #Cargar train data
        self.pushButton_7.clicked.connect(self.loadCSV)

        #Crear y Guardar Modelo
        self.pushButton_8.clicked.connect(self.entrenar)

        # Botón para cambiar el idioma
        self.pushButton.clicked.connect(self.change_language)

        # Avanzar segunda interfaz
        self.pushButton_9.clicked.connect(self.abrir_segunda_interfaz)
       
    def change_language(self):
        global idioma
        idioma=self.comboBox_7.currentText()

        if idioma == "English":

            # Crea un objeto QFont con el tamaño especificado y negrita
            font = QFont()
            font.setPointSize(28)
            font.setWeight(QFont.Weight.Bold)

            # Asigna la fuente al texto en self.Titulo
            self.Titulo.setFont(font)
            self.Titulo.setText(self.tr("Super-resolution using Deep Learning techniques"))

            self.setWindowTitle(self.tr("Super-resolution using Deep Learning techniques"))
            self.pushButton.setText(self.tr("Change Language"))
            self.pushButton_7.setText(self.tr("Load Train Series"))
            self.pushButton_8.setText(self.tr("Save Model"))
            self.pushButton_9.setText(self.tr("Next"))
            self.label_18.setText(self.tr("Optimizer"))
            self.label_20.setText(self.tr("Activation Function"))
            self.label_19.setText(self.tr("Window Size"))
            self.label_2.setText(self.tr("Language"))

            if self.label.text() != "":
                self.label.setText("Model Trained and Saved Successfully")

            if self.label_file_name_3.text() != "":
                self.label_file_name_3.setText(f"Folder Loaded  {os.path.basename(folder_path)}")

            font2 = QFont()
            font2.setPointSize(14)
            font2.setWeight(QFont.Weight.Bold)

            # Asigna la fuente al texto en self.Titulo
            self.label_15.setFont(font2)
            self.label_16.setFont(font2)
            self.label_17.setFont(font2)
            self.label_15.setText(self.tr("Step 1: Load Training Files (Optional)"))
            self.label_17.setText(self.tr("Step 2: Parameterize and Save Model (Optional)"))
            self.label_16.setText(self.tr("Step 3: Choose Data and Enhancements for Super-Resolution"))



        elif idioma == "Spanish":
            # Define el tamaño de la fuente
            font_size = 28  # Tamaño de la fuente en puntos

            # Crea un objeto QFont con el tamaño especificado y negrita
            font = QFont()
            font.setPointSize(font_size)
            font.setWeight(QFont.Weight.Bold)

            # Asigna la fuente al texto en self.Titulo
            self.Titulo.setFont(font)
            self.Titulo.setText(self.tr("Superresolucion aplicando tecnicas de DeepLearnig"))

            self.setWindowTitle(self.tr("Superresolucion aplicando tecnicas de DeepLearnig"))
            self.pushButton.setText(self.tr("Cambiar idioma"))
            self.pushButton_7.setText(self.tr("Cargar archivos de entrenamiento"))
            self.pushButton_8.setText(self.tr("Guardar Modelo"))
            self.pushButton_9.setText(self.tr("Siguiente"))
            self.label_18.setText(self.tr("Optimizador"))
            self.label_20.setText(self.tr("Funcion de activacion"))
            self.label_19.setText(self.tr("Tamaño de Ventana"))
            self.label_2.setText(self.tr("Idioma"))

            if self.label.text() != "":
                self.label.setText("Modelo Entrenado y Guardado correctamente")
            
            if self.label_file_name_3.text() != "":
                self.label_file_name_3.setText(f"Carpeta cargada  {os.path.basename(folder_path)}")

            font2 = QFont()
            font2.setPointSize(14)
            font2.setWeight(QFont.Weight.Bold)

            # Asigna la fuente al texto en self.Titulo
            self.label_15.setFont(font2)
            self.label_16.setFont(font2)
            self.label_17.setFont(font2)
            self.label_15.setText(self.tr("Paso 1 Cargar Archivos de Entrenamiento (Opcional)"))
            self.label_17.setText(self.tr("Paso 2 Parametrizar y Guardar Modelo (Opcional)"))
            self.label_16.setText(self.tr("Paso 3 Elegir datos y mejora a realizar Superresolucion"))


    def apply_Palette_white(self):

        palette = QPalette()
        # Colores para la ventana principal
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
        # Colores para la barra de herramientas (toolbar)
        #palette.setColor(QPalette.ColorRole.Light, QColor(255, 200, 200))  # Color de fondo de la toolbar
        palette.setColor(QPalette.ColorRole.Mid, QColor(240, 240, 240))    # Color de fondo de los botones de la toolbar
        palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))  # Color del texto de la toolbar
        # Colores para la barra de estado (status bar)
        palette.setColor(QPalette.ColorRole.Base, QColor(240, 240, 240))    # Color de fondo de la barra de estado
        palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))  # Color del texto de la barra de estado
        # Otros colores
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        # Establecer la palette de colores personalizada
        self.setPalette(palette)
       
          
    def entrenar(self):

        if idioma == "Spanish":
            reply = QMessageBox.question(None, 'Crear Modelo', '¿Estás seguro de que quieres crear el modelo?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        

        else:
            reply = QMessageBox.question(None, 'Create Model', 'Are you sure you want to train the model?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        
        if reply == QMessageBox.StandardButton.Yes:

            self.cargar_parametros()
            
            if idioma == "Spanish":
                self.label.setText("Entrenando y Guardando el Modelo. Por favor espere          ")

            else:
                self.label.setText("Training and Saving the Model. Please wait          ")
            
            self.train_model()

        
        elif reply == QMessageBox.StandardButton.No:
            self.label.setText("")
    
    def train_model(self):

        self.loading_label_2.setVisible(True)
        self.loading_movie.start()

        self.thread = TrainModelThread()
        self.thread.training_complete.connect(self.on_training_complete)
        self.thread.start()

    def on_training_complete(self, message):

        self.loading_movie.stop()
        self.loading_label_2.setVisible(False)

        if idioma == "Spanish":
            self.label.setText("Modelo Entrenado y Guardado correctamente")
            
        else:
            self.label.setText("Model Trained and Saved Successfully")

    def loadCSV(self):
        global archivos_csv_train, archivos_csv_test, folder_path_g_train, folder_path_g_test, folder_path
        if idioma == "Spanish":
            folder_path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta", "/")

        else:
            folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "/")

        if folder_path:
            csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
            
            #Almacenamiento de datos en variables globales
            
            folder_path_g_train = folder_path
            archivos_csv_train = csv_files

            if idioma == "Spanish":
                self.label_file_name_3.setText(f"Carpeta cargada  {os.path.basename(folder_path)}")

            else:
                self.label_file_name_3.setText(f"Folder Loaded  {os.path.basename(folder_path)}")
        

    def cargar_parametros(self):
        global sequence_length, mejora, funct_act, optimizer
        sequence_length = self.spinBox_4.value()
        funct_act = self.comboBox_6.currentText()
        optimizer = self.comboBox_5.currentText()
        
        

    def abrir_segunda_interfaz(self):

        print("Abrir segunda interfaz")
        self.segunda_interfaz = SegundaInterfaz()
        self.segunda_interfaz.show()
        


class SegundaInterfaz(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("Inter3.ui", self)

        #Aplicar paleta
        self.apply_Palette_white()
        self.frame_8.setStyleSheet("background-color: rgb(220, 233, 235)")
        self.label.setStyleSheet("color: rgb(0,0,0)")

        #Cargar logo
        pixmap =QPixmap("data/Logo_UCO.jpeg")
        pixmap = pixmap.scaled(180, 200)
        self.UCO.setPixmap(pixmap)

        #Cambiar idioma si se ha seleccionado ingles
        if idioma == "English":
            self.change_language()

        #Cargar datos a procesar
        self.pushButton_5.clicked.connect(self.loadCSV)

        #Cargar modelo
        self.pushButton_8.clicked.connect(self.cargar_modelo)
        
        #Abrir siguiente interfaz
        self.pushButton_2.clicked.connect(self.abrir_tercera_interfaz)

        #Cerrar interfaz y volver atras
        self.pushButton.clicked.connect(self.Cerrar)

        


    def change_language(self):

        # Crea un objeto QFont con el tamaño especificado y negrita
        font = QFont()
        font.setPointSize(28)
        font.setWeight(QFont.Weight.Bold)

        # Asigna la fuente al texto en self.Titulo
        self.label.setFont(font)
        self.label.setText(self.tr("Processing Options Selection"))

        self.setWindowTitle(self.tr("Processing Options Selection"))
        self.pushButton_5.setText(self.tr("Select Files"))
        self.pushButton_8.setText(self.tr("Select Model"))
        self.pushButton_2.setText(self.tr("Perform Super-Resolution"))
        self.pushButton.setText(self.tr("Go back"))
        
        self.label_7.setText(self.tr("The relationship between the amount of new data and the \nimprovement value will follow the following equation:\n\n NewData = (2 ^ Improvement) - 1"))
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter) 

        font2 = QFont()
        font2.setPointSize(14)
        font2.setWeight(QFont.Weight.Bold)
        
        # Asigna la fuente al texto en self.Titulo
        self.label_2.setFont(font2)
        self.label_6.setFont(font2)
        self.label_5.setFont(font2)
        self.label_2.setText(self.tr("Step 1: Load Data for Super-Resolution"))
        self.label_6.setText(self.tr("Step 2: Select the Super-Resolution Model to Use"))
        self.label_5.setText(self.tr("Step 3: Parameterize Quality Improvement and Perform Super-Resolution"))


    def apply_Palette_white(self):

        palette = QPalette()
        # Colores para la ventana principal
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
        # Colores para la barra de herramientas (toolbar)
        palette.setColor(QPalette.ColorRole.Mid, QColor(240, 240, 240))    # Color de fondo de los botones de la toolbar
        palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))  # Color del texto de la toolbar
        # Colores para la barra de estado (status bar)
        palette.setColor(QPalette.ColorRole.Base, QColor(240, 240, 240))    # Color de fondo de la barra de estado
        palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))  # Color del texto de la barra de estado
        # Otros colores
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(142, 45, 197).lighter())
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        # Establecer la palette de colores personalizada
        self.setPalette(palette)

    def cargar_parametros(self):
        global mejora
        mejora = self.spinBox.value()

    def loadCSV(self):
        global archivos_csv_train, archivos_csv_test, folder_path_g_train, folder_path_g_test, folder_path
        if idioma == "Spanish":
            folder_path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta", "/")

        else:
            folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "/")
        if folder_path:
            csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
            
            #Almacenamiento de datos en variables globales
            folder_path_g_test = folder_path
            archivos_csv_test = csv_files

            if idioma == "Spanish":
                self.label_file_name_4.setText(f"Carpeta cargada  {os.path.basename(folder_path)}")

            else:
                self.label_file_name_4.setText(f"Folder Loaded  {os.path.basename(folder_path)}")
            
            
    def cargar_modelo(self):
        global model, sequence_length, funct_act, optimizador, mse, mae, model_name, long_train, epochs

        # Cargar el modelo
        if idioma == "Spanish":
            model_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar modelo pre-entrenado", "/", "Model Files (*.joblib)")

        else:
            model_path, _ = QFileDialog.getOpenFileName(self, "Select Pre-trained Model", "/", "Model Files (*.joblib)")
        
        
        if model_path:
            # Cargar el modelo desde el archivo
            model = joblib.load(model_path)
            
            # Obtener la carpeta del modelo y el archivo de parámetros
            model_folder = os.path.dirname(model_path)
            params_name = f"{os.path.splitext(os.path.basename(model_path))[0]}_params.txt"
            params_path = os.path.join(model_folder, params_name)
            
            # Leer el archivo de parámetros y extraer los valores
            if os.path.exists(params_path):
                with open(params_path, 'r') as file:
                    params = file.readlines()
                    for param in params:
                        key, value = param.strip().split(": ")
                        if key == "sequence_length":
                            sequence_length = int(value)
                        elif key == "funct_act":
                            funct_act = value
                        elif key == "optimizador":
                            optimizador = value
                        elif key == "mse":
                            mse = value
                        elif key == "mae":
                            mae = value
                        elif key == "name":
                            model_name = value
                        elif key == "ndata_train":
                            long_train = value
                        elif key == "epochs":
                            epochs = value

                if idioma == "Spanish":
                    self.label_file_name_7.setText(f"Modelo {os.path.basename(model_path)} cargado correctamente")
                    
                else:
                    self.label_file_name_7.setText(f"Model {os.path.basename(model_path)} loaded Successfully")
                            
            
            else:
                if idioma == "Spanish":
                    self.label_file_name_7.setText(f"Modelo {os.path.basename(model_path)} cargado, pero el archivo de parámetros no se encontró")
                    
                else:
                    self.label_file_name_7.setText(f"Model {os.path.basename(model_path)} loaded, but the parameter file was not found.")
                            
        

    def Cerrar(self):
        
        if idioma == "Spanish":
            reply = QMessageBox.question(None, 'Cerrar Interfaz', '¿Estás seguro de que quieres cerrar la interfaz?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    
        else:
            reply = QMessageBox.question(None, 'Close Interface', 'Are you sure you want to close the interface?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.close()   

    def abrir_tercera_interfaz(self):
        self.cargar_parametros()

        print("Abrir tercera interfaz")
        self.tercera_interfaz = TerceraInterfaz()
        self.tercera_interfaz.show()


class TerceraInterfaz(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("Inter2.ui", self)
        #Inicializar variables
        self.n=0
        series_original=[]
        series_pred=[]
        self.superres=[]
        self.NB=1
        val_ori = 'Valores Originales Serie {}'
        val_pre = 'Valores Predichos Serie {}'
        val_res = 'Superresolución Serie {}'
        self.apply_Palette_white()

        #Seleccionar idioma
        if idioma == "English":
            self.change_language()
            val_ori = 'Original Serie Values {}'
            val_pre = 'Predicted Serie Values {}'
            val_res = 'Super-Resolution Serie {}'

        series_pred, series_original, self.superres=test(model,sequence_length, folder_path_g_test, mejora)
        # Convertir a array de arrays
        series_pred = [[subsubarray[0] for subsubarray in subarray] for subarray in series_pred]
        
        #Funciones principales UI
        self.Graficar(series_original[0],val_ori.format(self.n))
        self.show_Params_Info(len(series_original[self.n]), len(self.superres[self.n]))
        self.atras.clicked.connect(lambda: self.actualizar_n_y_graficar_menos(series_original,series_pred, self.superres, val_ori, val_pre, val_res))
        self.pasa.clicked.connect(lambda: self.actualizar_n_y_graficar_mas(series_original,series_pred, self.superres, val_ori, val_pre, val_res))
        self.Button1.clicked.connect(lambda: self.Graficar(series_original[self.n], val_ori.format(self.n)))
        self.Button2.clicked.connect(lambda: self.Graficar(series_pred[self.n],val_pre.format(self.n)))
        self.Button3.clicked.connect(lambda: self.Graficar(self.superres[self.n],val_res.format(self.n)))
        self.descargar.clicked.connect(lambda: self.Descargar(self.superres))
        self.cerrar.clicked.connect(self.Cerrar)

    def change_language(self):

        self.setWindowTitle(self.tr("Data Analysis"))
        self.descargar.setText(self.tr("Export Super-Resolution Series"))
        self.cerrar.setText(self.tr("Go Back"))
        self.Button1.setText(self.tr("Original Series"))
        self.Button2.setText(self.tr("Predicted Data"))
        self.Button3.setText(self.tr("Super-resolution"))
        self.atras.setText(self.tr("Previous Series"))
        self.pasa.setText(self.tr("Next Series"))
        
        #Estilo
        font = QFont()
        font.setPointSize(14)
        font.setWeight(QFont.Weight.Bold)
        
        self.label_2.setFont(font)
        self.label.setFont(font)
        self.label_3.setFont(font)
        self.label.setText(self.tr("Parameters Used"))
        self.label_2.setText(self.tr("Overall Model Evaluation"))
        self.label_3.setText(self.tr("Calculated Parameters in the Model"))

    def apply_Palette_white(self):

        palette = QPalette()
        # Colores para la ventana principal
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
        # Colores para la barra de herramientas (toolbar)
        palette.setColor(QPalette.ColorRole.Mid, QColor(240, 240, 240))    # Color de fondo de los botones de la toolbar
        palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))  # Color del texto de la toolbar
        # Colores para la barra de estado (status bar)
        palette.setColor(QPalette.ColorRole.Base, QColor(240, 240, 240))    # Color de fondo de la barra de estado
        palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))  # Color del texto de la barra de estado
        # Otros colores
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(142, 45, 197).lighter())
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        # Establecer la palette de colores personalizada
        self.setPalette(palette)

    def Cerrar(self):

        if idioma == "Spanish":
            reply = QMessageBox.question(None, 'Cerrar Interfaz', '¿Estás seguro de que quieres cerrar la interfaz?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    
        else:
            reply = QMessageBox.question(None, 'Close Interface', 'Are you sure you want to close the interface?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                            

        if reply == QMessageBox.StandardButton.Yes:
            self.close()

    def Descargar(self,series_temporales):
       

        if idioma == "Spanish":
            reply1 = QMessageBox.question(None, 'Exportar Series', '¿Estás seguro de que quieres exportar las series predichas?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    
        else:
            reply1 = QMessageBox.question(None, 'Export Series', 'Are you sure you want to export the predicted series?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                            

        
        if reply1 == QMessageBox.StandardButton.Yes:
            # Crear la carpeta si no existe
            carpeta_destino = os.path.join("data/csv_superres")
            if not os.path.exists(carpeta_destino):
                os.makedirs(carpeta_destino)
            
            # Obtener fecha y hora actual
            current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M")

            # Nombre del modelo y archivo de parámetros
            model_folder_name = f"superres_{current_datetime}"
            model_folder_path = os.path.join("csv_superres", model_folder_name)

            # Crear subcarpeta para el modelo
            if not os.path.exists(model_folder_path):
                os.makedirs(model_folder_path)

            # Guardar cada serie temporal en un archivo dentro de la carpeta
            for i, serie in enumerate(series_temporales):
                nombre_archivo = os.path.join(model_folder_path, f"serie_{i}.csv")
                with open(nombre_archivo, 'w', newline='') as archivo_csv:
                    escritor_csv = csv.writer(archivo_csv)
                    for valor in serie:
                        escritor_csv.writerow([valor])

    def actualizar_n_y_graficar_mas(self, series_original,series_pred,super, val_ori, val_pre, val_res):
        if self.n < len(self.superres)-1:
            self.n += 1
            if self.NB==1:
                self.Graficar(series_original[self.n], val_ori.format(self.n))

            if self.NB==2:
                self.Graficar(series_pred[self.n], val_pre.format(self.n))

            if self.NB==3:
                self.Graficar(super[self.n], val_res.format(self.n))
            

    def actualizar_n_y_graficar_menos(self, series_original,series_pred,super, val_ori, val_pre, val_res):
        if self.n >= 1:
            self.n -= 1
            if self.NB==1:
                self.Graficar(series_original[self.n], val_ori.format(self.n))

            if self.NB==2:
                self.Graficar(series_pred[self.n], val_pre.format(self.n))

            if self.NB==3:
                self.Graficar(super[self.n], val_res.format(self.n))
            

    def Graficar(self, data, titulo):
        global mejora
        colorB = QColor(240, 240, 240)
        # Limpiar el gráfico
        self.graphicsView_5.setHtml("")

        x_values = [i for i in range(1, len(data) + 1)]
        self.plotly_figure = go.Figure()
        
        pattern = r"^Superresolución Serie \d+$"
        pattern_2 = r"^Valores Originales Serie \d+$"

        if idioma == "English":
            pattern = r"^Super-Resolution Serie \d+$"
            pattern_2 = r"^Original Serie Values \d+$"


        if re.match(pattern, titulo):# SUperres
            self.NB=3
            # Cambiar el color del botón
            self.Button1.setStyleSheet(f"background-color: {colorB.name()};color: black;")
            self.Button2.setStyleSheet(f"background-color: {colorB.name()};color: black;")
            self.Button3.setStyleSheet("background-color: lightblue; color: black;")

            # Inicializas una lista para almacenar los colores de las líneas
            traces = []
            num_red_points = sum(2 ** (i - 1) for i in range(1, mejora + 1))

            # Añades una traza que une todos los puntos
            self.plotly_figure.add_trace(go.Scatter(x=x_values, y=data, mode='lines', line=dict(color='blue')))

            trace = go.Scatter(x=[x_values[0]], y=[data[0]], mode='lines+markers', name="", line=dict(color='red'))
            traces.append(trace)

            # Iteras sobre tus datos
            for i in range(len(data)-1):
                # Alternas entre rojo y azul cada 16 valores (15 azules + 1 rojo)
                if i % (num_red_points+1) < num_red_points:
                    color = 'blue'
                else:
                    color = 'red'
                
                # Creas una traza con un solo color para cada punto
                trace = go.Scatter(x=[x_values[i+1]], y=[data[i+1]], mode='lines+markers', name="", line=dict(color=color))
                traces.append(trace)

            # Añades todas las trazas a tu figura Plotly
            for trace in traces:
                self.plotly_figure.add_trace(trace)

           
        
        else:
            if re.match(pattern_2, titulo): #Valores Originales
                self.NB=1
                self.Button3.setStyleSheet(f"background-color: {colorB.name()};color: black;")
                self.Button2.setStyleSheet(f"background-color: {colorB.name()};color: black;")
                self.Button1.setStyleSheet("background-color: lightblue; color: black;")
                color='red'

            else : #Puntos calculados
                self.NB=2
                self.Button3.setStyleSheet(f"background-color: {colorB.name()};color: black;")
                self.Button1.setStyleSheet(f"background-color: {colorB.name()};color: black;")
                self.Button2.setStyleSheet("background-color: lightblue; color: black;")
                color='blue'
            # Graficar todo en un solo color
            self.plotly_figure.add_trace(go.Scatter(x=x_values, y=data, mode='lines+markers',line=dict(color=color)))

        

        self.plotly_figure.update_layout(title=titulo)
        
        html = self.plotly_figure.to_html(full_html=False, include_plotlyjs='cdn')
        self.graphicsView_5.setHtml(html)
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        size_policy.setHorizontalStretch(1)
        size_policy.setVerticalStretch(1)
        self.graphicsView_5.setSizePolicy(size_policy)

    def show_Params_Info(self, n_dat, n_sup):
        
        global sequence_length, funct_act, optimizador, long_train
        
        if idioma == "English":

            self.textBrowser.clear()
            self.textBrowser.append(f" ")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f"<b>Model Name:</b> {model_name}")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f"<b>Amount of training datae:</b> {long_train}")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f"<b>Sequence Length:</b> {sequence_length}")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f"<b>Activation Function:</b> {funct_act}")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f"<b>Optimizer:</b> {optimizer}")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f"<b>Epochs:</b> {epochs}")
            

            self.textBrowser_2.append(f" ")
            self.textBrowser_2.append(f" ")
            self.textBrowser_2.append(f"<b>MSE Scores for Cross-Validation:</b> {mse}")
            self.textBrowser_2.append(f" ")
            self.textBrowser_2.append(f" ")
            self.textBrowser_2.append(f"<b>MAE Scores for Cross-Validation:</b> {mae}")

            self.textBrowser_3.append(f" ")
            self.textBrowser_3.append(f"<b>Quality Improvement:</b> {mejora}")
            self.textBrowser_3.append(f" ")
            self.textBrowser_3.append(f"<b>Number of Initial Data:</b> {n_dat}")
            self.textBrowser_3.append(f" ")
            self.textBrowser_3.append(f"<b>Number of Data Points After Super-Resolution:</b> {n_sup}")

        
        else:
            self.textBrowser.clear()
            self.textBrowser.append(f" ")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f"<b>Nombre del modelo:</b> {model_name}")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f"<b>Numero de datos de entrenamiento:</b> {long_train}")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f"<b>Longitud de secuencia:</b> {sequence_length}")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f"<b>Funcion de activación:</b> {funct_act}")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f"<b>Optimizador:</b> {optimizer}")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f" ")
            self.textBrowser.append(f"<b>Epochs:</b> {epochs}")

            self.textBrowser_2.append(f" ")
            self.textBrowser_2.append(f" ")
            self.textBrowser_2.append(f"<b>MSE para Validación Cruzada:</b> {mse}")
            self.textBrowser_2.append(f" ")
            self.textBrowser_2.append(f" ")
            self.textBrowser_2.append(f"<b>MAE para Validación Cruzada:</b> {mae}")

            self.textBrowser_3.append(f" ")
            self.textBrowser_3.append(f"<b>Mejora:</b> {mejora}")
            self.textBrowser_3.append(f" ")
            self.textBrowser_3.append(f"<b>Numero de datos iniciales:</b> {n_dat}")
            self.textBrowser_3.append(f" ")
            self.textBrowser_3.append(f"<b>Numero de datos tras superresolucion:</b> {n_sup}")
            

class TrainModelThread(QThread):
    training_complete = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def run(self):
        # Simula el entrenamiento de un modelo LSTM
        train(folder_path_g_train, sequence_length,funct_act, optimizer)
            
        # Emitir la señal cuando el entrenamiento esté completo
        self.training_complete.emit("Entrenamiento completo")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    inicio_sesion = PrimeraInterfaz()
    inicio_sesion.show()
    
    sys.exit(app.exec())

