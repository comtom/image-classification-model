import sys
import os
import csv

#import matplotlib.pyplot as plt  
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard

K.clear_session()

data_entrenamiento = './data/train'
data_validacion = './data/train'

"""
Parametros
"""
epocas=16
longitud, altura = 60, 60
batch_size = 20
pasos = 500
validation_steps = 200
filtrosConv1 = 60
filtrosConv2 = 40
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 5
lr = 0.001


labels = {}

with open('trainLabels.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        # if line_count == 0:
        #     print(f'Column names are {", ".join(row)}')
        #print(f'\t{row["image"]}.tiff {row["level"]}')
        labels[f'{row["image"]}.tiff'] = row["level"]



# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
#                           write_graph=True, write_images=True)



##Preparamos nuestras imagenes
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    zoom_range=0.1,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

print('Indices:')
print(entrenamiento_generador.class_indices)

cnn = Sequential()
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura,3), activation='relu'))
cnn.add(Convolution2D(filtrosConv1, tamano_filtro2, padding ="same", activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same", activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dense(256, activation='relu'))
cnn.add(Dense(clases, activation='softmax'))
cnn.summary()

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])

print('Arranca el entrenamiento')
snn=cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps,
    verbose=2,
    shuffle=True,
    workers=3,
    #callbacks=tensorboard,
    )

print('Termino el entrenamiento')

print(entrenamiento_generador.class_indices)

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')


# #Graficos de salida
# plt.figure(0)  
# plt.plot(snn.history['acc'],'r')  
# plt.plot(snn.history['val_acc'],'g')  
# plt.xticks(np.arange(0, 21, 1.0))  
# plt.rcParams['figure.figsize'] = (8, 6)  
# plt.xlabel("Epocas")  
# plt.ylabel("Precisión")  
# plt.title("Precisión Entrenamiento vs Precisión Validación")  
# plt.legend(['Entrenamiento','Validación'])

# plt.figure(1)  
# plt.plot(snn.history['loss'],'r')  
# plt.plot(snn.history['val_loss'],'g')  
# plt.xticks(np.arange(0, 21, 1.0))  
# plt.rcParams['figure.figsize'] = (8, 6)  
# plt.xlabel("Epocas")  
# plt.ylabel("Error")  
# plt.title("Error Entrenamiento vs Error Validación")  
# plt.legend(['Entrenamiento','Validación'])

# plt.show()
