 # Paso 1: Importar bibliotecas
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import LeakyReLU

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Paso 2: Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
# Paso 3: Preprocesar los datos
x_train = x_train.astype('float32') / 255  # Normalización
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28 * 28)     # Aplanar las imágenes
x_test = x_test.reshape(-1, 28 * 28)

y_train = tf.keras.utils.to_categorical(y_train, 10)  # One-hot encoding
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Capas iniciales
model = models.Sequential()
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,), kernel_regularizer=regularizers.l2(0.001)))


model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))


model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))

# Capas ocultas adicionales
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))

# Capa de salida
model.add(layers.Dense(10, activation='softmax'))


# Compilar el modelo con la función de pérdida y el optimizador
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Función de pérdida
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(x_train, y_train, epochs=10, batch_size=200, validation_split=0.2)

# Paso 7: Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Precisión en el conjunto de prueba: {test_acc}')

#paso 8 guardar el modelo

model.save('model_1.55.h5')  # Guardar como archivo HDF5
print("Modelo guardado")
