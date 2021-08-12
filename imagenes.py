import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as mp


def run():
    datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

    # print(metadatos)

    datos_entrenamiento, datos_pruebas = datos['train'], datos['test']

    nombres_clases = metadatos.features['label'].names

    print(nombres_clases)

    # Normalizar datos (Pasar de 0-255 a 0-1)

    def normalizar(imagenes, etiquetas):
        imagenes = tf.cast(imagenes, tf.float32)
        imagenes /= 255  # Aqui se hace la conversion
        return imagenes, etiquetas

    # Normalizar los datos de entrenamiento y pruebas

    datos_entrenamiento = datos_entrenamiento.map(normalizar)
    datos_pruebas = datos_pruebas.map(normalizar)

    # Agregar a cache

    datos_entrenamiento = datos_entrenamiento.cache()
    datos_pruebas = datos_pruebas.cache()

    # Crear modelo

    modelo = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)), #1 - Blanco y negro
        tf.keras.layers.Dense(50, activation=tf.nn.relu),
        tf.keras.layers.Dense(50, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax) #A grandes rasgos, en redes de claficacion se usa la, se usa como capa de salida la funcion softmax para asegurar que la suma de todo de 1.
    ])

    #Compilar modelo

    modelo.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    num_ej_entrenamiento = metadatos.splits["train"].num_examples
    num_ej_pruebas = metadatos.splits["test"].num_examples

    TAMANO_LOTE = 32

    datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANO_LOTE)
    datos_pruebas = datos_pruebas.batch(TAMANO_LOTE)

    #Entrenar

    historical = modelo.fit(datos_entrenamiento, epochs=5, steps_per_epoch=math.ceil(num_ej_entrenamiento/TAMANO_LOTE))

    mp.xlabel("#Epoca")
    mp.ylabel("Magnitus de perdida")
    mp.plot(historical.history["loss"])
    mp.show()