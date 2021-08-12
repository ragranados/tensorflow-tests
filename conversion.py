import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mp

def conversion():
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
    fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

    capa = tf.keras.layers.Dense(units=1, input_shape=[1])

    modelo = tf.keras.Sequential([capa])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
    )

    print('Training')

    historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)

    print('Modelo entrenado')

    mp.xlabel("# Epoca")
    mp.ylabel("Magnitud de perdida")
    mp.plot(historial.history["loss"])
    mp.show()

    print("Prediccion")

    resultado = modelo.predict([100.0])

    print(resultado)

    print("Variables internas del modelo")
    print(capa.get_weights())