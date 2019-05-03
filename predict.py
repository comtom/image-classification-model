import os

import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model


width, height = 128, 128
model = 'model/model.h5'
model_weights = 'model/weights.h5'

cnn = load_model(model)
cnn.load_weights(model_weights)


def predict(path, filename):
    x = load_img(path, target_size=(width, height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)

    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)

    print(f'{filename} - Class: {answer}')
    return answer


if __name__ == 'main':
    for f in os.walk('data/validation128px'):
        for file in f[2]:
            path = f'data/validation128px/{file}'
            predict(path, f'{file}')
