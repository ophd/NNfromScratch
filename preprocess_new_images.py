import cv2
import os
import numpy as np

def load_new_images(path):
    path = 'fashion_new_images'
    files = os.listdir(path)
    imgs = []

    for file in files:
        print(f'Loading {file}...')
        image_data = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        image_data = cv2.resize(image_data, (28, 28))
        image_data = 255 - image_data
        imgs.append(image_data)

    imgs = np.array(imgs)
    imgs = (imgs.reshape(imgs.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    return imgs