import cv2
import numpy as np
import nnfs
import os

nnfs.init()

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset)
    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, datset, label, file),
                               cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    
    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_train, y_train

if __name__ == '__main__':
    # Parameters
    batch_size = 32
    steps = X.shape[0] // batch_size
    if steps * batch_size < X.shape[0]:
        steps += 1

    X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
    X = (X.astype(np.float32) - 255/2) /(255/2)
    X_test = (X_test.astype(np.float32) - 255/2) /(255/2)

    # reshape to 1D
    X = X.reshape(X.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Shuffle training data set
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X, y = X[keyss], y[keys]