from typing import Tuple

import numpy as np
from tensorflow.keras.datasets import mnist


class Dataset:
    def __init__(self, device_id: str):
        self.device_id = device_id

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (x_train, x_test), (y_train, y_test) = self._split()
        return self._normalize(x_train, x_test, y_train, y_test)

    def _split(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        (x_train, y_train), (x_test, y_test) = self._load_mnist_data()
        x_train = self._split_train_set(x_train)
        y_train = self._split_train_set(y_train)
        x_test = self._split_test_set(x_test)
        y_test = self._split_test_set(y_test)
        return (x_train, x_test), (y_train, y_test)

    def _normalize(self, x_train, x_test, y_train, y_test) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("---NORMALIZE---")
        # image_size = x_train.shape[1]
        # x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        # x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        return x_train, x_test, y_train, y_test

    def _split_test_set(self, dataset) -> np.ndarray:
        device_dict = {"1": 0,
                       "2": 3333,
                       "3": 6666}
        start_index = device_dict.get(self.device_id)
        end_index = start_index + 3333
        print("---SPLIT TEST SET---")
        print(start_index)
        print(end_index)
        print(len(dataset[start_index:end_index]))
        return dataset[start_index:end_index]

    def _split_train_set(self, dataset) -> np.ndarray:
        device_dict = {"1": 0,
                       "2": 20000,
                       "3": 40000}
        start_index = device_dict.get(self.device_id)
        end_index = start_index + 20000
        print("---SPLIT TRAIN SET---")
        print(start_index)
        print(end_index)
        print(len(dataset[start_index:end_index]))
        return dataset[start_index:end_index]

    def _load_mnist_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        return mnist.load_data()

