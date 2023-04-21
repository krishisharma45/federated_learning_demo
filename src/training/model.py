from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model


class CreateMnistModel:
    def __init__(self):
        pass

    def run(self):
        model = self._build_model()
        return self._compile_model(model)

    def _build_model(self):
        input = Input(shape=(28, 28, 1))
        x = Conv2D(32, kernel_size=4, activation="relu")(input)
        x = MaxPooling2D()(x)
        x = Conv2D(16, kernel_size=4, activation="relu")(x)
        x = Flatten()(x)
        output = Dense(10, activation="softmax")(x)

        model = Model(inputs=input, outputs=output)

        return model

    def _compile_model(self, model):
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model