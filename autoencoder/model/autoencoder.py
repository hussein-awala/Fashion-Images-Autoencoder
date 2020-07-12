from .utils import get_pool_sizes, find_shape
from tensorflow.keras.layers import Input, UpSampling2D, Flatten, Conv2D, MaxPooling2D, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class Autoencoder:
    def __init__(self, image_shape, filter_size, kernel_size, max_up_down_sampling):
        self.image_shape = image_shape
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.max_up_down_sampling = max_up_down_sampling
        self.vector_size = None
        self.autoencoder_model = self.build_model()
        optimizer = Adam(lr=0.001)
        self.autoencoder_model.compile(loss='mse', optimizer=optimizer)

    def __int__(self, path):
        self.autoencoder_model = load_model(path)
        optimizer = Adam(lr=0.001)
        self.autoencoder_model.compile(loss='mse', optimizer=optimizer)

    def build_encoder(self, image_shape, filter_size, kernel_size, downsampling_times):
        pool_sizes = get_pool_sizes(image_shape)[:downsampling_times][::-1]
        input_layer = Input(shape=image_shape)
        h = input_layer
        for i in range(len(pool_sizes)):
            h = Conv2D(filter_size, kernel_size, activation='relu', padding='same', )(h)
            h = MaxPooling2D(pool_sizes[i], padding='same')(h)
        output_layer = Flatten()(h)
        encoder = Model(input_layer, output_layer, )
        return encoder

    def build_decoder(self, image_shape, filter_size, kernel_size, upsampling_times, vector_size):
        new_shape = find_shape(image_shape, vector_size, filter_size)
        pool_sizes = get_pool_sizes(image_shape)[:upsampling_times]
        input_layer = Input(shape=vector_size)
        h = Reshape(new_shape)(input_layer)
        for i in range(len(pool_sizes)):
            h = Conv2D(filter_size, kernel_size, activation='relu', padding='same')(h)
            h = UpSampling2D(pool_sizes[i])(h)
        output_layer = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(h)
        decoder = Model(input_layer, output_layer,)
        return decoder

    def build_model(self):
        self.encoder = self.build_encoder(self.image_shape, self.filter_size, self.kernel_size, self.max_up_down_sampling)
        self.vector_size = self.encoder.output_shape[1]
        self.decoder = self.build_decoder(self.image_shape, self.filter_size, self.kernel_size, self.max_up_down_sampling,
                                     self.vector_size)
        image = self.encoder.input
        decoded_image = self.decoder(self.encoder(image))
        autoencoder = Model(self.encoder.input, decoded_image)
        return autoencoder

    def train_model(self, train_generator, validation_generator, epochs):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')

        history = self.autoencoder_model.fit(x=train_generator,
                                             validation_data=validation_generator,
                                             epochs=epochs,
                                             callbacks=[early_stopping])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def eval(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        return preds

    def save(self, path):
        self.autoencoder_model.save(path)

    def summary(self):
        return self.autoencoder_model.summary()


