from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGenerator:
    def __init__(self, dataset_path, batch_size, image_shape):
        datagen = ImageDataGenerator(preprocessing_function=lambda x: (x / 127.5) - 1, validation_split=0.2)

        self.train_generator = datagen.flow_from_directory(dataset_path,
                                                           class_mode='input',
                                                           target_size=image_shape[:2],
                                                           batch_size=batch_size,
                                                           subset='training')
        self.validation_generator = datagen.flow_from_directory(dataset_path,
                                                                class_mode='input',
                                                                target_size=image_shape[:2],
                                                                batch_size=batch_size,
                                                                subset='validation')
