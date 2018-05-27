from tensorflow import keras
import numpy as np

my_generator = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.3,
    shear_range=0.3
)

generated_data = my_generator.flow_from_directory('data', target_size=(100,100))

my_network = keras.Sequential()

my_network.add(keras.layers.Conv2D(32,3,3,
                input_shape=(100,100,3),
                activation='relu'))

my_network.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

my_network.add(keras.layers.Conv2D(64,3,3, activation='relu'))

my_network.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

my_network.add(keras.layers.Flatten())

my_network.add(keras.layers.Dense(128, activation='relu'))

my_network.add(keras.layers.Dropout(0.5))

my_network.add(keras.layers.Dense(3, activation='softmax'))

my_network.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.0004),
    loss='categorical_crossentropy'
)
