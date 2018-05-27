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

#my_network.fit_generator(generated_data, epochs=5)
#my_network.save_weights('fruits.h5')

my_network.load_weights('fruits.h5')

test_img = keras.preprocessing.image.load_img('banana.jpg')
test_img_arr = keras.preprocessing.image.img_to_array(
    test_img,
    data_format='channels_last'
)
test_img_arr = np.array([test_img_arr])

print generated_data.class_indices
print my_network.predict(test_img_arr)
