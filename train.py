import os
import glob
import numpy as np
import warnings
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, BatchNormalization, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras import backend as K

import tensorflow as tf
tf.compat.v1.reset_default_graph()  # Use the v1 compatibility function

warnings.filterwarnings("ignore")

# File paths for dataset
train_dir = r"C:\Users\Dharrsh\PycharmProjects\pythonProjectfinal\Medicinal Leaf Dataset\Segmented Medicinal Leaf Images"
test_dir = r"C:\Users\Dharrsh\PycharmProjects\pythonProjectfinal\Medicinal Leaf Dataset\Segmented Medicinal Leaf Images"

# Function to count images in the dataset
def get_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for current_path, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(current_path, dr + "/*")))
    return count

# Image data generators for data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   validation_split=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Image parameters
img_width, img_height = 256, 256
input_shape = (img_width, img_height, 3)
batch_size = 32

# GAN Threshold
THRESHOLD = 10

# CNN Model building
def build_cnn_model(num_classes):
    model = Sequential([
        Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.25),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# GAN Model building
def build_gan(img_shape):
    # Generator
    generator = Sequential([
        Dense(128, activation="relu", input_shape=(100,)),
        BatchNormalization(momentum=0.8),
        Dense(256, activation="relu"),
        BatchNormalization(momentum=0.8),
        Dense(np.prod(img_shape), activation="tanh"),
        Reshape(img_shape)
    ])

    # Discriminator
    discriminator = Sequential([
        Flatten(input_shape=img_shape),
        Dense(512, activation="relu"),
        Dense(256, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # GAN
    discriminator.trainable = False
    gan_input = np.random.randn(100)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(optimizer="adam", loss="binary_crossentropy")
    return generator, discriminator, gan

# Function to generate synthetic images
def generate_synthetic_images(generator, target_dir, num_images):
    noise = np.random.normal(0, 1, (num_images, 100))
    gen_images = generator.predict(noise)
    for i, img_array in enumerate(gen_images):
        img = array_to_img(img_array * 127.5 + 127.5, scale=False)
        img.save(os.path.join(target_dir, f"synthetic_{i}.png"))

# Checking dataset size and generating synthetic images if necessary
train_samples = get_files(train_dir)
if train_samples < THRESHOLD:
    generator, discriminator, gan = build_gan(input_shape)
    generate_synthetic_images(generator, train_dir, THRESHOLD - train_samples)

# CNN model
model = build_cnn_model(num_classes=len(glob.glob(train_dir + "/*")))

# Train the model
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical', shuffle=True)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical', shuffle=True)

train = model.fit(train_generator, epochs=10, steps_per_epoch=train_generator.samples // batch_size, validation_data=test_generator, validation_steps=test_generator.samples // batch_size, verbose=1)

# Save the model in the native Keras format
model.save('leaf.keras')

# Clear the session after saving
K.clear_session()
