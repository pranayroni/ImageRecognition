import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from keras.callbacks import LambdaCallback


# Define a callback function to save the model after training
# def save_model_callback(epoch, logs):
#     model.save('my_model2.h5')
#     print('Saved Model.')
# save_model_callback = LambdaCallback(on_epoch_end=save_model_callback)

saved = False
# if 1==2:
if saved:
    # Load the saved model
    saved_model = tf.keras.models.load_model('my_model.h5')
else:
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the input images
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Build the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=20)
    # model.save('my_model.h5')

    # Evaluate the model on the test set
    model.evaluate(x_test, y_test)

    model.save('my_model2.h5')
    print('\nSaved Model.\n')
    # Use the model to classify an image
    # image = x_test[0]
    # plt.imshow(image, cmap='gray')
    # plt.show()
saved_model = tf.keras.models.load_model('my_model2.h5')


image = Image.open('Drawing.jpeg')

image = image.convert('L')
image = image.resize((28, 28))
image = np.array(image) / 255.0
image = image.reshape((1, 28, 28))  
predictions = saved_model.predict(image)
predicted_label = predictions.argmax()

# Print the predicted label
print(f'The predicted label for the image is: {predicted_label}')