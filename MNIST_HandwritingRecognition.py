import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # suppress info and warning messages
import tensorflow.keras as keras
import math
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

mnist = keras.datasets.mnist

# Create training and test sets
(X_train, y_train),(X_test, y_test) = mnist.load_data()

print(f"X_train shape: {X_train.shape}")
print(type(X_train))
print(f"y_train shape: {y_train.shape}")
print(type(y_train))
print(f"X_test shape: {X_test.shape}")
print(type(X_test))
print(f"y_test shape: {y_test.shape}")
print(type(y_test))


X_train[0].shape

X_train[0]


# plt.figure(figsize=(5, 5))
# sns.heatmap(X_train[0], cmap='gray', cbar=False)
# plt.show()

sns.heatmap(X_train[0])
plt.show()

print(y_train[0])

# Function to visualize the data
def plot_imgs(images, labels=None):
    subplots_x = int(math.ceil(len(images) / 5))
    plt.figure(figsize=(10,2*subplots_x))
    for i in range(min(len(images), subplots_x*5)):
        plt.subplot(subplots_x,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        if labels is not None:
            plt.xlabel(labels[i])
    plt.show()

# Visualize some training examples
plot_imgs(X_train[:8], y_train[:8])


X_train = X_train / 255.0
X_test = X_test / 255.0


# X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
# X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

num_examples_X_train = X_train.shape[0]
X_train = np.reshape(X_train, (num_examples_X_train, 28, 28, 1))

num_examples_X_test = X_test.shape[0]
X_test = np.reshape(X_test, (num_examples_X_test, 28, 28, 1))


# 1. Create CNN model object
cnn_model = keras.Sequential()


# 2. Create the input layer and add it to the model object:
input_layer = keras.layers.InputLayer(input_shape=X_train[0].shape)
cnn_model.add(input_layer)


# 3. Create the first convolutional layer and add it to the model object:
conv_1 = keras.layers.Conv2D(filters=16, kernel_size=3)
batchNorm_1 = keras.layers.BatchNormalization()
ReLU_1 = keras.layers.ReLU()
cnn_model.add(conv_1)
cnn_model.add(batchNorm_1)
cnn_model.add(ReLU_1)

# 4. Create the second convolutional layer and add it to the model object:
conv_2 = keras.layers.Conv2D(filters=32, kernel_size=3)
batchNorm_2 = keras.layers.BatchNormalization()
ReLU_2 = keras.layers.ReLU()
cnn_model.add(conv_2)
cnn_model.add(batchNorm_2)
cnn_model.add(ReLU_2)


# 5. Create the third convolutional layer and add it to the model object:
conv_3 = keras.layers.Conv2D(filters=64, kernel_size=3)
batchNorm_3 = keras.layers.BatchNormalization()
ReLU_3 = keras.layers.ReLU()
cnn_model.add(conv_3)
cnn_model.add(batchNorm_3)
cnn_model.add(ReLU_3)


# 6. Create the fourth convolutional layer and add it to the model object:
conv_4 = keras.layers.Conv2D(filters=128, kernel_size=3)
batchNorm_4 = keras.layers.BatchNormalization()
ReLU_4 = keras.layers.ReLU()
cnn_model.add(conv_4)
cnn_model.add(batchNorm_4)
cnn_model.add(ReLU_4)


# 7. Create the pooling layer and add it to the model object:
pooling_layer = keras.layers.GlobalAveragePooling2D()
cnn_model.add(pooling_layer)


# 8. Create the output layer and add it to the model object:
output_layer = keras.layers.Dense(units=10)
cnn_model.add(output_layer)

cnn_model.summary()


sgd_optimizer = keras.optimizers.SGD(learning_rate=0.1)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

cnn_model.compile(optimizer=sgd_optimizer, loss=loss_fn, metrics=['accuracy'])


num_epochs = 1 # Number of epochs

t0 = time.time() # start time

history = cnn_model.fit(X_train, y_train, epochs=num_epochs)

t1 = time.time() # stop time

print('Elapsed time: %.2fs' % (t1-t0))

loss, accuracy = cnn_model.evaluate(X_test, y_test)

print('Loss: ', str(loss) , 'Accuracy: ', str(accuracy))

# Make predictions on the test set
logits = cnn_model.predict(X_test)
predictions = logits.argmax(axis = 1)

## Plot individual predictions
plot_imgs(X_test[:25], predictions[:25])
