import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout,Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils


batch_size = 128
nb_classes = 10

#input data dimensions
img_rows, img_cols = 28,28

#data shuffled and split between training and test sets
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#reshape data
X_train = X_train.reshape(X_train.shape[0],1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0],1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /=255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], ' train samples')
print(X_test.shape[0], ' test samples')

#convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes)
print('Encoding becomes: {}'.format(Y_train[0,:]))

for i in range(9):
    plt.subplot(3,3, i+1)
    plt.imshow(X_train[i,0], cmap='gray')
    plt.axis('off')


#making model
model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#Train
model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta')

nb_epoch = 2 #this says (iterations on a dataset)

model.fit(X_train,Y_train, batch_size=batch_size, epochs = nb_epoch,
        verbose=1,validation_data=(X_test,Y_test))

score = model.evaluate(X_test,Y_test, verbose = 0)
print('accuracy:{}%'.format(100-score))

res = model.predict_classes(X_test[:9])
plt.figure(figsize=(10,10))

#visualizing some images
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[i,0],cmap='gray')
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.ylabel('prediction = %d' % res[i], fontsize = 18)
