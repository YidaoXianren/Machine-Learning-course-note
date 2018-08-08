import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data

#categorical_crossentropy

def load_data():
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    x_train, y_train = mnist.train.images,mnist.train.labels  
    x_test, y_test = mnist.test.images, mnist.test.labels  
    x_train = x_train.reshape(-1, 28, 28,1).astype('float32')  
    x_test = x_test.reshape(-1,28, 28,1).astype('float32') 
    
    x_train = x_train.reshape(55000, 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    #print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_test=np.random.normal(x_test) #aims to add some noise
    #x_train = x_train/255
    #x_test = x_test/255
    #print(x_train.shape[0],'train samples')
    #print(x_test.shape[0],'test samples')
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

model = Sequential()
model.add(Dense(input_dim = 28*28, units = 689, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units = 689, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units = 689, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units = 10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1),metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=20, verbose=2)

result = model.evaluate(x_train, y_train, batch_size = 10000)
print '\nTrain Acc:',result[1]

result = model.evaluate(x_test, y_test, batch_size = 10000)
print '\nTest Acc:',result[1]

    
    
