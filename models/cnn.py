import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from datasets.load_data import load_mnist
import time

class CNN(tf.keras.Model):
    def __init__(self, input_shape, output_shape, activation, 
                 output_activation, filters, poolings, fc_layers, loss, optimizer, 
                 epochs =50, batch_size = 64, verbose=1, 
                 regularizer=tf.keras.regularizers.L2, regularizer_rate=1e-4, 
                 initializer=tf.keras.initializers.HeNormal(), dropout_rate=0.5, **kwargs):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dropout_rate = dropout_rate
        self.filters = filters
        self.poolings = poolings
        self.fc_layers = fc_layers

        self.regularizer = regularizer
        self.regularizer_rate = regularizer_rate
        self.initializer = initializer
        self.activation = activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = ['accuracy']
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = self.create_model()
    
    @property
    def metrics(self):
        return self._metrics
    @metrics.setter
    def metrics(self, value):
        self._metrics = value

    def create_model(self):
        model = models.Sequential()
        model.add(layers.InputLayer(shape=self.input_shape))
        
        # add convolutional and pooling layers
        for i, j in zip(self.filters, self.poolings):
            n_filters, filter_size = i
            model.add(layers.Conv2D(n_filters, kernel_size=(filter_size, filter_size),
                                    activation=self.activation, 
                                    kernel_initializer=self.initializer, 
                                    kernel_regularizer=self.regularizer(self.regularizer_rate)))
            if j > 0:
                model.add(layers.MaxPooling2D(pool_size=(j, j)))
        
        # flatten the output of the convolutional layers
        model.add(layers.Flatten())

        # add fully connected layers
        for i in self.fc_layers:
            model.add(layers.Dense(i, activation=self.activation, 
                                  kernel_initializer=self.initializer, 
                                  kernel_regularizer=self.regularizer(self.regularizer_rate)))
            model.add(layers.Dropout(self.dropout_rate))

        # add output layer
        model.add(layers.Dense(self.output_shape, activation=self.output_activation))
        
        return model

    def compile_model(self):
        self.model.compile(loss=self.loss, 
                           optimizer=self.optimizer, 
                           metrics=self.metrics)
    
    def train(self, x_train, y_train, x_val, y_val):
        history = self.model.fit(x_train, y_train,
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          validation_data=(x_val, y_val),
                          verbose=self.verbose)
        return history
    
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    
    def predict(self, x):
        return self.model.predict(x)
    

if __name__ == '__main__':
    # model parameters
    input_shape = (28, 28, 1)
    output_shape = 10
    # filters = [[n_filters, filter_size], ...]
    filters = [[16, 5], [32, 5], [64, 5]]
    # poolings = [pool_size, ...]
    poolings = [0, 0, 2]
    # fc_layers = [n_units, ...]
    fc_layers = [64, 32]
    activation = 'relu'
    output_activation = 'softmax'
    loss = losses.CategoricalCrossentropy()
    learning_rate = 1e-3
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    metrics = ['accuracy']
    epochs = 50
    batch_size = 64
    verbose = 1
    regularizer = tf.keras.regularizers.L2
    regularizer_rate = 1e-4
    initializer = tf.keras.initializers.HeNormal()
    dropout_rate = 0.5

    # load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()

    # create model
    model = CNN(input_shape, output_shape, activation, 
                output_activation, filters, poolings, fc_layers, 
                loss, optimizer, epochs, batch_size, verbose,
                regularizer, regularizer_rate, initializer, dropout_rate)
    
    # compile model
    model.compile_model()

    #Set a timer to measure the time it takes to train the model
    start = time.time()
    history = model.train(x_train, y_train, x_val, y_val)
    end = time.time()
    print(f"Training time: {end - start} seconds")

    #evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test)


