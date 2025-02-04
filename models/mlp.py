import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from datasets.load_data import load_mnist
import time

class MLP(tf.keras.Model):
    def __init__(self, input_shape, output_shape, hidden_layers, 
                 hidden_units, activation, 
                 output_activation, loss, optimizer, 
                 epochs =50, batch_size = 64, verbose=1, 
                 regularizer=tf.keras.regularizers.L2, regularizer_rate=1e-4, 
                 initializer=tf.keras.initializers.HeNormal(), dropout_rate=0.5, **kwargs):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
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
        for i in range(self.hidden_layers):
            model.add(layers.Dense(self.hidden_units, 
                                   activation=self.activation, 
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self.regularizer(self.regularizer_rate)
                                    ))
            
            model.add(layers.Dropout(self.dropout_rate))

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
                          verbose=self.verbose,
                          validation_data=(x_val, y_val))
        return history
        

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = models.load_model(path)
    



if __name__ == __main__:
    
    #model parameters
    input_shape = (784,)
    output_shape = 10
    hidden_layers = 2
    hidden_units = 128
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

    #load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist(flatten=True)
    
    #create model
    mlp = MLP(input_shape, output_shape, hidden_layers, hidden_units, 
              activation, output_activation, loss, optimizer, 
              epochs, batch_size, verbose, regularizer, 
              regularizer_rate, initializer, dropout_rate)

    #compile model
    mlp.compile_model()

    #train model

    #Set a timer to measure the time it takes to train the model
    start = time.time()
    history = mlp.train(x_train, y_train, x_val, y_val)
    end = time.time()
    print(f"Training time: {end - start} seconds")

    #evaluate model
    test_loss, test_acc = mlp.evaluate(x_test, y_test)

