from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, time, math
import keras
from keras import backend as K
from keras import regularizers
from keras.regularizers import l2
from keras.models import Sequential, model_from_json
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,BatchNormalization,AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

from models.neural_network import NeuralNetwork


class CNN(NeuralNetwork):
    """Convolutional Neural Network - 2 hidden layers (for now) """

    def __init__(self, non_linearity='relu', **kwargs):
        """
        Params:
        non_linearity(str): can be 'relu', 'tanh', 'sigmoid'
        """
        self.non_linearity = non_linearity
        super(CNN, self).__init__(**kwargs)
        
    def create_model(self, dataset='mnist'):
        """
        Create a Keras Sequential Model
        """

        if 'mnist' in dataset.lower():
            model = self.mnist_cnn()
                
        elif 'cifar10' in dataset.lower():
            model = self.resnet_v1()
               
        return model
    
    
    def get_params(self):
        """
        Desc:
            Required for getting params to be used in HVP Lissa calculations
        """
        all_params = []
        for layer in self.model.layers:
            for weight in layer.trainable_weights:
                all_params.append(weight)
        return all_params        
        

    def create_input_placeholders(self):
        """
        Desc:
            Create input place holders for graph and model.
        """
        input_shape_all = (None,self.input_side, self.input_side,self.input_channels)
        label_shape_all = (None,self.num_classes)
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=input_shape_all,
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=label_shape_all,
            name='labels_placeholder')
        input_shape_one = (1,self.input_side, self.input_side,self.input_channels)
        label_shape_one = (1,self.num_classes)
        
        return input_placeholder, labels_placeholder, input_shape_all, label_shape_all
    
    
    def get_logits_preds(self, inputs):
        """
        Desc:
            Get logits of models
        """
        preds = self.model(inputs)
        logits, = preds.op.inputs #inputs to the softmax operation
        return logits, preds
    
    def predict(self, x):
        input_shape = (-1,self.input_side, self.input_side,self.input_channels)
        feed_dict = {
                self.input_placeholder: x.reshape(input_shape),
                K.learning_phase(): 0
            } 
        preds = self.sess.run(self.preds, feed_dict=feed_dict)
        return preds
        
    def compile_model(self, dataset='mnist'):
        """
        Initialize the model
        """
        if dataset.lower() == 'cifar10':
            self.model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=self.lr_schedule(0)),
              metrics=['accuracy'])             
        else:
            self.model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
        
    def save_model(self, store_path_model, store_path_weights):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(store_path_model, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(store_path_weights)
        print("Saved model to disk")
 
    def load_model(self, load_path_model, load_path_weights):
        # load json and create model
        json_file = open(load_path_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        # load weights into new model
        self.model.load_weights(load_path_weights)
        print("Loaded model from disk")
        
    def reshape_data(self):
        """
        Desc:
            Reshapes data to original size
        """    
        self.train_data = self.train_data.reshape(-1, self.input_side, self.input_side, self.input_channels)
        self.val_data = self.val_data.reshape(-1, self.input_side, self.input_side, self.input_channels)
        self.test_data = self.test_data.reshape(-1, self.input_side, self.input_side, self.input_channels)
    
        self.train_data = self.train_data.astype('float32')
        self.val_data = self.val_data.astype('float32')
        self.test_data = self.test_data.astype('float32')
  
    
    def lr_schedule(self,epoch):
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr
    
    def mnist_cnn(self):
        layers = [Conv2D(32, (3, 3), padding='valid', input_shape=(self.input_side, self.input_side, self.input_channels), name='conv1'),
                Activation(self.non_linearity),
                Conv2D(64, (3, 3), name='conv2'),
                Activation(self.non_linearity),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(self.dropout_prob),
                Flatten(),
                Dense(128, name='dense1'),
                Activation(self.non_linearity),
                Dropout(self.dropout_prob),
                Dense(self.num_classes, name='logits'),
                Activation('softmax')
            ]
            
        model = Sequential()
        for layer in layers:
            model.add(layer)
                
        return model

        
    def resnet_layer(self,inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

        # Returns
        x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x


    def resnet_v1(self):
        """ResNet Version 1 Model builder [a]

        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
        model (Model): Keras model instance
        """
        depth = 3 * 6 + 2
        num_classes=self.num_classes
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)
        input_shape=(self.input_side, self.input_side, self.input_channels)
        inputs = Input(shape=input_shape)
        x = self.resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = self.resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
                y = self.resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        logits = Dense(num_classes,kernel_initializer='he_normal')(y)
        outputs = Activation('softmax')(logits)
        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model       
        