from __future__ import division
from __future__ import print_function

import sys, os
import bottleneck as bn
sys.path.append('../')
from attacks import *
from PIL import Image

import tensorflow as tf
from keras.datasets import cifar10, mnist, fashion_mnist
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import CarliniWagnerL2, CarliniWagnerL0, DeepFool, SaliencyMapMethod,FastGradientMethod,BasicIterativeMethod

import numpy as np
import os, time, math, gc
from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils
from util import *
from scipy.fftpack import dct, idct

class NeuralNetwork(object):
    """General Neural Network Class for multi-class classification """
    SEED = 14
    
    def __init__(self, model_name=None, dataset='mnist', project=True, transform='dct', batch_size=128, load_from_file=False, load_model_path='', load_weights_path='', seed=14):
        """
        Desc:
            Constructor
        
        Params:
            model_name(str): Name of model (will be saved as such)
            dataset(str): Name of dataset to load - also determines which model will be loaded
            batch_size(int): Batch size to be used during training
            load_from_file(bool): load parameters of model from file
            model_chkpt_file: tf model file containing params
                  
        """
        SEED = seed
        #Reproduciblity of experiments
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        
        
        self.model_name = model_name
        self.dataset = dataset
        self.transform = transform
        
        if dataset.lower() == 'mnist':
            self.num_classes = 10
            self.batch_size = batch_size
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('mnist', project)
        
        elif dataset.lower() == 'fashion_mnist':
            self.num_classes = 10
            self.batch_size = batch_size
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('fashion_mnist', project)
            
        elif dataset.lower() == 'cifar10':
            self.num_classes = 10
            self.batch_size = 32
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('cifar10', project)
            
        elif dataset.lower() == 'cifar10-big':
            self.num_classes = 10
            self.batch_size = 32
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('cifar10-big', project)
            
                      
        
        
        # Initialize Tf and Keras
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.set_session(self.sess)
       
        #Dropout and L2 reg hyperparams
        self.beta = 0.01
        self.dropout_prob = 0.5
        
        #Setup Input Placeholders required for forward pass, implemented by child classes (eg: CNN or FCN)
        self.input_placeholder, self.labels_placeholder, self.input_shape, self.label_shape = self.create_input_placeholders() 
                
        # Setup model operation implemented by child classes
        self.model = self.create_model(dataset.lower())
        self.logits, self.preds = self.get_logits_preds(self.input_placeholder)
        self.params = self.get_params()
        
        #Get total number of params in model
        num_params = 0
        for j in range(len(self.params)):
            num_params = num_params + int(np.product(self.params[j].shape))
        self.num_params = num_params

        # Setup loss 
        self.training_loss = self.get_loss_op(self.logits, self.labels_placeholder)

        # Setup gradients 
        self.grad_loss_wrt_param = tf.gradients(self.training_loss, self.params)
        self.grad_loss_wrt_input = tf.gradients(self.training_loss, self.input_placeholder)  
        
        
        #Load model parameters from file or initialize 
        if load_from_file == False:
            if dataset.lower() == 'cifar10':
                self.compile_model(dataset)
            else:
                self.compile_model()
        else:
            self.load_model(load_model_path, load_weights_path)
        return

    def project_images(self,X,k):
        n = X.shape[1]
        x_rec = np.zeros((X.shape))
        
        #Transform data
        if self.transform == 'dct':
            for i in range(X.shape[0]):
                f_x = dct(X[i,:,:,0].flatten(),norm='ortho')
                f_x = f_x.reshape(int(n),int(n))
                top_k = get_top_k(f_x,k=k)
                f_recon = idct(top_k.flatten(),norm='ortho').reshape(int(n),int(n))
                x_rec[i,:,:,0]= f_recon                       
        
        elif self.transform == 'dct-matrix':
            #Form the DCT matrix
            D = get_matrix(n*n,tf='dct')          
            for i in range(X.shape[0]):
                f_x = np.dot(D,X[i,:,:,0].flatten())
                f_x = f_x.reshape(int(n),int(n))
                top_k = get_top_k(f_x,k=k)
                f_recon = np.dot(D.T,top_k.flatten()).reshape(int(n),int(n))               
                x_rec[i,:,:,0]= f_recon
                
        elif self.transform=='dct-2d':
            for i in range(X.shape[0]):
                f_x = dct(dct(X[i,:,:,0].T, norm='ortho').T,norm='ortho')
                top_k = get_top_k(f_x,k=k)
                f_recon = idct(idct(top_k.T, norm='ortho').T,norm='ortho')
                x_rec[i,:,:,0]= f_recon
                
        elif self.transform=='dct-3d':
            #To speed up using pool
            for i in range(X.shape[0]):
                f_x_r = dct(X[i,:,:,0].flatten(),norm='ortho').reshape(int(n),int(n))
                f_x_g = dct(X[i,:,:,1].flatten(),norm='ortho').reshape(int(n),int(n))
                f_x_b = dct(X[i,:,:,2].flatten(),norm='ortho').reshape(int(n),int(n))
                top_k_r = get_top_k(f_x_r,k=k)
                top_k_g = get_top_k(f_x_g,k=k)
                top_k_b = get_top_k(f_x_b,k=k)
                f_recon_r = idct(top_k_r.flatten(),norm='ortho').reshape(int(n),int(n))
                f_recon_g = idct(top_k_g.flatten(),norm='ortho').reshape(int(n),int(n))
                f_recon_b = idct(top_k_b.flatten(),norm='ortho').reshape(int(n),int(n))
  
                x_rec[i,:,:,0]= f_recon_r
                x_rec[i,:,:,1]= f_recon_g
                x_rec[i,:,:,2]= f_recon_b

        x_all = np.concatenate((X,x_rec), axis=0)
        
        return x_all

            
    def load_dataset(self, dataset='mnist', project=True,k=40):
        """
        Desc: Load the required dataset into the model
        """
        
        if dataset.lower() == 'mnist':
            (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
            X_train = X_train.reshape(-1, 28, 28, 1)
            X_test = X_test.reshape(-1, 28, 28, 1)
            
            Y_train = np_utils.to_categorical(Y_train, 10)
            Y_test = np_utils.to_categorical(Y_test, 10)
            
            self.input_side = 28
            self.input_channels = 1
            self.input_dim = self.input_side * self.input_side * self.input_channels
                                 
            
        elif dataset.lower() == 'fashion_mnist':
            (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
            X_train = X_train.reshape(-1, 28, 28, 1)
            X_test = X_test.reshape(-1, 28, 28, 1)
            
            Y_train = np_utils.to_categorical(Y_train, 10)
            Y_test = np_utils.to_categorical(Y_test, 10)
            
            self.input_side = 28
            self.input_channels = 1
            self.input_dim = self.input_side * self.input_side * self.input_channels
            
            
        elif dataset.lower() == 'cifar10':
            (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
            
            Y_train = np_utils.to_categorical(Y_train, 10)
            Y_test = np_utils.to_categorical(Y_test, 10)
            
            self.input_side = 32
            self.input_channels = 3
            self.input_dim = self.input_side * self.input_side * self.input_channels

            
        elif dataset.lower() == 'cifar10-big':
            (X_train_sm, Y_train), (X_test_sm, Y_test) = cifar10.load_data()
            X_train_sm = X_train_sm.reshape(-1, 32, 32, 3)
            X_test_sm = X_test_sm.reshape(-1, 32,32, 3)
            
            Y_train = np_utils.to_categorical(Y_train, 10)
            Y_test = np_utils.to_categorical(Y_test, 10)
            
            X_train = np.zeros((X_train_sm.shape[0],125,125,3))
            X_test = np.zeros((X_test_sm.shape[0],125,125,3))
            
            for i in range(X_train.shape[0]):
                img = X_train_sm[i,:,:,:]
                img = Image.fromarray(img)
                basewidth = 125
                wpercent = (basewidth/float(img.size[0]))
                hsize = int((float(img.size[1])*float(wpercent)))
                img = img.resize((basewidth,hsize), Image.ANTIALIAS)
                X_train[i,:,:,:] = np.asarray(img)
                
            for i in range(X_test.shape[0]):
                img = X_test_sm[i,:,:,:]
                img = Image.fromarray(img)
                basewidth = 125
                wpercent = (basewidth/float(img.size[0]))
                hsize = int((float(img.size[1])*float(wpercent)))
                img = img.resize((basewidth,hsize), Image.ANTIALIAS)
                X_test[i,:,:,:] = np.asarray(img)
            
            self.input_side = 125
            self.input_channels = 3
            self.input_dim = self.input_side * self.input_side * self.input_channels
                      
    
        #Normalize data
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        
        if 'cifar' in dataset.lower():
            X_train_mean = np.mean(X_train, axis=0)
            X_train -= X_train_mean
            #X_test -= X_train_mean
        
        num_val = int(X_test.shape[0]/2.0)
        
        #Get validation sets as well
        val_indices = np.random.choice(range(X_test.shape[0]), num_val)
        X_val = X_test[val_indices]
        Y_val = Y_test[val_indices]
    
        mask = np.ones(X_test.shape[0], dtype=bool)
        mask[val_indices] = False
    
        X_test = X_test[mask]
        Y_test = Y_test[mask]
        
        if project:
            if dataset=='cifar10':
                k = 75
            elif dataset == 'cifar10-big':
                k = 275
            elif 'mnist' in dataset.lower():
                k = 40
            
            #Project and concatenate data
            X_train = self.project_images(X_train,k)
            X_val = self.project_images(X_val,k)
                         
            y_train = np.zeros(Y_train.shape)                
            y_train[:,:] = Y_train[:,:]       
            y_train_all = np.concatenate((Y_train,y_train), axis=0)
                         
            y_val = np.zeros(Y_val.shape)
                         
            y_val[:,:] = Y_val[:,:]
            y_val_all = np.concatenate((Y_val,y_val), axis=0)
            
            return X_train, y_train_all, X_val, y_val_all, X_test, Y_test
        else:
            return X_train, Y_train, X_val, Y_val, X_test, Y_test
    

    def get_loss_op(self, logits, labels):
        """
        Desc:
            Create operation used to calculate loss during network training 
        """
        out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return out
    
    def train(self, epochs, use_aug=False):
        """
        Desc:
            Trains model for a specified number of epochs.
        """    
        if use_aug:
            
            datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(self.train_data)

            # Fit the model on the batches generated by datagen.flow().
            self.model.fit_generator(datagen.flow(self.train_data, self.train_labels, batch_size=self.batch_size),
                        validation_data=(self.val_data, self.val_labels),
                        epochs=epochs, steps_per_epoch=len(self.train_data)/self.batch_size,verbose=1,
                        callbacks=callbacks) 
                        
            self.model.evaluate(self.test_data, self.test_labels, batch_size=self.batch_size)


        else:
            self.model.fit(
                self.train_data, 
                self.train_labels,
                epochs=epochs,
                batch_size=128,
                validation_data=(self.val_data, self.val_labels),
                verbose=1,
                shuffle=True
            )

            self.model.evaluate(self.test_data, self.test_labels, batch_size=self.batch_size)
        

    def get_gradients_wrt_params(self, X, Y):
        """Get gradients of Loss(X,Y) wrt network params"""
        
        num_params = self.num_params
        inp_size = X.shape[0]
        gradient = np.zeros((inp_size, num_params), dtype=np.float32)
        
        
        #Get one gradient at a time
        for i in range(inp_size):
            grad = self.sess.run(
                    self.grad_loss_wrt_param,
                    feed_dict={
                        self.input_placeholder: X[i].reshape(self.input_shape),                    
                        self.labels_placeholder: Y[i].reshape(self.label_shape),
                        K.learning_phase(): 0
                        }
                    )
            temp = np.array([])
            for j in range(len(grad)):
                layer_size = np.prod(grad[j].shape) 
                temp = np.concatenate((temp, np.reshape(grad[j], (layer_size))), axis=0)
        
            gradient[i,:] = temp
        return gradient
    
    def get_gradients_wrt_input(self, X, Y):
        """Get gradients of Loss(X,Y) wrt input"""
        
        inp_size = X.shape[0]
        inp_shape = np.prod(X[0].shape)
        gradient = np.zeros((inp_size, inp_shape), dtype=np.float32)
        
        #Get one gradient at a time
        for i in range(inp_size):
            grad = self.sess.run(
                    self.grad_loss_wrt_input,
                    feed_dict={
                        self.input_placeholder: X[i].reshape(self.input_shape),                    
                        self.labels_placeholder: Y[i].reshape(self.label_shape),
                        K.learning_phase(): 0
                        }
                    )[0]
            gradient[i,:] = grad.reshape(inp_shape)
        return gradient
    
        
    def get_adversarial_version(self, x, y=None, eps=0.3, iterations=100,attack='FGSM', targeted=False,y_tar=None,clip_min=0.0, clip_max = 1.0, nb_candidate=10, num_params=100):
        """
        Desc:
            Caclulate the adversarial version for point x using FGSM
            x: matrix of n x input_shape samples
            y: matrix of n x input_label samples
            eps: used for FGSM
            attack: FGMS or CW
        
        """
        if self.dataset == 'cifar10':
            model = KerasModelWrapper(self.model)
        else:
            model = KerasModelWrapper(self.model.model)
        if attack == 'CW-l2':
            K.set_learning_phase(0)
            # Instantiate a CW attack object
            cw = CarliniWagnerL2(model, sess=self.sess)
            
            
            cw_params = {'batch_size':10,
                 'confidence':0,
                'learning_rate':1e-2,
                'binary_search_steps':5,
                'max_iterations':iterations,
                'abort_early':True,
                'initial_const':1e-4,
                'clip_min':0.0,
                'clip_max':1.0}    

            x_adv = cw.generate_np(x,**cw_params)
            
        elif attack == 'CW-l0':
            K.set_learning_phase(0)
            # Instantiate a CW attack object
            cw = CarliniWagnerL0(model, sess=self.sess)

            cw_params = {'batch_size':1,
                 'confidence':0.,
                'learning_rate':1e-2,
                'binary_search_steps':5,
                'max_iterations':iterations,
                'abort_early':True,
                'initial_const':1e-4,
                'clip_min':0.0,
                'clip_max':1.0}    

            x_adv = cw.generate_np(x,**cw_params)
            
        elif attack == 'DF':
            K.set_learning_phase(0)
            df = DeepFool(model, sess=self.sess)
            df_params = {'nb_candidate': nb_candidate}
            x_adv = df.generate_np(x,**df_params)
        
        elif attack == 'JSMA':
            K.set_learning_phase(0)
            jsma = SaliencyMapMethod(model,sess=self.sess)
            jsma_params = {'theta': 1., 
                           'gamma': 0.03,
                           'clip_min': clip_min, 
                           'clip_max': clip_max,
                           'y_target': y_tar}
            x_adv = jsma.generate_np(x, **jsma_params)
            
        elif attack == 'FGSM':
            K.set_learning_phase(0)
            fgsm = FastGradientMethod(model,sess=self.sess)
            fgsm_params = {'eps':0.15,
                           'clip_min': clip_min, 
                           'clip_max': clip_max,
                           'y_target': y_tar}
            x_adv = fgsm.generate_np(x, **fgsm_params)
        
        elif attack == 'BIM':
            K.set_learning_phase(0)
            fgsm = BasicIterativeMethod(model,sess=self.sess)
            fgsm_params = {'eps':0.015,
                            'eps_iter':0.005,
                           'nb_iter':100,
                           'clip_min': clip_min, 
                           'clip_max': clip_max,
                           'y_target': y_tar}
            x_adv = fgsm.generate_np(x, **fgsm_params)
        
            
        return x_adv
        
    
    def generate_perturbed_data(self, x, y=None, eps=0.3, iterations=100,seed=SEED, perturbation='FGSM',nb_candidate=10):
        """
        Generate a perturbed data set using FGSM, CW, or random uniform noise.
        x: n x input_shape matrix
        y: n x input_labels matrix
        seed: seed to use for reproducing experiments
        perturbation: FGSM, CW, or Noise.
        
        return:
        x_perturbed: perturbed version of x
        """
        if perturbation == 'CW-l0':
            x_perturbed = self.get_adversarial_version(x,y,attack='CW-l0')  
        elif perturbation == 'CW-l2':
            x_perturbed = self.get_adversarial_version(x,y,attack='CW-l2') 
        elif perturbation == 'DF':
            x_perturbed = self.get_adversarial_version(x,y,attack='DF', nb_candidate=nb_candidate)
        elif perturbation == 'JSMA':
            x_perturbed = self.get_adversarial_version(x,y,attack='JSMA')
        elif perturbation == 'FGSM':
            x_perturbed = self.get_adversarial_version(x,y,attack='FGSM')
        elif perturbation == 'BIM':
            x_perturbed = self.get_adversarial_version(x,y,attack='BIM')
        return x_perturbed
    
    
 
    #Random sampling from dataset
    def gen_rand_indices_all_classes(self, y=None, seed=SEED,num_samples=10):
        """
           Generate random indices to be used for sampling points
           y: n x label_shape matrix containing labels
           
        """
        if y is not None:
            np.random.seed(seed)
            all_class_indices = list()
            for c_ in range(self.num_classes):
                class_indices = self.gen_rand_indices_class(y,class_=c_,num_samples=num_samples) 
                all_class_indices[c_*num_samples: c_*num_samples+num_samples] = class_indices[:]
            
            return all_class_indices
        else:
            print ('Please provide training labels')
            return
        
    def gen_rand_indices_class(self, y=None, class_=0, num_samples=10):
        """
        Generate indices for the given class
        """
        if y is not None:
            c_indices = np.random.choice(np.where(np.argmax(y,axis=1) == class_)[0], num_samples)
            return c_indices
        else:
            print ('Please provide training labels')
        
    def gen_rand_indices(self, low=0, high=1000,seed=SEED, num_samples=1000):
        """
        Randomly sample indices from a range
        """
        np.random.seed(seed)
        indices = np.random.choice(range(low,high), num_samples)
        return indices
        