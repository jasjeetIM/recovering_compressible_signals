from __future__ import division
from __future__ import print_function

import sys, os
import bottleneck as bn
sys.path.append('../')
from attacks import *
from PIL import Image

import tensorflow as tf
from keras.datasets import cifar10, mnist, fashion_mnist
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import CarliniWagnerL2, CarliniWagnerL0, BasicIterativeMethod, DeepFool, SaliencyMapMethod, FastGradientMethod, MadryEtAl

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
    
    def __init__(self, model_name=None, dataset='mnist', project=True, transform='dft', batch_size=512, initial_learning_rate=8e-1, load_from_file=False, load_model_path='', load_weights_path='', seed=14):
        """
        Desc:
            Constructor
        
        Params:
            model_name(str): Name of model (will be saved as such)
            dataset(str): Name of dataset to load - also determines which model will be loaded
            batch_size(int): Batch size to be used during training
            initial_learning_rate(float): Learning rate to start training the model. 
            load_from_file(bool): load parameters of model from file
            model_chkpt_file: tf model file containing params
                  
        """
        SEED = seed
        #Reproduciblity of experiments
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.transform = transform
        self.initial_learning_rate = initial_learning_rate
        
        if dataset.lower() == 'mnist':
            self.num_classes = 10
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('mnist', project)
        
        elif dataset.lower() == 'fashion_mnist':
            self.num_classes = 10
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('fashion_mnist', project)
            
        elif dataset.lower() == 'mnist-big':
            self.num_classes = 10
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('mnist-big', project)
        
        elif dataset.lower() == 'fashion_mnist-big':
            self.num_classes = 10
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('fashion_mnist-big', project)
                      
        elif 'cifar' in dataset.lower():
            self.num_classes = 10
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('cifar10', project)
        
        
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
            self.compile_model()
        else:
            self.load_model(load_model_path, load_weights_path)
        return

    def project_images(self,X,k):
        n = X.shape[1]
        x_rec = np.zeros((X.shape))
        if self.transform == 'dft':
            for i in range(X.shape[0]):
                f_x = np.fft.fft2(X[i,:,:,0], norm='ortho')
                top_k = get_top_k(f_x,k=k)
                f_recon = np.fft.ifft2(top_k,norm='ortho')
                x_rec[i,:,:,0]= f_recon
        elif self.transform == 'dct-matrix':
            #Form the DCT matrix
            D = np.zeros((n*n,n*n))
            for p in range(n*n):
                for q in range(n*n):
                    if p == 0:
                        D[p,q] = 1/np.sqrt(float(n*n))
                    else:
                        D[p,q] = np.sqrt(2/float(n*n))*(  math.cos(  (math.pi*(2*q + 1)*p) / (2*float(n*n))   )    )
        
            for i in range(X.shape[0]):
                f_x = np.dot(D,X[i,:,:,0].flatten())
                f_x = f_x.reshape(int(n),int(n))
                top_k = get_top_k(f_x,k=k)
                f_recon = np.dot(D.T,top_k.flatten()).reshape(int(n),int(n))  
             
                x_rec[i,:,:,0]= f_recon
        elif self.transform=='dct':
            for i in range(X.shape[0]):
                f_x = dct(dct(X[i,:,:,0].T, norm='ortho').T,norm='ortho')
                top_k = get_top_k(f_x,k=k)
                f_recon = idct(idct(top_k.T, norm='ortho').T,norm='ortho')
                x_rec[i,:,:,0]= f_recon
        
        x_all = np.concatenate((X,x_rec), axis=0)
        
        return x_all

            
    def load_dataset(self, dataset='mnist', project=True,k=40):
        """
        Desc: Load the required dataset into the model
        """
        
        if dataset == 'mnist':
            (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
            X_train = X_train.reshape(-1, 28, 28, 1)
            X_test = X_test.reshape(-1, 28, 28, 1)
            
            Y_train = np_utils.to_categorical(Y_train, 10)
            Y_test = np_utils.to_categorical(Y_test, 10)
            
            self.input_side = 28
            self.input_channels = 1
            self.input_dim = self.input_side * self.input_side * self.input_channels
            
           
                       
            
        elif dataset == 'fashion_mnist':
            (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
            X_train = X_train.reshape(-1, 28, 28, 1)
            X_test = X_test.reshape(-1, 28, 28, 1)
            
            Y_train = np_utils.to_categorical(Y_train, 10)
            Y_test = np_utils.to_categorical(Y_test, 10)
            
            self.input_side = 28
            self.input_channels = 1
            self.input_dim = self.input_side * self.input_side * self.input_channels
            
        elif dataset == 'mnist-big':
            (X_train_sm, Y_train), (X_test_sm, Y_test) = mnist.load_data()
            X_train_sm = X_train_sm.reshape(-1, 28, 28, 1)
            X_test_sm = X_test_sm.reshape(-1, 28, 28, 1)
            
            Y_train = np_utils.to_categorical(Y_train, 10)
            Y_test = np_utils.to_categorical(Y_test, 10)
            
            X_train = np.zeros((X_train_sm.shape[0],200,200,1))
            X_test = np.zeros((X_test_sm.shape[0],200,200,1))
            
            for i in range(X_train.shape[0]):
                img = X_train_sm[i,:,:,0]
                img = Image.fromarray(img)
                basewidth = 200
                wpercent = (basewidth/float(img.size[0]))
                hsize = int((float(img.size[1])*float(wpercent)))
                img = img.resize((basewidth,hsize), Image.ANTIALIAS)
                X_train[i,:,:,0] = np.asarray(img)
                
            for i in range(X_test.shape[0]):
                img = X_test_sm[i,:,:,0]
                img = Image.fromarray(img)
                basewidth = 200
                wpercent = (basewidth/float(img.size[0]))
                hsize = int((float(img.size[1])*float(wpercent)))
                img = img.resize((basewidth,hsize), Image.ANTIALIAS)
                X_test[i,:,:,0] = np.asarray(img)
            
            self.input_side = 200
            self.input_channels = 1
            self.input_dim = self.input_side * self.input_side * self.input_channels
        
        elif dataset == 'fashion_mnist-big':
            (X_train_sm, Y_train), (X_test_sm, Y_test) = fashion_mnist.load_data()
            X_train_sm = X_train_sm.reshape(-1, 28, 28, 1)
            X_test_sm = X_test_sm.reshape(-1, 28, 28, 1)
            
            Y_train = np_utils.to_categorical(Y_train, 10)
            Y_test = np_utils.to_categorical(Y_test, 10)
            
            X_train = np.zeros((X_train_sm.shape[0],200,200,1))
            X_test = np.zeros((X_test_sm.shape[0],200,200,1))
            
            for i in range(X_train.shape[0]):
                img = X_train_sm[i,:,:,0]
                img = Image.fromarray(img)
                basewidth = 200
                wpercent = (basewidth/float(img.size[0]))
                hsize = int((float(img.size[1])*float(wpercent)))
                img = img.resize((basewidth,hsize), Image.ANTIALIAS)
                X_train[i,:,:,0] = np.asarray(img)
                
            for i in range(X_test.shape[0]):
                img = X_test_sm[i,:,:,0]
                img = Image.fromarray(img)
                basewidth = 200
                wpercent = (basewidth/float(img.size[0]))
                hsize = int((float(img.size[1])*float(wpercent)))
                img = img.resize((basewidth,hsize), Image.ANTIALIAS)
                X_test[i,:,:,0] = np.asarray(img)
            
            self.input_side = 200
            self.input_channels = 1
            self.input_dim = self.input_side * self.input_side * self.input_channels
            
          
        elif dataset == 'cifar10':
            (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
            X_train = X_train.reshape(-1, 32, 32, 3)
            X_test = X_test.reshape(-1, 32, 32, 3)
            Y_train = Y_train.reshape((Y_train.shape[0],))
            Y_test = Y_test.reshape((Y_test.shape[0],))
                       
            #Convert to one hot
            Y_train = np_utils.to_categorical(Y_train, 10)
            Y_test = np_utils.to_categorical(Y_test, 10)
            
            self.input_side = 32
            self.input_channels = 3
            self.input_dim = self.input_side * self.input_side * self.input_channels
            
            
    
        #Normalize data
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        
     
        
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
            X_train = self.project_images(X_train,k)
            X_val = self.project_images(X_val,k)
                         
        y_train = np.zeros(Y_train.shape)                
        y_train[:,:] = Y_train[:,:]
        y_train_all = np.concatenate((Y_train,y_train), axis=0)
                         
        y_val = np.zeros(Y_val.shape)
                         
        y_val[:,:] = Y_val[:,:]
        y_val_all = np.concatenate((Y_val,y_val), axis=0)
            
        return X_train, y_train_all, X_val, y_val_all, X_test, Y_test
    

    def get_loss_op(self, logits, labels):
        """
        Desc:
            Create operation used to calculate loss during network training and for influence calculations. 
        """
        out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return out
    
    def train(self, epochs):
        """
        Desc:
            Trains model for a specified number of epochs.
        """    
       
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
    
        
    def get_adversarial_version(self, x, y=None, eps=0.3, iterations=100,attack='FGSM', targeted=False, x_tar=None, x_labs=None,y_tar=None,clip_min=0.0, clip_max = 1.0, use_cos_norm_reg=False, use_logreg=False, num_logreg=0,nb_candidate=10, train_grads=None, num_params=100):
        """
        Desc:
            Caclulate the adversarial version for point x using FGSM
            x: matrix of n x input_shape samples
            y: matrix of n x input_label samples
            eps: used for FGSM
            attack: FGMS or CW
        
        """
        
        if attack == 'CW-l2':
            K.set_learning_phase(0)
            # Instantiate a CW attack object
            model = KerasModelWrapper(self.model.model)
            cw = CarliniWagnerL2(model, sess=self.sess)
            yname = 'y'
            
            cw_params = {'batch_size':10,
                 'confidence':0,
                'learning_rate':0.1,
                'binary_search_steps':5,
                'max_iterations':100,
                'abort_early':True,
                'initial_const':1e-2,
                'clip_min':0,
                'clip_max':1}    

            x_adv = cw.generate_np(x,**cw_params)
            
        elif attack == 'CW-l0':
            K.set_learning_phase(0)
            # Instantiate a CW attack object
            model = KerasModelWrapper(self.model.model)
            cw = CarliniWagnerL0(model, sess=self.sess)
            yname = 'y'
            
            cw_params = {'batch_size':10,
                 'confidence':0,
                'learning_rate':0.1,
                'binary_search_steps':5,
                'max_iterations':100,
                'abort_early':True,
                'initial_const':1e-4,
                'clip_min':0,
                'clip_max':1}    

            x_adv = cw.generate_np(x,**cw_params)
            
        elif attack == 'DF':
            K.set_learning_phase(0)
            model = KerasModelWrapper(self.model.model)
            df = DeepFool(model, sess=self.sess)
            df_params = {'nb_candidate': nb_candidate}
            x_adv = df.generate_np(x,**df_params)

        elif attack == 'JSMA':
            K.set_learning_phase(0)
            model = KerasModelWrapper(self.model.model)
            jsma = SaliencyMapMethod(model,sess=self.sess)
            jsma_params = {'theta': 1., 
                           'gamma': 0.03,
                           'clip_min': clip_min, 
                           'clip_max': clip_max,
                           'y_target': y_tar}
            x_adv = jsma.generate_np(x, **jsma_params)
            
        elif attack == 'FGSM':
            K.set_learning_phase(0)
            # Instantiate a CW attack object
            fgsm_model = KerasModelWrapper(self.model.model)
            fgsm = FastGradientMethod(fgsm_model, sess=self.sess)
           
            fgsm_params = {'eps':0.3,
                       'ord':np.inf,
                       'y':None,
                       'y_target':None,
                       'clip_min':0.0,
                       'clip_max':1.0,
                       'sanity_checks':True}
            x_adv = fgsm.generate_np(x,**fgsm_params)     


        elif attack == 'BIM':
            K.set_learning_phase(0)
            # Instantiate a CW attack object
            bim_model = KerasModelWrapper(self.model.model)
            bim = BasicIterativeMethod(bim_model, sess=self.sess)
            bim_params = { 'eps':0.3,
                   'eps_iter':0.03,
                   'nb_iter':10,
                   'y':None,
                   'ord':np.inf,
                   'clip_min':0.0,
                   'clip_max':1.0,
                   'y_target':None,
                   'sanity_checks':True}
            x_adv = bim.generate_np(x,**bim_params)     
            
        elif attack == 'MADRY':
            K.set_learning_phase(0)
            # Instantiate a CW attack object
            madry_model = KerasModelWrapper(self.model.model)
            madry = MadryEtAl(madry_model, sess=self.sess)
            yname = 'y'
            madry_params = { 'eps':0.3,
                   'eps_iter':0.03,
                   'nb_iter':10,
                   'y':None,
                   'ord':np.inf,
                   'clip_min':0.0,
                   'clip_max':1.0,
                   'y_target':None,
                   'sanity_checks':True}
            x_adv = madry.generate_np(x,**madry_params)     
  

        
            
        return x_adv
        
    
    def generate_perturbed_data(self, x, y=None, eps=0.3, iterations=100,seed=SEED, perturbation='FGSM', targeted=False, x_tar=None,y_tar=None, x_labs = None, nb_candidate=10, train_grads=None, num_params=100):
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
            x_perturbed = self.get_adversarial_version(x,y,attack='CW-l0', eps=eps)  
        elif perturbation == 'CW-l2':
            x_perturbed = self.get_adversarial_version(x,y,attack='CW-l2', eps=eps) 
        elif perturbation == 'DF':
            x_perturbed = self.get_adversarial_version(x,y,attack='DF', nb_candidate=nb_candidate)
        elif perturbation == 'JSMA':
            x_perturbed = self.get_adversarial_version(x,y,attack='JSMA')
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
        