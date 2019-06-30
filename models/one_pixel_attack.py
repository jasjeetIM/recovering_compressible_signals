#!/usr/bin/env python3
#Modified from #https://github.com/Hyperparticle/one-pixel-attack-keras/


import numpy as np
from differential_evolution import differential_evolution

class OnePixelAttack(object):
    def __init__(self,dimensions=(32,32,3)):
        self.dimensions=dimensions
    
    def predict_classes(self, xs, img, label, model,minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = self.perturb_image(xs, img)
        predictions = model.predict(imgs_perturbed)[:,label]
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions

    def attack_success(self, x, img, label, model):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = self.perturb_image(x, img)

        confidence = model.predict(attack_image)[0]
        predicted_class = np.argmax(confidence)
        
        # If the prediction is what we want return True
        if predicted_class != label:
            return True

    def attack(self, img, model, label,pixel_count=1,maxiter=75, popsize=400):
        # Define bounds for a flat vector of x,y,r,g,b values
        dim_x, dim_y, channels = self.dimensions
        if channels == 3:
            bounds = [(0,dim_x), (0,dim_y), (0,256), (0,256), (0,256)] * pixel_count
        else:
            bounds = [(0,dim_x), (0,dim_y), (0,256)] * pixel_count


        
        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))
        
        # Format the predict/callback functions for the differential evolution algorithm
        predict_fn = lambda xs: self.predict_classes(xs, img, label, model)
        callback_fn = lambda x, convergence: self.attack_success(
            x, img, label, model)
        
        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)
        # Calculate some useful statistics to return from this function
        attack_image = self.perturb_image(attack_result.x, img)[0]
        prior_probs = model.predict(np.expand_dims(img,axis=0))[0]
        predicted_probs = model.predict(np.expand_dims(attack_image,axis=0))[0]
        predicted_class = np.argmax(predicted_probs)
        success = predicted_class != label
        cdiff = prior_probs[label] - predicted_probs[label]


        return [predicted_class,attack_image]

    def attack_all(self, model, images,labels,maxiter=75, popsize=400):
        adv_images = np.zeros(images.shape)
        adv_preds = np.zeros(labels.shape)
        
        for i in range(images.shape[0]):
            adv_preds[i],adv_images[i], = self.attack(images[i], model, np.argmax(labels[i]))
        return adv_images,adv_preds
    
    def perturb_image(self, x, img):
        x = x.astype(int)
        if len(x.shape) < 2:
            x = x.reshape((-1,x.shape[0]))
        tile = [len(x)] + [1]*(x.ndim+1)
        imgs = np.tile(img, tile)
        pixels = np.split(x.flatten(), (x.shape[0]*x.shape[1])/5)
        for xs,img in zip(pixels,imgs):
            img[xs[0],xs[1],:] = xs[2:]                
        return imgs