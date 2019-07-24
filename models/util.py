import numpy as np
import bottleneck as bn
import matplotlib, gc
import matplotlib.pyplot as plt
from tensorflow import gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops
from scipy.fftpack import dct, idct
import cvxpy as cp
import scipy.linalg as la





def get_top_k(X,k):
    idx = bn.argpartition(np.absolute(X), X.size-k, axis=None)[-k:]
    width = X.shape[1]
    idx = np.array([divmod(i, width) for i in idx])
    top = np.zeros((X.shape)) + 1j*0
    top[idx[:,0],idx[:,1]] = X[idx[:,0],idx[:,1]]
         
    return top

def get_matrix(n,tf='dct'):
    if tf=='dft':
        F = la.dft(n,scale='sqrtn')
    elif tf=='dct':
        I = np.identity(n)
        F = dct(I, norm='ortho')
    return F

def get_top_and_bot_k(X,k):
    idx = bn.argpartition(np.absolute(X), X.size-k, axis=None)[-k:]
    width = X.shape[1]
    idx = np.array([divmod(i, width) for i in idx])
    top = np.zeros((X.shape)) + 1j*0
    top[idx[:,0],idx[:,1]] = X[idx[:,0],idx[:,1]]
    bot = X - top
         
    return top,bot



def get_top_bot_k_vec(x,k):
    ind = np.argpartition(np.absolute(x), -k)[-k:]
    temp = np.zeros(x.shape)
    temp[ind] = x[ind]
    return temp, x - temp

def get_topk_vec(x,k):
    ind = np.argpartition(np.absolute(x), -k)[-k:]
    temp = np.zeros(x.shape)
    temp[ind] = x[ind]
    return temp

def iht(y,t, T=100,k=20,transform='dct'):
    x_hat = np.zeros(y.shape) + 1j*0
    e_hat = np.zeros(y.shape) + 1j*0
    if transform == 'dft':               
        for i in range(T):
            x_hat = get_topk_vec(np.fft.fft(y - e_hat, norm='ortho'), k)
            e_hat = get_topk_vec(y - np.fft.ifft(x_hat,norm='ortho'), t)
    elif transform == 'dct':
        for i in range(T):
            x_hat = get_topk_vec(dct((y - e_hat), norm='ortho'), k)
            e_hat = get_topk_vec(y - idct(x_hat, norm='ortho'), t)   
    return x_hat,e_hat


def socp(y, D, n=784, eta=2.7):
    x = cp.Variable(n)
    c = np.zeros((n))
    constraints = [cp.SOC(c.T*x + eta, D*x - y)]

    # Form objective.
    obj = cp.Minimize(cp.norm(x,1))

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()

    return x.value



def dantzig(y, D, n=784, eta_1=0.16, eta_2=6.0):
    x = cp.Variable(n)
    b = cp.Variable(n)
    obj = cp.Minimize(cp.norm(x,1))
    constraints = [ b <= x, -b <= x, D.T*(y - D*b) <= eta_2, -D.T*(y - D*b) <=eta_2, (y-D*b)<=eta_1, (D*b-y) <=eta_1]
    prob = cp.Problem(obj, constraints)

    print("Optimal value", prob.solve(solver='ECOS'))
    return b.value

def avg_l2_dist(orig, adv):
    """Get the mean l2 distortion between two orig and adv images"""
    l2_dist = 0.0
    num_ = orig.shape[0]
    if num_ > 0:
        for i in range(orig.shape[0]):
            l2_dist+= np.linalg.norm(orig[i].flatten() - adv[i].flatten())
        return l2_dist/orig.shape[0]
    else:
        return np.nan

def avg_l0_dist(orig, adv):
    """Get the mean l2 distortion between two orig and adv images"""
    l0_dist = 0
    num_ = orig.shape[0]
    if num_ > 0:
        for i in range(orig.shape[0]):
            diff= np.abs(orig[i] - adv[i])
            l0_dist+=len(np.where(diff.flatten()>0.0001)[0])
        return l0_dist/orig.shape[0]
    else:
        return np.nan
    
    
def avg_linf_dist(orig, adv):
    """Get the mean l2 distortion between two orig and adv images"""
    linf_dist = 0.0
    num_ = orig.shape[0]
    if num_ > 0:
        for i in range(orig.shape[0]):
            linf_dist+= np.linalg.norm(orig[i].flatten() - adv[i].flatten(),ord=np.inf)
        return linf_dist/orig.shape[0]
    else:
        return np.nan
    
def visualize(image_list, num_images, savefig=''):
    """Visualize images in a grid"""
    assert(len(image_list) == num_images)
    fig=plt.figure(figsize=(15,15))
    columns = num_images
    for i in range(1, columns+1):
        img = image_list[i-1]
        
        fig.add_subplot(1, columns, i)
        if img.shape[-1] == 1:
            img = np.squeeze(img)
            plt.imshow(img,cmap='Greys')
        else:
            plt.imshow(img)
        plt.axis('off')    
        
    plt.show()
    fig.savefig(savefig,bbox_inches='tight')

#Normalize rows of a given matrix
def normalize(matrix):
    """Normalize each row vector in a matrix"""
    matrix_nm = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        norm = np.linalg.norm(matrix[i]) 
        if norm > 0:
            matrix_nm[i] = matrix[i]/np.linalg.norm(matrix[i]) 
    return matrix_nm

def preds_to_labels(preds):
    labels = np.zeros(preds.shape)
    labels[np.arange(preds.shape[0]),np.argmax(preds, axis=1)] = 1
    return labels

def get_test_from_train_idx(a, b):
    mask = np.ones_like(a,dtype=bool)
    mask[b] = False
    return a[mask]
