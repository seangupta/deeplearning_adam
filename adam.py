from __future__ import division
import numpy as np
import struct as st
from matplotlib import pyplot as plt
import cProfile
import scipy.special

# #displaying images
# for n in range(10):                
#     y = images_array[n,  :]
#     plt.figure()
#     plt.imshow(np.reshape(y, (28,28)),interpolation="None",cmap='gray')
#     plt.axis('off')

N = 100 #training set size
m = 100 #mini-batch size
#number of iterations until 500 passes through training set have been completed
tau = int(N/m * 500) 


#load data
X = images_array.reshape((nImg,nR * nC))[:N,:]/255
H = 30

def sigmoid(x):
    '''avoids overflow in calculating sigmoid'''
    '''expects col vector as input'''
    # a, b = np.shape(x)
    # y = np.zeros(np.shape(x))
    # for i in range(a):
    #     for j in range(b):
    #             y[i,j] = np.exp(x[i,j])/(1 + np.exp(x[i,j])) if x[i,j] < 0 else 1/(1 + np.exp(-x[i,j]))
    # return y
    return scipy.special.expit(x) #this is much faster than my explicit implementation

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
    
def relu(x):
    return x * (x > 0)

def relu_prime(x):
    return x >= 0

sigma = sigmoid
sigma_prime = sigmoid_prime #maybe change to relu
phi = sigmoid 
phi_prime = sigmoid_prime

def L_sq(x,y):
    '''squared loss'''
    return np.square(x - y) / 2
    
def L_sq_prime(x,y):
    '''derivative of squared loss'''
    return y - x

def L_lik(x,y):
    '''likelihood loss'''
    return -x * np.log(y) - (1 - x) * np.log(1 - y)
    
def L_lik_prime(x,y):
    '''derivative of likelihood loss'''
    return -x / y + (1 - x) / np.max(np.hstack([1 - y,10**-16 * np.ones((y.shape[0],1))]), axis = 1)

def E(X,W1,W2,b1,b2, pred = False):
    '''returns error on all examples in X (cols are examples)'''
    
    #print "W1 = ", W1
    #print "X = ", X
    
    f1 = np.matmul(W1,X)
    #print "f[0] = ", f1 #BECOMING TOO LARGE
    f2 = b1 + f1
    f3 = sigma(f2) #use same transfer function
    f4 = np.matmul(W2,f3)
    f5 = b2 + f4
    f6 = sigma(f5)
    
    if pred:
        return f6
    
    f7 = L_sq(X,f6)
    f8 = np.sum(f7)
    
    f = (f1, f2, f3, f4, f5, f6, f7, f8)
    
    #for i in range(8):
        #print("f[%s] = %s" %(i,f[i]))
    
    return f

def adam(X, beta1 = 0.9, beta2 = 0.999, gamma = 10**(-8), maxiter = 1000):
    '''runs adaptive moment optimiser in autoencoder with data X.
    returns optimal parameters'''
    
    eps0 = 50 #main parameter to tune
    eps_tau = eps0/100   
    
    prev_loss = float("inf")
    # m_W1 = np.zeros((H,784)) #moving averages 
    # v_W1 = np.zeros((H,784))
    # m_W2 = np.zeros((784,H))
    # v_W2 = np.zeros((784,H))
    # m_b1 = np.zeros((H,1))
    # v_b1 = np.zeros((H,1))
    # m_b2 = np.zeros((784,1))
    # v_b2 = np.zeros((784,1))
    
    #initialise parameters
    W1 = np.random.randn(H,784)/np.sqrt(784)
    W2 = np.random.randn(784,H)/np.sqrt(784)
    b1 = np.random.randn(H,1)/np.sqrt(784)
    b2 = np.random.randn(784,1)/np.sqrt(784)
    
    for k in range(1, maxiter + 1):
        
        if k % 100 == 1: #TODO: adjust when check for convergence is done
            print "k = ", k
            #compute loss across entire training set
            curr_loss = E(X.T,W1,W2,b1,b2)[-1]/N
            print "loss = ", curr_loss
            if abs(prev_loss - curr_loss) < 10**(-5): #return if converged
                break
            else:
                prev_loss = curr_loss

        #sample minibatch
        #indices = np.random.choice(range(N), size = m, replace = False)
        #X_mini = X[indices]
        X_mini = X
               
        #forward pass
        f = E(X_mini.T,W1,W2,b1,b2)

        #backward pass
        t8 = 1
        t7 = t8 * np.ones((784, m))
        t6 = L_sq_prime(X_mini.T,f[5]) * t7
        #print "t6 = ", t6
        t5 = sigma_prime(f[4]) * t6
        #print "t5 = ", t5
        t4 = t5
        t3 = np.sum(np.tile(np.expand_dims(W2,2),m) * np.expand_dims(t4,1), axis = 0)
        #print "t3 = ", t3
        
        t2 = phi_prime(f[1]) * t3
        t1 = t2
        t_W1 = np.matmul(t1,X_mini)
        t_W2 = np.matmul(t4,f[2].T)
        t_b1 = np.sum(t2, axis = 1, keepdims = True)
        t_b2 = np.sum(t5, axis = 1, keepdims = True)  
        
        #divide by number of minibatch examples
        g_W1 = t_W1/m
        #print "g_W1 = ", g_W1 #WHY IS THE GRADIENT THE SAME IN EVERY ROW??
        #print "max g_W1 = ", np.round(np.max(g_W1),20)
        g_W2 = t_W2/m
        g_b1 = t_b1/m
        g_b2 = t_b2/m
        
        #update moving averages
        #m_W1 = (beta1 * m_W1 + (1 - beta1) * g_W1) / (1 - beta1 ** k)
        #print "m_W1 = ", np.round(m_W1,20)
        #print "previous v_W1 = ", v_W1.T
        #v_W1 = (beta2 * v_W1 + (1 - beta2) * np.square(g_W1)) / (1 - beta2 ** k)
        #print "new g_W1 = ", g_W1.T
        #print "next v_W1 = ", v_W1.T
        # m_W2 = (beta1 * m_W2 + (1 - beta1) * g_W2) / (1 - beta1 ** k)
        # v_W2 = (beta2 * v_W2 + (1 - beta2) * np.square(g_W2)) / (1 - beta2 ** k)
        # m_b1 = (beta1 * m_b1 + (1 - beta1) * g_b1) / (1 - beta1 ** k)
        # v_b1 = (beta2 * v_b1 + (1 - beta2) * np.square(g_b1)) / (1 - beta2 ** k)
        # m_b2 = (beta1 * m_b2 + (1 - beta1) * g_b2) / (1 - beta1 ** k)
        # v_b2 = (beta2 * v_b2 + (1 - beta2) * np.square(g_b2)  ) / (1 - beta2 ** k)  
        
        #compute learning rate
        if k < tau:
            alpha = k/tau 
            eps = (1 - alpha) * eps0 + alpha * eps_tau
        else:
            eps = eps_tau
        if k % 100 == 1:
            print "eps = ", eps
        
        # #update parameters (note mistake in slides)
        # delta_W1 = eps/(np.sqrt(v_W1) + gamma) * m_W1
        # print "v_W1 = ", np.round(v_W1,2)
        # print "m_W1 = ", np.round(m_W1,2)
        # print "delta_W1 =", np.round(delta_W1,20)
        # W1 -= delta_W1
        # delta_W2 = eps/(np.sqrt(v_W2) + gamma) * m_W2
        # #print "W1 = ", np.round(W1,2)
        # W2 -= delta_W2
        # b1 -= eps/(np.sqrt(v_b1) + gamma) * m_b1
        # b2 -= eps/(np.sqrt(v_b2) + gamma) * m_b2
        
        W1 -= eps * g_W1
        #print "W1 = ", np.round(W1,2)
        W2 -= eps * g_W2
        b1 -= eps * g_b1    
        b2 -= eps * g_b2        
                
    return W1, W2, b1, b2

W1, W2, b1, b2 = adam(X)
print "W1 = ", W1
print "W2 = ", W2
print "b1 = ", b1.T
print "b2 = ", b2.T

for n in range(10):                
    y = np.atleast_2d(images_array[n]).T
    plt.figure()
    plt.imshow(np.reshape(y, (28,28)),interpolation="None",cmap='gray')
    plt.axis('off')
    plt.title("original")
    reconstr = E(np.reshape(y,(784,1)),W1,W2,b1,b2, pred = True)
    plt.figure()
    plt.imshow(np.reshape(reconstr, (28,28)),interpolation="None",cmap='gray')
    plt.axis('off')
    plt.title("reconstruction")