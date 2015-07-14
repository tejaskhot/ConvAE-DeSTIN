"""Cost

This module implements several general cost functions

+ Mean-square cost
+ Categorical Cross Entropy cost
+ Binary Cross Entropy cost
+ L1 regularization
+ L2 regularization
+ KL divergence
"""

import theano.tensor as T;

def mean_square_cost(Y_hat, Y_star):
    """Mean Square Reconstruction Cost
    
    Parameters
    ----------
    Y_hat : tensor
        predicted output of neural network
    Y_star : tensor
        optimal output of neural network
        
    Returns
    -------
    costs : scalar
        cost of mean square reconstruction cost
    """
    
    #cost=T.sum(T.pow(T.sub(Y_hat, Y_star),2), axis=1);
    
    #return 0.5*T.mean(cost);
    return T.mean((Y_hat - Y_star) ** 2);

def binary_cross_entropy_cost(Y_hat, Y_star):
    """Binary Cross Entropy Cost
    
    Parameters
    ----------
    Y_hat : tensor
        predicted output of neural network
    Y_star : tensor
        optimal output of neural network
        
    Returns
    -------
    costs : scalar
        cost of binary cross entropy cost
    """
    
    return T.sum(T.nnet.binary_crossentropy(Y_hat, Y_star), axis=1).mean();

def categorical_cross_entropy_cost(Y_hat, Y_star):
    """Categorical Cross Entropy Cost
    
    Parameters
    ----------
    Y_hat : tensor
        predicted output of neural network
    Y_star : tensor
        optimal output of neural network
        
    Returns
    -------
    costs : scalar
        cost of Categorical Cross Entropy Cost
    """
    
    return T.nnet.categorical_crossentropy(Y_hat, Y_star).mean();

def L1_regularization(params, L1_rate=0.):
    """L1 Regularization
    
    Parameters
    ----------
    params : tuple
        list of params
    L1_rate : double
        decay rate of L1 regularization
        
    Returns
    -------
    cost : scalar
        L1 regularization decay
    """
    
    cost=0;
    for param in params:
        cost+=T.sum(T.abs_(param));
        
    return L1_rate*cost;

def L2_regularization(params, L2_rate=0.):
    """L2 Regularization
    
    Parameters
    ----------
    params : tuple
        list of params
    L2_rate : double
        decay rate of L2 regularization
        
    Returns
    -------
    cost : scalar
        L2 regularization decay
    """
    
    cost=0;
    for param in params:
        cost+=(param**2).sum();
    
    return L2_rate*cost;

def kl_divergence(p, p_hat):
    """Compute KL divergence
    
    Parameter
    ---------
    p : float
        sparsity parameter
    p_hat : float
        average activation of a hidden neuron
    """
    
    return p_hat-p+p*T.log(p/p_hat);

