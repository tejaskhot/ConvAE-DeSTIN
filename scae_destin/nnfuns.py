"""Mask for neural network functions

Current documented NN functions:

+ Hyperbolic tangent nonlinearity
+ Standard sigmoid nonlinearity
+ Soft Plus nonlinearity
+ Rectified linear unit
+ Identity function
"""

import theano.tensor as T

def tanh(x):
    """Hyperbolic tangent nonlinearity
    
    Parameters
    ----------
    x : matrix
        input matrix
    
    Returns
    -------
    y : matrix
        output matrix
  
    """
    return T.tanh(x);

def sigmoid(x):
    """Standard sigmoid nonlinearity
    
    Parameters
    ----------
    x : matrix
        input matrix
    
    Returns
    -------
    y : matrix
        output matrix

    """
    return T.nnet.sigmoid(x);

def softplus(x):
    """Softplus nonlinearity
    
    Parameters
    ----------
    x : matrix
        input matrix
    
    Returns
    -------
    y : matrix
        output matrix
    """
    return T.nnet.softplus(x);

def relu(x):
    """Rectified linear unit
  
    Parameters
    ----------
    x : matrix
        input matrix
    
    Returns
    -------
    y : matrix
        output matrix
    """
    return x*(x>1e-13);

def softmax(x):
    """Softmax function
    
    Parameters
    ----------
    x : matrix
        input matrix
    
    Returns
    -------
    y : matrix
        output matrix
    """
    return T.nnet.softmax(x);

def identity(x):
    """Identity function
    
    Parameters
    ----------
    x : matrix
        input matrix
    
    Returns
    -------
    y : matrix
        output matrix
    """
    return x;
