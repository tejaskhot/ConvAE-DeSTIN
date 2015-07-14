"""Optimization method

The methods are implemented by referencing Lasagne

Supported methods:

+ Stochastic gradient descent (Momentum and Nestrov Momentum)
+ Adagrad
+ Adadelta
+ RMSprop
+ Adam
"""

from collections import OrderedDict;

import numpy as np;

import theano;
import theano.tensor as T;

def sgd(cost,
        params,
        updates=None,
        learning_rate=0.001):
    """Stochastic Gradient Descent (SGD)
    
    Parameters
    ----------
    cost : scalar
        total cost of the cost function.
    params : list
        parameter list
    learning_rate : float
        learning rate of SGD
        
    Returns
    -------
    updates : OrderedDict
        dictionary of updates
    """
    
    if updates is None:
        updates=OrderedDict();
    
    gparams=T.grad(cost, params);
    
    for param, gparam in zip(params, gparams):
        updates[param] = param - learning_rate * gparam;
        
    return updates;

def apply_momentum(updates,
                   params,
                   momentum=0.):
    """Apply momentum to update updates dictionary
    
    Parameters
    ----------
    updates : OrderedDict
        list of parameter updates
    params : list
        list of params to be updated
    momentum : float
        The amount of momentum to apply
    
    Returns
    -------
    updates : OrderedDict
        updated list of parameters
    """
    
    updates=OrderedDict(updates);
        
    for param in params:
        value = param.get_value(borrow=True);
        velocity=theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable);
        x=momentum*velocity+updates[param]
        updates[velocity]=x-param;
        updates[param]=x;
    
    return updates;

def apply_nestrov_momentum(updates,
                           params,
                           momentum=0.):
    """Apply Nestrov momentum to update updates dictionary
    
    Parameters
    ----------
    updates : OrderedDict
        list of parameter updates
    params : list
        list of params to be updated
    momentum : float
        The amount of momentum to apply
    
    Returns
    -------
    updates : OrderedDict
        updated list of parameters
    """
    
    updates=OrderedDict(updates);
        
    for param in params:
        value = param.get_value(borrow=True);
        velocity=theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable);
        x=momentum*velocity+updates[param]-param;
        updates[velocity]=x;
        updates[param]=momentum*x+updates[param];
    
    return updates;

def adagrad(cost,
            params,
            updates=None,
            learning_rate=0.001,
            eps=1e-6):
    """Adagrad
    
    Parameters
    ----------
    cost : scalar
        total cost of the cost function.
    params : list
        parameter list
    learning_rate : float
        learning rate of SGD
    eps : float
        Small value added for numerical stability
        
    Returns
    -------
    updates : OrderedDict
        list of updated parameters
    """
    
    if updates is None:
        updates=OrderedDict();
        
    gparams=T.grad(cost, params);
    
    for param, gparam in zip(params, gparams):
        value=param.get_value(borrow=True);
        accu=theano.shared(np.zeros(value.shape, dtype=value.dtype),
                           broadcastable=param.broadcastable);
                           
        accu_new=accu+gparam**2;
        updates[accu]=accu_new;
        updates[param]=param-(learning_rate*gparam/T.sqrt(accu_new+eps));
        
    return updates;

def adadelta(cost,
             params,
             updates=None,
             learning_rate=0.001,
             eps=1e-6,
             rho=0.95):
    """Adadelta
    
    Parameters
    ----------
    cost : scalar
        total cost of the cost function.
    params : list
        parameter list
    learning_rate : float
        learning rate of SGD
    eps : float
        Small value added for numerical stability
    rho : float
        Squared gradient moving average decay factor
        
    Returns
    -------
    updates : OrderedDict
        list of updated parameters
    """
    
    if updates is None:
        updates=OrderedDict();
        
    gparams=T.grad(cost, params);
    
    for param, gparam in zip(params, gparams):
        value=param.get_value(borrow=True);
        accu=theano.shared(np.zeros(value.shape, dtype=value.dtype),
                           broadcastable=param.broadcastable);
                           
        delta_accu=theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable);
                           
        accu_new=rho*accu+(1-rho)*gparam**2;
        updates[accu]=accu_new;
        
        update=(gparam*T.sqrt(delta_accu+eps)/T.sqrt(accu_new+eps));
        updates[param]=param-learning_rate*update;
        
        delta_accu_new=rho*delta_accu+(1-rho)*update**2;
        updates[delta_accu]=delta_accu_new;
        
    return updates;

def rmsprop(cost,
            params,
            updates=None,
            learning_rate=0.001,
            eps=1e-6,
            rho=0.9):
    """RMSprop
    
    Parameters
    ----------
    cost : scalar
        total cost of the cost function.
    params : list
        parameter list
    learning_rate : float
        learning rate of SGD
    eps : float
        Small value added for numerical stability
    rho : float
        Gradient moving average decay factor
        
    Returns
    -------
    updates : OrderedDict
        list of updated parameters
    """
    
    if updates is None:
        updates=OrderedDict();
        
    gparams=T.grad(cost, params);
    
    for param, gparam in zip(params, gparams):
        value=param.get_value(borrow=True);
        accu=theano.shared(np.zeros(value.shape, dtype=value.dtype),
                           broadcastable=param.broadcastable);
                           
        accu_new=rho*accu+(1-rho)*T.sqr(gparam);
        updates[accu]=accu_new;
        updates[param]=param-(learning_rate*gparam/T.sqrt(accu_new+eps));
        
    return updates;

def adam(cost,
         params,
         updates=None,
         learning_rate=0.001,
         beta_1=0.9,
         beta_2=0.999,
         eps=1e-8):
    """Adam
    
    Parameters
    ----------
    cost : scalar
        total cost of the cost function.
    params : list
        parameter list
    learning_rate : float
        learning rate of SGD
    beta_1 : float
        Exponential decay rate for the first moment estimates.
    beta_2 : float
        Exponential decay rate for the second moment estimates.
    eps : float
        Small value added for numerical stability
        
    Returns
    -------
    updates : OrderedDict
        list of updated parameters
    """
    
    if updates==None:
        updates=OrderedDict();
        
    gparams=T.grad(cost, params);
    
    t_prev=theano.shared(np.asarray(0, dtype="float32"));
    for param, gparam in zip(params, gparams):
        m_prev = theano.shared(param.get_value() * 0.);
        v_prev = theano.shared(param.get_value() * 0.);
        
        t=t_prev+1;
        
        m_t = beta_1*m_prev + (1-beta_1)*gparam;
        v_t = beta_2*v_prev + (1-beta_2)*gparam**2;
        
        a_t = learning_rate*T.sqrt(1-beta_2**t)/(1-beta_1**t);
        step = a_t*m_t/(T.sqrt(v_t) + eps);
        
        updates[m_prev] = m_t;
        updates[v_prev] = v_t;
        updates[param] = param - step;
        
    updates[t_prev] = t
    return updates;

def gd_updates(cost,
               params,
               updates=None,
               momentum=None,
               nesterov=False,
               learning_rate=0.001,
               eps=1e-6,
               rho=0.95,
               beta_1=0.9,
               beta_2=0.999,
               method="sgd"):
    """Gradient Descent based optimization
    
    Note: should be a class to make flexible call
    
    Parameters
    ----------
    cost : scalar
        total cost of the cost function.
    params : list
        parameter list
    method : string
        optimization method: "sgd", "adagrad", "adadelta", "rmsprop"
        
    Returns
    -------
    updates : OrderedDict
        dictionary of updates
    """
    
    if method=="adagrad":
        updates=adagrad(cost, params, updates, learning_rate=learning_rate, eps=eps);
    elif method=="adadelta":
        updates=adadelta(cost, params, updates, learning_rate=learning_rate, eps=eps, rho=rho);
    elif method=="sgd":
        updates=sgd(cost, params, updates, learning_rate=learning_rate);
        
        if momentum is not None:
            if nesterov==True:
                updates=apply_nestrov_momentum(updates, params, momentum=momentum);
            else:
                updates=apply_momentum(updates, params, momentum=momentum);
    elif method=="rmsprop":
        updates=rmsprop(cost, params, updates, learning_rate=learning_rate, eps=eps, rho=rho);
    elif method=="adam":
        updates=adam(cost, params, updates, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, eps=eps);
    else:
        raise ValueError("Method %s is not a valid choice" % method);
            
    return updates;

theano_rng=T.shared_randomstreams.RandomStreams(np.random.randint(2 ** 30));

def dropout(shape, prob=0.):
    """generate dropout mask
    
    Parameters
    ----------
    shape : tuple
        shape of the dropout mask
    prob : double
        probability of each sample
        
    Returns
    -------
    mask : tensor
        dropout mask
    """
    
    if shape is not None:
        if prob==1.:
            mask=theano_rng.binomial(n=1, p=1-prob, size=shape);
        else:
            mask=theano_rng.binomial(n=1, p=1-prob, size=shape)/(1. - prob);
        return T.cast(x=mask, dtype="float32");
    else:
        return None;

def multi_dropout(shapes, prob=0.):
    """generate a list of dropout mask
    
    Parameters
    ----------
    shapes : tuple of tuples
        list of shapes of dropout masks
    prob : float
        probability of each sample
    
    Returns
    -------
    masks : tuple of tensors
        list of dropout masks
    """
    return [dropout(shape, prob) for shape in shapes];

def apply_dropout(X, mask=None):
    """apply dropout operation
    
    Parameters
    ----------
    X : tensor
        data to be masked
    mask : dropout mask
    
    Returns
    -------
    masked_X : tensor
        dropout masked data
    """
    
    if mask is not None:
        return X*mask;
    else:
        return X;
    
def corrupt_input(X, corruption_level=0., noise_type="binomial"):
    """Add noise on data
    
    Parameters
    ----------
    X : tensor
        data to be corrupted
    corruption_level : double
        probability of the corruption level, std if noise type is gaussian. 
    noise_type : string
        type of noise: "binomial" or "gaussian"
    Returns
    -------
    corrupted_out : tensor
        corrupted output 
    """
    # a=theano.tensor('a')
    # b=theano.fscalar('b')
    # c=a/b
    # f=theano.function([a,b],c)
    if noise_type=="binomial":
        corrupted_out=theano_rng.binomial(size=X.shape, n=1,
                                          p=1 - corruption_level,
                                          dtype="float32")*X;
        # corrupted_out=f(corrupted_out, 1 - corruption_level)
        corrupted_out=corrupted_out/(1 - corruption_level)
    elif noise_type=="gaussian":
        corrupted_out=theano_rng.normal(size=X.shape, std=corruption_level, dtype="float32")+X;
    
    return corrupted_out;