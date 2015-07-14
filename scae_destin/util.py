"""Utility functions
"""

import numpy as np;

import theano;

def init_weights(name,
                 out_dim,
                 in_dim=None,
                 weight_type="none"):
    """Create shared weights or bias
    
    Parameters
    ----------
    out_dim : int
        output dimension
    in_dim : int
        input dimension
    weight_type : string
        type of weights: "none", "tanh", "sigmoid"
    
    Returns
    -------
    Weights : matrix or vector
        shared matrix with respect size
    """
  
    if in_dim is not None:
        if weight_type=="tanh":
            lower_bound=-np.sqrt(6. / (in_dim + out_dim));
            upper_bound=np.sqrt(6. / (in_dim + out_dim));
        elif weight_type=="sigmoid":
            lower_bound=-4*np.sqrt(6. / (in_dim + out_dim));
            upper_bound=4*np.sqrt(6. / (in_dim + out_dim));
        elif weight_type=="none":
            lower_bound=0;
            upper_bound=1./(in_dim+out_dim);
  
    if in_dim==None:
        return theano.shared(value=np.asarray(np.random.uniform(low=0,
                                                                high=1./out_dim,
                                                                size=(out_dim, )),
                                              dtype="float32"),
                             name=name,
                             borrow=True);
    else:
        return theano.shared(value=np.asarray(np.random.uniform(low=lower_bound,
                                                                high=upper_bound,
                                                                size=(in_dim, out_dim)),
                                              dtype="float32"),
                             name=name,
                             borrow=True);