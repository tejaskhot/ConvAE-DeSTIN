"""Basic Layer of Neural Network

This module describes a base layer of neural network (not including ConvNet)

It can be base of:

+ Identity layer
+ Tanh layer
+ Sigmoid layer
+ ReLU layer
+ Softmax layer
"""

import theano.tensor as T;

import scae_destin.util as util;

class Layer(object):
    """Abstract layer for Feed-forward Neural Networks"""
    
    def __init__(self,
                 in_dim,
                 out_dim,
                 layer_name="Layer",
                 W=None,
                 bias=None,
                 use_bias=True,
                 is_recursive=False,
                 **kwargs):
        """Base Layer initalization
        
        Parameters
        ----------
        in_dim : int
            input dimension of the layer
        out_dim : int
            output dimension of the layer
        W : matrix
            weight matrix for the layer, the size should be (in_dim, out_dim),
            if it is None, then the class will create one
        bias : vector
            bias vector for the layer, the size should be (out_dim),
            if it is None, then the class will create one
        """
        
        self.in_dim=in_dim;
        self.out_dim=out_dim;
        self.W=W;
        self.bias=bias;
        self.use_bias=use_bias;
        self.is_recursive=is_recursive;
        
        self.initialize();
        
        super(Layer, self).__init__(**kwargs);
        
    def initialize(self, weight_type="none"):
        """Initialize weights and bias
        
        Parameters
        ----------
        weight_type : string
            type of weights: "none", "tanh", "sigmoid"
        """
        
        if self.W==None:
            self.W=util.init_weights("W", self.out_dim, self.in_dim, weight_type=weight_type);
            
        if self.use_bias==True and self.bias==None:
            self.bias=util.init_weights("bias", self.out_dim, weight_type=weight_type);
    
    def apply_lin(self, X):
        """Apply linear transformation
        
        Parameters
        ----------
        X : matrix
            input samples, the size is (number of cases, in_dim)
            
        Returns
        -------
        Y : matrix
            output results, the size is (number of cases, out_dim);
        """
        
        Y=T.dot(X, self.W);
        
        if self.use_bias==True:
            Y+=self.bias;
        
        return Y;
    
    def get_dim(self, name):
        """Get dimension
        
        Parameters
        ----------
        name : string
            "input" or "output"
        
        Returns
        -------
        dimension : int
            input or output dimension
        """
        
        if name=="input":
            return self.in_dim;
        elif name=="output":
            return self.out_dim;
    
    @property    
    def params(self):
        return (self.W, self.bias);
    
    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value());
        self.bias.set_value(param_list[1].get_value());