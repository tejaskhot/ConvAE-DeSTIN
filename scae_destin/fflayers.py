"""Feed-forward Layers (not includeing ConvNet Layer)

This module contains feedforward layers for 

+ Identity layer
+ Tanh layer
+ Sigmoid layer
+ ReLU layer
+ Softmax layer
"""

import theano.tensor as T;

import scae_destin.nnfuns as nnfuns;
from scae_destin.layer import Layer;

class IdentityLayer(Layer):
    """Identity Layer
    """
    
    def __init__(self, **kwargs):
        super(IdentityLayer, self).__init__(**kwargs);
    
    def apply(self, X):
        return self.apply_lin(X);
        
class TanhLayer(Layer):
    """Tanh Layer
    """
    
    def __init__(self, **kwargs):
        super(TanhLayer, self).__init__(**kwargs);
        
        self.initialize("tanh");
        
    def apply(self, X):
        return nnfuns.tanh(self.apply_lin(X));
    
class SigmoidLayer(Layer):
    """Sigmoid Layer"""
    
    def __init__(self, **kwargs):
        
        super(SigmoidLayer, self).__init__(**kwargs);
        
        self.initialize("sigmoid");
        
    def apply(self, X):
        return nnfuns.sigmoid(self.apply_lin(X));

class ReLULayer(Layer):
    """ReLU Layer"""
    
    def __init__(self, **kwargs):
        super(ReLULayer, self).__init__(**kwargs);
        
    def apply(self, X):
        return nnfuns.relu(self.apply_lin(X));
    
class SoftmaxLayer(Layer):
    """Softmax Layer"""
    def __init__(self, **kwargs):
        super(SoftmaxLayer, self).__init__(**kwargs);
        
    def apply(self, X):
        return nnfuns.softmax(self.apply_lin(X));
    
    def predict(self, X_out):
        """Predict label
        
        Parameters
        ----------
        X_out : matrix
            input sample outputs, the size is (number of cases, number of classes)
            
        Returns
        -------
        Y_pred : vector
            predicted label, the size is (number of cases)
        """
        
        return T.argmax(X_out, axis=1);
    
    def error(self, X_out, Y):
        """Mis-classified label
        
        Parameters
        ----------
        X_out : vector
            predict labels, the size is (number of cases, number of classes)
        Y : vector
            correct labels, the size is (number of cases)
            
        Returns
        -------
        error : scalar
            difference between predicted label and true label.
        """
    
        return T.mean(T.neq(self.predict(X_out), Y));