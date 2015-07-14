"""Models

This module documented some training models

+ Feedforward Neural Network (including ConvNets)
"""

from scae_destin.optimize import corrupt_input;
from scae_destin.optimize import apply_dropout;

class FeedForward(object):
    """Feedforward Neural Network model"""
    
    def __init__(self,
                 layers=None,
                 dropout=None):
        """Initialize feedforward model
        
        Parameters
        ----------
        in_dim : int
            number of input size
        layers : list
            list of layers
        drouput : list
            list of dropout mask
        """
        self.layers=layers;
        if dropout is None:
            self.dropout=[];
        else:
            self.dropout=dropout;
        
    def fprop(self,
              X):
        """Forward propagation
        
        Parameters
        ----------
        X : matrix or 4D tensor
            input samples, the size is (number of cases, in_dim)
            
        Returns
        -------
        out : list
            output list from each layer
        """
        
        out=[];
        level_out=X;
        for k, layer in enumerate(self.layers):
            if len(self.dropout)>0:
                level_out=apply_dropout(level_out, self.dropout[k]);
            
            level_out=layer.apply(level_out);
            
            out.append(level_out);
            
        return out;
    
    @property
    def params(self):
        return [param for layer in self.layers if hasattr(layer, 'params') for param in layer.params];
    
class AutoEncoder(object):
    """AutoEncoder model for MLP layers
    
    This model only checking the condition of auto-encoders,
    the training is done by FeedForward model
    """
    
    def __init__(self, layers=None):
        """Initialize AutoEncoder
        
        Parameters
        ----------
        layers : tuple
            list of MLP layers
        """
        
        self.layers=layers;
        self.check();
        
    def check(self):
        """Check the validity of an AutoEncoder
        """
        
        assert self.layers[0].get_dim("input")==self.layers[-1].get_dim("output"), \
            "Input dimension is not match to output dimension";
           
        for layer in self.layers:
            assert hasattr(layer, 'params'), \
                "Layer doesn't have necessary parameters";
                
    def fprop(self,
              X,
              corruption_level=None,
              noise_type="binomial",
              epoch=None,
              decay_rate=1.):
        """Forward pass of auto-encoder
        
        Parameters
        ----------
        X : matrix
            number of samples in (number of samples, dim of sample)
        corruption_level : float
            corruption_level on data
        noise_type : string
            type of noise: "binomial" or "gaussian"
        
        Returns
        -------
        out : matrix
            output list for each layer
        """
        
        out=[];
        
        if epoch is not None:
            self.corruption_level=corruption_level*(epoch**(-decay_rate));
        else:
            self.corruption_level=corruption_level;
        
        if self.corruption_level == None:
            level_out=X;
        else:
            level_out=corrupt_input(X, self.corruption_level, noise_type);
        for k, layer in enumerate(self.layers):
            
            level_out=layer.apply(level_out);
            
            out.append(level_out);
            
        return out;
    
    @property
    def params(self):
        return [param for layer in self.layers if hasattr(layer, 'params') for param in layer.params];
    
class ConvAutoEncoder(object):
    """Convolutional Auto-Encoder model"""
    def __init__(self, layers):
        """Initialize ConvAE
        
        Parameters
        ----------
        layers : tuple
            list of feedforward layers
        """
        
        self.layers=layers;
        self.check();
        
    def check(self):
        """Checking the validity of a ConvAutoEncoder"""
        pass
        
#         assert self.layers[0].get_dim("input")==self.layers[-1].get_dim("output"), \
#             "Input dimension is not match to output dimension";
        
    def fprop(self,
              X,
              corruption_level=None,
              noise_type="binomial",
              epoch=None,
              decay_rate=1.):
        """Forward pass of convolutional auto-encoder
        
        Parameters
        ----------
        X : 4D tensor
            data in (batch size, channel, height, width)
        corruption_level : float
            corruption_level on data
        noise_type : string
            type of noise: "binomial" or "gaussian"
        
        Returns
        -------
        out : 4-D tensor
            output list for each layer
        """
        
        out=[];
        
        if epoch is not None:
            self.corruption_level=corruption_level*(epoch**(-decay_rate));
        else:
            self.corruption_level=corruption_level;
        
        if self.corruption_level is None:
            level_out=X;
        else:
            level_out=corrupt_input(X, self.corruption_level, noise_type);
            
        for k, layer in enumerate(self.layers):
            
            level_out=layer.apply(level_out);
            
            out.append(level_out);
            
        return out;
    
    @property
    def params(self):
        return [param for layer in self.layers if hasattr(layer, 'params') for param in layer.params];
                
class ConvKMeans(object):
    """Convolutional K-means"""
    
    def __init__(self, layers):
        """Init a Conv K-means model
        
        Parameters
        ----------
        layers : tuple of size 2
            one conv layer, one arg-max pooling layer
        """
        
        assert len(layers)==2, \
            "Too many layers for Conv K-means";
        
        self.layers=layers;
        
    def get_layer(self):
        """Get trained convolution layer
        """
        return self.layers[0];
    
    def fprop(self, X):
        """Get activation map
        
        Parameters
        ----------
        X : matrix or 4D tensor
            input samples, the size is (number of cases, in_dim)
            
        Returns
        -------
        out : list
            output list from each layer
        """
        
        out=[];
        level_out=X;
        for k, layer in enumerate(self.layers):
            level_out=layer.apply(level_out);
            out.append(level_out);
            
        return out;
        