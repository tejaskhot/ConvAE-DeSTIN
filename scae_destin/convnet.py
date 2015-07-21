"""ConvNet Layer

ConvNet base layer and extended layer

"""

import numpy as np;

import theano;
import theano.tensor as T;
from theano.tensor.signal import downsample;
from theano.tensor.nnet import conv;

import scae_destin.util as util;
import scae_destin.nnfuns as nnfuns;

class ConvNetBase(object):
    """ConvNet base layer"""
    
    def __init__(self,
                 filter_size,
                 num_filters,
                 num_channels,
                 fm_size=None,
                 batch_size=None,
                 step=(1,1),
                 border_mode="valid",
                 use_bias=True,
                 **kwargs):
        """Initialize ConvNet structure
        
        Parameters
        ----------
        filter_size : tuple
            height and width of the filter (height, width)
        num_filters : int
            number of filters
        num_channels : int
            number of channels
        fm_size : tuple
            feature map size (height, width)
        batch_size : int 
            number of example in one batch
        step : tuple
            The step (or stride) with which to slide the filters over the
            image. Defaults to (1, 1).
        border_mode : string
            valid, same or full convolution : "valid", "same", "full"
        use_bias : bool
            either if use bias
        """
        
        super(ConvNetBase, self).__init__(**kwargs);
         
        self.filter_size=filter_size;
        self.num_filters=num_filters;
        self.num_channels=num_channels;
        self.fm_size=fm_size;
        self.batch_size=batch_size;
        self.step=step;
        self.border_mode=border_mode;
        self.use_bias=use_bias;
        
        self.initialize();
        
    def initialize(self, weight_type="none"):
        """Initialize weights and bias
        
        Parameters
        ----------
        weight_type : string
            type of weights: "none", "tanh", "sigmoid"
        """
        
        # should have better implementation for convnet weights
        
        fan_in = self.num_channels*np.prod(self.filter_size);
        fan_out = self.num_filters*np.prod(self.filter_size);
        
        filter_bound=np.sqrt(6./(fan_in + fan_out));
        filter_shape=(self.num_filters, self.num_channels)+(self.filter_size);
        self.filters = theano.shared(np.asarray(np.random.uniform(low=-filter_bound,
                                                                  high=filter_bound,
                                                                  size=filter_shape),
                                                                  dtype='float32'),
                                                                  borrow=True);
        
        if self.use_bias==True:
            self.bias=util.init_weights("bias", self.num_filters, weight_type=weight_type);
        
    def apply_lin(self, X):
        """Apply convoution operation
        
        Parameters
        ----------
        X : 4D tensor
            data with shape (batch_size, num_channels, height, width)
            
        Returns
        -------
        
        """
        
        if self.border_mode in ['valid', 'full']:
            Y=conv.conv2d(input=X,
                          filters=self.filters,
                          image_shape=(self.batch_size, self.num_channels)+(self.fm_size),
                          filter_shape=(self.num_filters, self.num_channels)+(self.filter_size),
                          border_mode=self.border_mode,
                          subsample=self.step);
        elif self.border_mode=="same":
            Y = conv.conv2d(input=X,
                            filters=self.filters,
                            image_shape=(self.batch_size, self.num_channels)+(self.fm_size),
                            filter_shape=(self.num_filters, self.num_channels)+(self.filter_size),
                            border_mode="full",
                            subsample=self.step);
            shift_x = (self.filter_size[0] - 1) // 2
            shift_y = (self.filter_size[1] - 1) // 2
            
            Y = Y[:, :, shift_x:self.fm_size[0] + shift_x, shift_y:self.fm_size[1] + shift_y];
                      
        if self.use_bias:
            Y+=self.bias.dimshuffle('x', 0, 'x', 'x');
        
        return Y;
    
    def get_dim(self, name):
        """Get dimensions for feature map and filter
        
         Parameters
        ----------
        name : string
            "input" or "output"
        
        Returns
        -------
        dimension : tuple
            input or output dimension
        """
        
        if name=="input":
            return (self.num_channels,)+self.fm_size;
        elif name=="output":
            if self.border_mode=="same":
                return (self.num_channels,)+self.fm_size;
            else:
                return ((self.num_filters,)+
                        conv.ConvOp.getOutputShape(self.fm_size, self.filter_size,
                                                   self.step, self.border_mode));
    
    @property    
    def params(self):
        return (self.filters, self.bias);
    
    @params.setter
    def params(self, param_list):
        self.filters.set_value(param_list[0].get_value());
        self.bias.set_value(param_list[1].get_value());
        
####################################
# ConvNet Layer
####################################

class IdentityConvLayer(ConvNetBase):
    """Identity ConvNet Layer"""
    
    def __init__(self, *args, **kwargs):
        super(IdentityConvLayer, self).__init__(**kwargs);
        
    def apply(self, X):
        return self.apply_lin(X);
    
class TanhConvLayer(ConvNetBase):
    """Tanh ConvNet Layer"""
    
    def __init__(self, *args, **kwargs):
        super(TanhConvLayer, self).__init__(**kwargs);
        
    def apply(self, X):
        return nnfuns.tanh(self.apply_lin(X));
    
class SigmoidConvLayer(ConvNetBase):
    """Sigmoid ConvNet Layer"""
    
    def __init__(self, *args, **kwargs):
        super(SigmoidConvLayer, self).__init__(**kwargs);
        
    def apply(self, X):
        return nnfuns.sigmoid(self.apply_lin(X));
    
class ReLUConvLayer(ConvNetBase):
    """ReLU ConvNet Layer"""
    
    def __init__(self, *args, **kwargs):
        super(ReLUConvLayer, self).__init__(**kwargs);
        
    def apply(self, X):
        return nnfuns.relu(self.apply_lin(X));

class LCNLayer(ReLUConvLayer):
    """returns LCN processed output"""
    def __init__(self, *args, **kwargs):
        super(ReLUConvLayer, self).__init__(**kwargs)

    def apply(self, X):
        X_conv = nnfuns.relu(self.apply_lin(X))             #full convolution

        #for each pixel remove mean of (filter_size[0]xfilter_size[1]) neighbourhood
        mid = int(np.floor(self.filter_size[0]/2.))         #middle value
        X_centered = X - X_conv[:,:,mid:-mid, mid:-mid]     #same shape as X

        X_sq = nnfuns.relu(self.apply_lin(X_centered ** 2))

        denom = T.sqrt(X_sq[:,:,mid:-mid, mid:-mid])
        per_img_mean = denom.mean(axis = [2,3])
        divisor = T.largest(per_img_mean.dimshuffle(0,1, 'x', 'x'), denom)
        new_X = X_centered / T.maximum(1., divisor)         #same format as input
        return new_X
    
####################################
# Pooling Layer
####################################

class MaxPooling(object):
    """ Max Pooling Layer """
    
    def __init__(self,
                 pool_size,
                 step=None,
                 input_dim=None,
                 mode="max",
                 **kwargs):
        """Initialize max pooling
        
        Parameters
        ----------
        pool_size : tuple
            height and width of pooling region
        step : tuple
            The vertical and horizontal shift (stride)
        input_dim : tuple
            Dimension of input feature maps
        mode : string
            Pooling method: "max", "sum", "average_inc_pad", "average_exc_pad"
            Max-pooling, Sum-pooling or Average-pooling
        """
        self.pool_size=pool_size;
        self.step=step;
        self.input_dim=input_dim;
        self.mode=mode;
        
    def apply(self, X):
        """apply max-pooling
        
        Parameters
        ----------
        X : 4D tensor
            data with shape (batch_size, num_channels, height, width)
            
        Returns
        -------
        pooled : 4D tensor
            pooled out features
        """
        
        ## Check if have bleeding edge support
        if theano.__version__=="0.7.0":
            if self.mode=="max":
                return downsample.max_pool_2d(X, self.pool_size, st=self.step);
            else:
                raise ValueError("Value %s is not a valid choice of pooling method for %s"
                                 % (self.mode, theano.__version__));
        else:
            return downsample.max_pool_2d(X, self.pool_size, st=self.step, mode=self.mode);
        
    def get_dim(self, name):
        """Get dimensions for feature map and filter
        
        (need to consider average mode?)
        
         Parameters
        ----------
        name : string
            "input" or "output"
        
        Returns
        -------
        dimension : tuple
            input or output dimension
        """
        
        if name=="input":
            return self.input_dim;
        elif name=="output":
            return tuple(downsample.DownsampleFactorMax.out_shape(self.input_dim,
                                                                  self.pool_size,
                                                                  st=self.step));
        
class MaxPoolingSameSize(object):
    """
    Same size Max-pooling layer

    Takes as input a 4-D tensor. It sets all non maximum values
    of non-overlapping patches of size (patch_size[0],patch_size[1]) to zero,
    keeping only the maximum values. The output has the same dimensions as
    the input.
    
    :type input: 4-D theano tensor of input images.
    :param input: input images. Max pooling will be done over the 2 last
        dimensions.
    :type patch_size: tuple of length 2
    :param patch_size: size of the patch (patch height, patch width).
        (2,2) will retain only one non-zero value per patch of 4 values.
    """
    
    def __init__(self, pool_size):
        """Init a max-pooling same size layer
        
        Parameters
        ----------
        pool_size : tuple
            size of the pool patch (patch height, patch width)
        """
    
        self.pool_size=pool_size;
        
    def apply(self, X):
        """Apply same size max-pooling operation
        
        Parameters
        ----------
        X : 4D tensor
            Max pooling will be done over the 2 last dimensions.
            
        Returns
        -------
        pooled : 4D tensor
            Pooled feature maps
        """
        
        if theano.__version__=="0.7.0":
            raise ValueError("Same size pooling is not supported in %s"
                                 % (theano.__version__));
        
        return downsample.max_pool_2d_same_size(X, self.pool_size);
    
class ArgMaxPooling(object):
    """Perform Argmax operation to a 4D tensor"""
    
    def __init__(self, relex_level=1.):
        """Init a argmax operation
        
        Parameters
        ----------
        relex_level : float
            relex level for argmax operation
        """
        self.relex_level=relex_level;
    
    def apply(self, X):
        """Apply argmax on 4D tensor
        
        Parameters
        ----------
        X : 4D tensor
            Max pooling will be done over the 2 last dimensions.
            
        Returns
        -------
        pooled : 4D tensor
            Pooled feature maps
        """
        
        return T.cast(T.ge(X,
                           self.relex_level*T.max(X,
                                                  axis=(1),
                                                  keepdims=True)),
                      dtype="float32");
        
class Flattener(object):
    """Flatten feature maps"""
    
    def apply(self, X):
        """flatten feature map
        
        Parameters
        ----------
        X : 4D tensor
            data with shape (batch_size, num_channels, height, width)
            
        Returns
        -------
        flatten_result : 2D matrix
        """
        
        return X.flatten(ndim=2);