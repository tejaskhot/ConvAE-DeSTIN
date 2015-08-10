"""
feature map input : (batch_size, num_channels, fm_size[0], fm_size[1])
feature map output size : (input_size, num_filters, fm_size[0], fm_size[1])
filter_size : (, fm_size[0], fm_size[1])

"""
import numpy as np;

import theano;
import theano.tensor as T;

from scae_destin.convnet import ConvNetBase

class CKMLayer(ConvNetBase):
  """
  A implementation of Convolutional K-means Layer
  """
  def __init__(self,
              feature_shape,
              filter_size,
              num_filters,
              num_channels,
              fm_size=None,
              batch_size=None,
              step=(1,1),
              border_mode="valid",
              use_bias=True,
              **kwargs):
    """
    Initialize a CKM Layer.
    """
    super(CKMLayer, self).__init__(filter_size=filter_size,
                                   num_filters=num_filters,
                                   num_channels=num_channels,
                                   fm_size=fm_size,
                                   batch_size=batch_size,
                                   step=step,
                                   border_mode=border_mode,
                                   use_bias=use_bias,
                                   **kwargs)
    self.feature_shape=feature_shape
                                                               
  def ckm_updates(self, X):
    """
    This function computes updates of filters and total changes 
    """
    

    feature_shape_temp=np.asarray(self.feature_shape);
    filter_shape_temp=np.asarray([self.num_filters, self.num_channels, self.filter_size[0], self.filter_size[1]]);
    ams_shape=(filter_shape_temp[1],
               feature_shape_temp[0],
               feature_shape_temp[2]-filter_shape_temp[2]+1,
               feature_shape_temp[3]-filter_shape_temp[3]+1);
    
    fmaps = self.apply_lin(X,
                           image_shape=self.feature_shape,
                           filter_shape=filter_shape_temp) # (num_examples, num_filters, fm_height, fm_width)
    fmaps = fmaps.dimshuffle(1,0,2,3) # (num_filters, num_examples , fm_height, fm_width)
    print "sum of filters is : ",T.sum(self.filters)

    activation_maps=T.cast(T.max(fmaps, axis=(0), keepdims=True), dtype="float32")
    # ams_sum=T.cast(T.sum(activation_maps, axis=(1,2,3), keepdims=True), dtype="float32");
    
    feature_shape_new=(feature_shape_temp[1], feature_shape_temp[0], feature_shape_temp[2], feature_shape_temp[3])
    
    update_out = self.apply_lin(X.dimshuffle(1,0,2,3),
                                filters=activation_maps,
                                image_shape=feature_shape_new,
                                filter_shape=ams_shape)
    
    update_out=update_out.dimshuffle(1,0,2,3);
    update_out+=self.filters
    # update_out/=(ams_sum+1);

    
    updates=[(self.filters, 0*self.filters+update_out)];
    
    return updates, T.sum(self.filters), update_out;

