"""


@note: CKM tests
"""
import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import time
import theano
import theano.tensor as T

# from scae_destin.ckm import CKMLayer
from scae_destin.convnet import IdentityConvLayer
from scae_destin.model import FeedForward
import scae_destin.datasets as ds

# from scae_destin.ckm_layer import CKMLayer;
from scae_destin.ckm2 import CKMLayer
import scae_destin.datasets as ds

rng=np.random.RandomState(125);

## data already pre-processed
X = np.load('../data/image_patches.npy')
batch_size=100
training_portion=1 #entire dataset
n_epochs=100
num_filters=64
num_batches=X.shape[0]/batch_size
print "INPUT SHAPE : ",X.shape

## make it shared
train_set_x = theano.shared(np.asarray(X, dtype="float32"),
                borrow=True)
n_train_batches=int(train_set_x.get_value(borrow=True).shape[0]*training_portion)
n_train_batches /= batch_size; # number of train data batches

print "[MESSAGE] The data is loaded"
print "[MESSAGE] Building model"


index=T.lscalar(); # batch index

X=T.matrix('X');  # input data source
y=T.ivector('y'); # input data label

images=X.reshape((batch_size, 1, 16, 16))

layer=CKMLayer(rng=rng,
               feature_maps=images,
               feature_shape=(batch_size, 1, 16, 16),
               filter_shape=(num_filters, 1, 16, 16),
               pool=False,
               activate_mode="relu")

updates, cost, update_out=layer.ckm_updates();
print type(update_out)
train_model=theano.function(inputs=[index],
                            outputs=[cost, update_out],
                            updates=updates,
                            givens={X: train_set_x[index * batch_size: (index + 1) * batch_size]});
                            
print "... model is built"

epoch = 0;
while (epoch < n_epochs):
  epoch = epoch + 1;
  c = []
  for batch_index in xrange(n_train_batches):
    co, update_out=train_model(batch_index)
    c.append(co)
    
    #print update_out
    #print np.mean(update_out, axis=(1,2,3), keepdims=True)
    #print np.max(update_out, axis=(1,2,3), keepdims=True)
    #print np.array(update_out>=np.mean(update_out, axis=(2,3), keepdims=True), dtype=float);
    #plt.imshow(update_out[0,:,:], cmap = plt.get_cmap('gray'), interpolation='nearest')
    #plt.show()
    
#     for i in xrange(nkerns[0]):
#       plt.subplot(8, 7, i);
#       plt.imshow(update_out[i,0,:,:], cmap = plt.get_cmap('gray'), interpolation='nearest');
#       plt.axis('off')
#     plt.show();
    
  print 'Training epoch %d, cost ' % epoch, np.mean(c);
  
filters=layer.filters;

for i in xrange(num_filters):
  plt.subplot(8, 8, i);
  plt.imshow(filters.get_value(borrow=True)[i,0,:,:], cmap = plt.get_cmap('gray'), interpolation='nearest');
  plt.axis('off')

plt.show();