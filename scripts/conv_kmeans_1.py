"""
Convolutional K-means with patches from CIFAR-10

ALGORITHM:

- Load data in normal format
- perform convolution and get n feature maps
- for each location (i,j), iterate through all feature maps and 
  find which feature map has the highest value
  keep that feature map's value at location (i,j) as it is and
  make the value at (i,j) = 0 for all other feature maps 
- now perform convolutions on the image with these newly obtained
  filters of the same size
- add the original filters to these newly obtained filters and normalize
  the results by 
- save these filters to file 
"""

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import time
import theano
import theano.tensor as T

from scae_destin.ckm import CKMLayer
from scae_destin.convnet import IdentityConvLayer
from scae_destin.model import FeedForward
import scae_destin.datasets as ds


## data already pre-processed
X = np.load('../data/image_patches.npy')
batch_size=1
training_portion=1 #entire dataset
n_epochs=1
num_filters=64
num_batches=X.shape[0]/batch_size
print "INPUT SHAPE : ",X.shape

## make it shared
train_set_x = theano.shared(np.asarray(X, dtype="float32"),
			  			 	borrow=True)
n_train_batches=int(train_set_x.get_value(borrow=True).shape[0]*training_portion)
n_train_batches /= batch_size; # number of train data batches

X=T.matrix("data")
index=T.lscalar()

images=X.reshape((batch_size, 1, 16, 16))

## Convolution Layer
conv_layer = CKMLayer( filter_size=(5,5),
             feature_shape=(batch_size, 1, 16, 16),
					   num_filters=num_filters,
					   num_channels=1,
					   fm_size=(16,16),
					   batch_size=batch_size,
					   border_mode="valid")

# model = FeedForward(layers=[conv_layer])
# out = model.fprop(images)

updates, cost, update_out=conv_layer.ckm_updates(images)

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
    # print type(co), co.shape, co
  print 'Training epoch %d, cost ' % epoch, np.mean(c);

filters=conv_layer.filters
pickle.dump(filters, open("conv_kmeans_1.pkl","w"))
for i in xrange(num_filters):
	# image_adr="/home/tejas/Documents/ConvAE-DeSTIN/plots/ckm_1/filter_%d.eps" % (i)
	# plt.imshow(filters[i, 0, :, :], cmap = plt.get_cmap('gray'), interpolation='nearest')
	# plt.axis('off')
	# plt.savefig(image_adr , bbox_inches='tight', pad_inches=0)
	# if i%10 == 0:
	#     print 'completed saving filters till : ', i
  plt.subplot(8, 8, i);
  plt.imshow(filters.get_value(borrow=True)[i,0,:,:], cmap = plt.get_cmap('gray'), interpolation='nearest');
  plt.axis('off')

plt.show();














# f1 = theano.function(inputs=[idx], 
# 					 outputs=out,
# 					 givens={X: shared_X[idx* batch_size: (idx+1)*batch_size]})

# fmaps=np.asarray([])
# for idx in xrange(0, 2):
# 	temp=np.asarray(f1(idx)[0])
# 	if idx==0:
# 		fmaps=temp
# 	else:
# 		fmaps=np.vstack((fmaps, temp))

# print "Feature Maps size is: ", fmaps.shape

# parameters = conv_layer.params 
# filters=parameters[0].get_value()
# old_filters=filters

# # print filters.shape
# for i in xrange(0, filters.shape[2]):
# 	for j in xrange(0, filters.shape[3]):
# 		max_intensity=-1
# 		index=-1
# 		for f in xrange(0, len(filters)):
# 			if filters[f][0][i][j] > max_intensity:
# 				max_intensity=filters[f][0][i][j]
# 				index=f 
# 		for f in xrange(0, len(filters)):
# 			if f!=index:
# 				filters[f][0][i][j]=old_filters[f][0][i][j]
# 			else:
# 				filters[f][0][i][j]+=old_filters[f][0][i][j]

# ## normalization
# # for f in filters:
# # 	f[0] = (f[0]-np.min(f[0]))/(np.max(f[0])-np.min(f[0]))
# print np.argmax(filters,axis=0)
# print np.argmax(filters,axis=0).shape
# # print filters[1]
# filters_shared=theano.shared(filters,  borrow=True)
# ## use the new filters for convolution
# param_list = [filters_shared, parameters[1]]
# conv_layer.params(param_list)


print "DONE!"