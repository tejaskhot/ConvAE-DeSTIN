"""CIFAR-10 ConvNet Test
"""

import sys;
sys.path.append("..");

import numpy as np;
import matplotlib.pyplot as plt;
import cPickle as pickle;

import theano;
import theano.tensor as T;

import scae_destin.datasets as ds;
from scae_destin.fflayers import ReLULayer;
from scae_destin.fflayers import SoftmaxLayer;
from scae_destin.convnet import ReLUConvLayer;
from scae_destin.convnet import LCNLayer
from scae_destin.convnet import MaxPooling;
from scae_destin.convnet import Flattener;
from scae_destin.model import FeedForward;
from scae_destin.optimize import gd_updates;
from scae_destin.cost import categorical_cross_entropy_cost;
from scae_destin.cost import L2_regularization;

n_epochs=1;
batch_size=100;

Xtr, Ytr, Xte, Yte=ds.load_CIFAR10("/home/tejas/Desktop/cifar-10-batches-py");

Xtr=np.mean(Xtr, 3);
Xte=np.mean(Xte, 3);
Xtrain=Xtr.reshape(Xtr.shape[0], Xtr.shape[1]*Xtr.shape[2])/255.0;
Xtest=Xte.reshape(Xte.shape[0], Xte.shape[1]*Xte.shape[2])/255.0;
print Xtrain.shape
print Xtrain[0].shape
train_set_x, train_set_y=ds.shared_dataset((Xtrain, Ytr));
test_set_x, test_set_y=ds.shared_dataset((Xtest, Yte));

n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size;
n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size;

print "[MESSAGE] The data is loaded"

X=T.matrix("data");
y=T.ivector("label");
idx=T.lscalar();

images=X.reshape((batch_size, 1, 32, 32))

layer_0=LCNLayer(filter_size=(7,7),
                      num_filters=64,
                      num_channels=1,
                      fm_size=(32,32),
                      batch_size=batch_size,
                      border_mode="full");
                      
pool_0=MaxPooling(pool_size=(2,2));
                      
layer_1=LCNLayer(filter_size=(5,5),
                      num_filters=32,
                      num_channels=64,
                      fm_size=(16,16),
                      batch_size=batch_size,
                      border_mode="full");

pool_1=MaxPooling(pool_size=(2,2));

flattener=Flattener();

layer_2=ReLULayer(in_dim=32*64,
                  out_dim=800);
                  
layer_3=SoftmaxLayer(in_dim=800,
                     out_dim=10);
                     
model=FeedForward(layers=[layer_0, pool_0, layer_1, pool_1, flattener, layer_2, layer_3]);

out=model.fprop(images);
cost=categorical_cross_entropy_cost(out[-1], y);
updates=gd_updates(cost=cost, params=model.params, method="sgd", learning_rate=0.01, momentum=0.9);

extract=theano.function(inputs=[idx],
                        outputs=layer_0.apply(images),
                        givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]});
print extract(1).shape


train=theano.function(inputs=[idx],
                      outputs=cost,
                      updates=updates,
                      givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size],
                              y: train_set_y[idx * batch_size: (idx + 1) * batch_size]});

test=theano.function(inputs=[idx],
                     outputs=model.layers[-1].error(out[-1], y),
                     givens={X: test_set_x[idx * batch_size: (idx + 1) * batch_size],
                             y: test_set_y[idx * batch_size: (idx + 1) * batch_size]});
                              
print "[MESSAGE] The model is built"

test_record=np.zeros((n_epochs, 1));
epoch = 0;
while (epoch < n_epochs):
    epoch+=1;
    for minibatch_index in xrange(n_train_batches):
        mlp_minibatch_avg_cost = train(minibatch_index);
        
        iteration = (epoch - 1) * n_train_batches + minibatch_index;
        
        if (iteration + 1) % n_train_batches == 0:
            print 'MLP MODEL';
            test_losses = [test(i) for i in xrange(n_test_batches)];
            test_record[epoch-1] = np.mean(test_losses);
            
            print(('     epoch %i, minibatch %i/%i, test error %f %%') %
                  (epoch, minibatch_index + 1, n_train_batches, test_record[epoch-1] * 100.));

print "DONE !"                  
# filters=model.layers[0].filters.get_value(borrow=True);

# pickle.dump(test_record, open("../data/ConvNet_test_errors.pkl", "w"));

# for i in xrange(100):
#     image_adr="../data/ConvNet_filters/ConvNet_filter_%d.eps" % (i);
#     plt.imshow(filters[i, 0, :, :], cmap = plt.get_cmap('gray'), interpolation='nearest');
#     plt.axis('off');
#     plt.savefig(image_adr , bbox_inches='tight', pad_inches=0);
#     plt.subplot(8, 8, i);
#     plt.imshow(filters[i, 0, :, :], cmap = plt.get_cmap('gray'), interpolation='nearest');
#     plt.axis('off')
# plt.show();