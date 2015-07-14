"""Stacked fixed noise dConvAE test"""

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

import theano
import theano.tensor as T

import scae_destin.datasets as ds
from scae_destin.fflayers import ReLULayer
from scae_destin.fflayers import SoftmaxLayer
from scae_destin.convnet import ReLUConvLayer
from scae_destin.convnet import SigmoidConvLayer
from scae_destin.model import ConvAutoEncoder
from scae_destin.convnet import MaxPoolingSameSize, MaxPooling
from scae_destin.convnet import Flattener
from scae_destin.model import FeedForward
from scae_destin.optimize import gd_updates
from scae_destin.cost import mean_square_cost
from scae_destin.cost import categorical_cross_entropy_cost
from scae_destin.cost import L2_regularization

n_epochs=100
batch_size=100
nkerns=100

Xtr, Ytr, Xte, Yte=ds.load_CIFAR10("../cifar-10-batches-py/")

Xtr=np.mean(Xtr, 3)
Xte=np.mean(Xte, 3)
Xtrain=Xtr.reshape(Xtr.shape[0], Xtr.shape[1]*Xtr.shape[2])/255.0
Xtest=Xte.reshape(Xte.shape[0], Xte.shape[1]*Xte.shape[2])/255.0

train_set_x, train_set_y=ds.shared_dataset((Xtrain, Ytr))
test_set_x, test_set_y=ds.shared_dataset((Xtest, Yte))

n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size
n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size

print "[MESSAGE] The data is loaded"

################################## FIRST LAYER #######################################

X=T.matrix("data")
y=T.ivector("label")
idx=T.lscalar()
corruption_level=T.fscalar()

images=X.reshape((batch_size, 1, 32, 32))

layer_0_en=ReLUConvLayer(filter_size=(4,4),
                         num_filters=50,
                         num_channels=1,
                         fm_size=(32,32),
                         batch_size=batch_size,
                         border_mode="same")
                                                  
layer_0_de=SigmoidConvLayer(filter_size=(4,4),
                            num_filters=1,
                            num_channels=50,
                            fm_size=(32,32),
                            batch_size=batch_size,
                            border_mode="same")
                         
layer_1_en=ReLUConvLayer(filter_size=(2,2),
                         num_filters=50,
                         num_channels=50,
                         fm_size=(8,8),
                         batch_size=batch_size,
                         border_mode="same")
                                                   
layer_1_de=SigmoidConvLayer(filter_size=(2,2),
                            num_filters=50,
                            num_channels=50,
                            fm_size=(8,8),
                            batch_size=batch_size,
                            border_mode="same")


model_0=ConvAutoEncoder(layers=[layer_0_en, MaxPoolingSameSize(pool_size=(4,4)), layer_0_de])
out_0=model_0.fprop(images, corruption_level=corruption_level)
cost_0=mean_square_cost(out_0[-1], images)+L2_regularization(model_0.params, 0.005)
updates_0=gd_updates(cost=cost_0, params=model_0.params, method="sgd", learning_rate=0.1)

## append a max-pooling layer

model_trans=FeedForward(layers=[layer_0_en, MaxPooling(pool_size=(4,4))]);
out_trans=model_trans.fprop(images);


model_1=ConvAutoEncoder(layers=[layer_1_en, MaxPoolingSameSize(pool_size=(2,2)), layer_1_de])
out_1=model_1.fprop(out_trans[-1], corruption_level=corruption_level)
cost_1=mean_square_cost(out_1[-1], out_trans[-1])+L2_regularization(model_1.params, 0.005)
updates_1=gd_updates(cost=cost_1, params=model_1.params, method="sgd", learning_rate=0.1)


train_0=theano.function(inputs=[idx, corruption_level],
                        outputs=[cost_0],
                        updates=updates_0,
                        givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]})

train_1=theano.function(inputs=[idx, corruption_level],
                        outputs=[cost_1],
                        updates=updates_1,
                        givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]})

print "[MESSAGE] The 2-layer model is built"

corr={}
corr[0]=corr[1]=np.random.uniform(low=0.1, high=0.3, size=1).astype("float32")


min_cost={0:None,
          1:None}

corr_best={0:corr[0],
           1:corr[0]}

max_iter={0:0,
          1:0}

epoch = 0
while (epoch < n_epochs):
    epoch = epoch + 1
    c_0 = c_1 = []
    for batch_index in xrange(n_train_batches):
      for rep in xrange(8):
        train_cost=train_1(batch_index, corr_best[1][0])
        c_1.append(train_cost)
        train_cost=train_0(batch_index, corr_best[0][0])
        c_0.append(train_cost)
        
    if min_cost[0]==None:
        min_cost[0]=np.mean(c_0)
    else:
        if (np.mean(c_0)<min_cost[0]*0.5) or (max_iter[0]>=20):
            min_cost[0]=np.mean(c_0)
            corr_best[0][0]=corr[0]
            corr[0]=np.random.uniform(low=corr_best[0][0], high=corr_best[0][0]+0.1, size=1).astype("float32")
            max_iter[0]=0
        else:
            max_iter[0]+=1

    if min_cost[1]==None:
            min_cost[1]=np.mean(c_1)
    else:
        if (np.mean(c_1)<min_cost[1]*0.5) or (max_iter[1]>=20):
            min_cost[1]=np.mean(c_1)
            corr_best[1][0]=corr[1]
            corr[1]=np.random.uniform(low=corr_best[1][0], high=corr_best[1][0]+0.1, size=1).astype("float32")
            max_iter[1]=0
        else:
            max_iter[1]+=1

            
    print 'Training epoch %d, cost ' % epoch, np.mean(c_0), str(corr_best[0][0]), min_cost[0], max_iter[0]
    print '                        ', np.mean(c_1), str(corr_best[1][0]), min_cost[1], max_iter[1]
    
print "[MESSAGE] The model is trained"

################################## BUILD SUPERVISED MODEL #######################################

pool_0=MaxPooling(pool_size=(4,4));
pool_1=MaxPooling(pool_size=(2,2));
flattener=Flattener()
layer_2=ReLULayer(in_dim=50*4*4,
                  out_dim=400)
layer_3=SoftmaxLayer(in_dim=400,
                     out_dim=10)

model_sup=FeedForward(layers=[layer_0_en, pool_0, layer_1_en, pool_1, flattener, layer_2, layer_3])

 
out_sup=model_sup.fprop(images)
cost_sup=categorical_cross_entropy_cost(out_sup[-1], y)
updates=gd_updates(cost=cost_sup, params=model_sup.params, method="sgd", learning_rate=0.1)
 
train_sup=theano.function(inputs=[idx],
                          outputs=cost_sup,
                          updates=updates,
                          givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size],
                                  y: train_set_y[idx * batch_size: (idx + 1) * batch_size]})
 
test_sup=theano.function(inputs=[idx],
                         outputs=model_sup.layers[-1].error(out_sup[-1], y),
                         givens={X: test_set_x[idx * batch_size: (idx + 1) * batch_size],
                                 y: test_set_y[idx * batch_size: (idx + 1) * batch_size]})
                              
print "[MESSAGE] The supervised model is built"

n_epochs=100
test_record=np.zeros((n_epochs, 1))
epoch = 0
while (epoch < n_epochs):
    epoch+=1
    for minibatch_index in xrange(n_train_batches):
        mlp_minibatch_avg_cost = train_sup(minibatch_index)
         
        iteration = (epoch - 1) * n_train_batches + minibatch_index
         
        if (iteration + 1) % n_train_batches == 0:
            print 'MLP MODEL'
            test_losses = [test_sup(i) for i in xrange(n_test_batches)]
            test_record[epoch-1] = np.mean(test_losses)
             
            print(('     epoch %i, minibatch %i/%i, test error %f %%') %
                  (epoch, minibatch_index + 1, n_train_batches, test_record[epoch-1] * 100.))

# filters=[]     
# filters.append(model_sup.layers[0].filters.get_value(borrow=True))
# filters.append(model_sup.layers[1].filters.get_value(borrow=True))
# filters.append(model_sup.layers[2].filters.get_value(borrow=True))
# filters.append(model_sup.layers[3].filters.get_value(borrow=True))
 
filters=model_1.layers[0].filters.get_value(borrow=True);

pickle.dump(test_record, open("convae_destin_100epochs_maxpooling_BtoT.pkl", "w"))
 
for i in xrange(50):
    image_adr="convae_destin/layer_filter_%d.eps" % (i)
    plt.imshow(filters[i, 0, :, :], cmap = plt.get_cmap('gray'), interpolation='nearest')
    plt.axis('off')
    plt.savefig(image_adr , bbox_inches='tight', pad_inches=0)