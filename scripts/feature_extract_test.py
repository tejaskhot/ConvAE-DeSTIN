"""Feature extraction test"""

import numpy as np;
import sys
import theano;
import theano.tensor as T;
sys.path.append("..")

import scae_destin.datasets as ds;
from scae_destin.convnet import ReLUConvLayer;
from scae_destin.convnet import LCNLayer


n_epochs=1;
batch_size=100;

Xtr, Ytr, Xte, Yte=ds.load_CIFAR10("/home/tejas/Desktop/cifar-10-batches-py");

Xtr=np.mean(Xtr, 3);
Xte=np.mean(Xte, 3);
Xtrain=Xtr.reshape(Xtr.shape[0], Xtr.shape[1]*Xtr.shape[2])
Xtest=Xte.reshape(Xte.shape[0], Xte.shape[1]*Xte.shape[2])

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
                      num_filters=50,
                      num_channels=1,
                      fm_size=(32,32),
                      batch_size=batch_size,
                      border_mode="full");
                      
extract=theano.function(inputs=[idx],
                        outputs=layer_0.apply(images),
                        givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]});
                        
print extract(1).shape