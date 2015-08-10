"""

"""

import sys
sys.path.append("..")

import numpy as np

import numpy as np
from scipy import linalg
from sklearn.utils import array2d, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator

import scae_destin.datasets as ds


Xtr, Ytr, Xte, Yte=ds.load_CIFAR10_Processed("../data/train.npy",
                                             "../data/train.pkl",
                                             "../data/test.npy",
                                             "../data/test.pkl");
del Ytr,Xte,Yte

Xtr=Xtr.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).mean(3);
# Xtr=np.mean(Xtr, 3)


patches=[]
for image in Xtr:
	patches.append(image[:16,:16].flatten())
	patches.append(image[:16,16:].flatten())
	patches.append(image[16:,:16].flatten())
	patches.append(image[16:,16:].flatten())

patches=np.asarray(patches)

norm_patches=[]
##normalize data
for patch in patches:
	mean = np.mean(patch)
	var = np.var(patch)
	norm_patch=(patch-mean)/np.sqrt(var+10)
	norm_patches.append(norm_patch)

norm_patches=np.asarray(norm_patches)	
print norm_patches.shape
np.save('../data/image_patches.npy', patches)