"""Datasets

This module implements the interface that loads datasets

+ MNIST
+ CIFAR-10
"""

import os;
import gzip;
import cPickle as pickle;

import csv
import numpy as np;

import theano;
import theano.tensor as T;

def shared_dataset(data_xy, borrow=True):
    """Create shared data from dataset
    
    Parameters
    ----------
    data_xy : list
        list of data and its label (data, label)
    borrow : bool
        borrow property
        
    Returns
    -------
    shared_x : shared matrix
    shared_y : shared vector
    """
        
    data_x, data_y = data_xy;
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype='float32'),
                             borrow=borrow);
                             
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype='float32'),
                             borrow=borrow);
        
    return shared_x, T.cast(shared_y, 'int32');

def load_mnist(dataset):
    """Load MNIST dataset
    
    Parameters
    ----------
    dataset : string
        address of MNIST dataset
    
    Returns
    -------
    rval : list
        training, valid and testing dataset files
    """
    
    # Load the dataset
    f = gzip.open(dataset, 'rb');
    train_set, valid_set, test_set = pickle.load(f);
    f.close();
  
    #mean_image=get_mean_image(train_set[0]);

    test_set_x, test_set_y = shared_dataset(test_set);
    valid_set_x, valid_set_y = shared_dataset(valid_set);
    train_set_x, train_set_y = shared_dataset(train_set);

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)];
    return rval;

def load_CIFAR_batch(filename):
    """
    load single batch of cifar-10 dataset
    
    code is adapted from CS231n assignment kit
    
    @param filename: string of file name in cifar
    @return: X, Y: data and labels of images in the cifar batch
    """
    
    with open(filename, 'r') as f:
        datadict=pickle.load(f);
        
        X=datadict['data'];
        Y=datadict['labels'];
        
        X=X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float");
        Y=np.array(Y);
        
        return X, Y;
    
def load_CIFAR10_Processed(train_set,
                           train_set_label,
                           test_set,
                           test_set_label):
    """Load CIFAR-10 Processed image
    
    Parameters
    ----------
    train_set : string
        path to train_set
    test_set : string
        path to test_set
        
    Returns
    -------
    Xtr : object
        training data
    Xte : object
        testing data
    """
    
    Xtr=np.load(open(train_set, 'r'));
    Ytr=pickle.load(open(train_set_label, 'r')).y;
    Xte=np.load(open(test_set, 'r'));
    Yte=pickle.load(open(test_set_label, 'r')).y;
    
    return Xtr, Ytr, Xte, Yte;
        
        
def load_CIFAR10(ROOT):
    """
    load entire CIFAR-10 dataset
    
    code is adapted from CS231n assignment kit
    
    @param ROOT: string of data folder
    @return: Xtr, Ytr: training data and labels
    @return: Xte, Yte: testing data and labels
    """
    
    xs=[];
    ys=[];
    
    for b in range(1,6):
        f=os.path.join(ROOT, "data_batch_%d" % (b, ));
        X, Y=load_CIFAR_batch(f);
        xs.append(X);
        ys.append(Y);
        
    Xtr=np.concatenate(xs);
    Ytr=np.concatenate(ys);
    
    del X, Y;
    
    Xte, Yte=load_CIFAR_batch(os.path.join(ROOT, "test_batch"));
    
    return Xtr, Ytr, Xte, Yte;

def load_fer_2013(filename, expect_labels=True):
    """Load Facial Expression Recognition dataset
    
    Parameters
    ----------
    filename : string
        path of the dataset file
    expect_labels: bool
        if load label
        
    Returns
    -------
    X : array
        images in (number of images, dimension of image)
    Y : array
        label of images
    """
    
    X_path = filename[:-4] + '.X.npy'
    Y_path = filename[:-4] + '.Y.npy'
    if os.path.exists(X_path):
        X = np.load(X_path)
        if expect_labels:
            y = np.load(Y_path)
        else:
            y = None
        return X, y
    
    csv_file = open(filename, 'r');
    reader = csv.reader(csv_file);
    
    row = reader.next()
    
    y_list = [];
    X_list = [];
    
    for row in reader:
        if expect_labels:
            y_str=row[0];
            X_row_str = row[1];
            y = int(y_str)
            y_list.append(y)
        else:
            X_row_str ,= row
        X_row_strs = X_row_str.split(' ')
        X_row = map(lambda x: float(x), X_row_strs)
        X_list.append(X_row)

    X = np.asarray(X_list).astype('float32')
    if expect_labels:
        y = np.asarray(y_list)
    else:
        y = None

    np.save(X_path, X)
    if y is not None:
        np.save(Y_path, y)
    
    return X, y;