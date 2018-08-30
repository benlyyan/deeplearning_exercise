# -*- coding:utf-8 -*-

import numpy as np 
from keras import layers,optimizers  
from keras.layers import Input,Add,Dense,Activation,BatchNormalization,MaxPooling2D,Conv2D,Flatten,ZeroPadding2D
from keras.models import Model,load_model 
from keras.preprocessing import image 
from keras.utils import layer_utils 
from keras.utils.data_utils import get_file 
from keras.applications.imagenet_utils import preprocess_input 
from utils.resnets_utils import * 
from keras.initializers import glorot_uniform 
import scipy.misc 
from matplotlib.pyplot import imshow 

import keras.backend as K 
K.set_image_data_format("channels_last")

def identity_block(X,f,filters,stage,block):
    conv_name = 'res'+str(stage)+block+'_branch'
    bn_name = 'bn'+str(stage)+block+'_branch'
    f1,f2,f3 = filters 
    X_shortcut = X 
    X = Conv2D(filters=f1,kernel_size=(1,1),strides=(1,1),\
        padding='valid',name=conv_name+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name+'2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=f2,kernel_size=(f,f),strides=(1,1),\
        padding='same',name=conv_name+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name+'2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=f3,kernel_size=(1,1),strides=(1,1),\
        padding='valid',name=conv_name+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name+'2c')(X)
    X = layers.add([X,X_shortcut])
    X = Activation('relu')(X)
    return X 

def convolution_block(X,f,filters,stage,block,s=2):
    conv_name = 'res'+str(stage)+block+'_branch'
    bn_name = 'bn'+str(stage)+block+'_branch'
    f1,f2,f3 = filters 
    X_shortcut = X  
    X = Conv2D(filters=f1,kernel_size=(1,1),strides=(s,s),padding='valid',name=conv_name+'2a',\
        kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name+'2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=f2,kernel_size=(f,f),strides=(1,1),padding='same',name=conv_name+'2b',\
        kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name+'2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=f3,kernel_size=(1,1),strides=(1,1),padding='valid',name=conv_name+'2c',\
        kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name+'2c')(X)

    X_shortcut = Conv2D(filters=f3,kernel_size=(1,1),strides=(s,s),padding='valid',kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    X = layers.add([X,X_shortcut])
    X = Activation('relu')(X)
    return X 

def Resnet50(input_shape=(64,64,3),classes=6):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(X_input)
    # stage 1
    X = Conv2D(filters=64,kernel_size=(7,7),strides=(2,2),padding='valid',kernel_initializer=glorot_uniform(seed=0),name='conv1')(X)
    X = BatchNormalization(axis=3,name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
    # stage 2
    X = convolution_block(X,3,[64,64,256],stage=2,block='a',s=1)
    X = identity_block(X,3,[64,64,256],stage=2,block='b')
    X = identity_block(X,3,[64,64,256],stage=2,block='c')
    # stage 3
    X = convolution_block(X,3,[128,128,512],stage=3,block='a',s=2)
    X = identity_block(X,3,[128,128,512],stage=3,block='b')
    X = identity_block(X,3,[128,128,512],stage=3,block='c')
    X = identity_block(X,3,[128,128,512],stage=3,block='d')
    # stage 4 
    X = convolution_block(X,3,[256,256,1024],stage=4,block='a',s=2)
    X = identity_block(X,3,[256,256,1024],stage=4,block='b')
    X = identity_block(X,3,[256,256,1024],stage=4,block='c')
    X = identity_block(X,3,[256,256,1024],stage=4,block='d')
    X = identity_block(X,3,[256,256,1024],stage=4,block='e')
    X = identity_block(X,3,[256,256,1024],stage=4,block='f')
    # stage 5
    X = convolution_block(X,3,[512,512,2048],stage=5,block='a',s=2)
    X = identity_block(X,3,[512,512,2048],stage=5,block='b')
    X = identity_block(X,3,[512,512,2048],stage=5,block='c')
    X = layers.AveragePooling2D(pool_size=(2,2),name='avg_pool')(X)
    X = Flatten()(X)
    X = Dense(classes,activation='softmax',name='fc'+str(classes),kernel_initializer=glorot_uniform(seed=0))(X)
    model = Model(inputs=X_input,outputs=X,name='resnet50')
    return model 


if __name__=="__main__":
    x,y,xtest,ytest,classes = load_dataset()
    x = x/255.
    xtest = xtest/255. 
    y = convert_to_one_hot(y,6).T 
    ytest = convert_to_one_hot(ytest,6).T 
    print('train x shape:'+str(x.shape))
    print('train y shape:'+str(y.shape))
    model = Resnet50(input_shape=(64,64,3),classes=6)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x,y,epochs=2,batch_size=32)
    preds = model.evaluate(xtest,ytest)
    print('preds:'+str(preds))

    pre_trainedmodel = load_model('Resnet50.h5')
    preds1 = pre_trainedmodel.evaluate(xtest,ytest)
    print('preds:'+str(preds1)) 