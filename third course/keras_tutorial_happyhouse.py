# -*- coding:utf-8 -*-

import numpy as np 
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalAveragePooling2D,GlobalMaxPooling2D 
from keras.models import Model 
import keras 
from keras.preprocessing import image 
from keras.utils import layer_utils 
from keras.utils.data_utils import get_file 
from keras.applications.imagenet_utils import preprocess_input 
import pydot  
from keras.utils.vis_utils import model_to_dot 
from keras.utils import plot_model 
from utils.kt_utils import * 

import keras.backend as K 
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt 

def model(input_shape):
    x_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(x_input)
    X = Conv2D(32,(7,7),strides=(1,1),name="conv0")(X)
    X = BatchNormalization(axis=3,name="bn0")(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2),name='max_pool')(X)
    X = Flatten()(X)
    X = Dense(1,activation='sigmoid',name='fc')(X)
    model = Model(inputs=x_input,outputs=X,name='happyhouse')
    return model 

def HappyModel(input_shape):
    x_input = Input(input_shape)
    X = ZeroPadding2D((1,1))(x_input)
    X = Conv2D(8,kernel_size=(3,3),strides=(1,1),name='conv1')(X)
    X = BatchNormalization(axis=3,name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2),strides=(2,2))(X)
    X = ZeroPadding2D((1,1))(X)
    X = Conv2D(16,kernel_size=(3,3),strides=(1,1),name='conv2')(X)
    X = BatchNormalization(axis=3,name='bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2),strides=(2,2))(X)
    X = ZeroPadding2D((1,1))(X)
    X = Conv2D(32,kernel_size=(3,3),strides=(1,1),name='conv3')(X)
    X = BatchNormalization(axis=3,name='bn3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2),strides=(2,2))(X)
    X = Flatten()(X)
    X = Dense(1,activation='sigmoid',name='fc')(X)
    model = Model(inputs=x_input,outputs=X,name='happymodel')
    return model 


if __name__ =="__main__":
    x,y,xtest,ytest,classes = load_dataset() 
    x = x/255. 
    y = y.T 
    xtest = xtest/255. 
    ytest = ytest.T 
    print('x shape:'+str(x.shape))
    print('y shape:'+str(y.shape))
    happymodel = HappyModel((64,64,3))
    happymodel.compile(optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0),loss='binary_crossentropy',metrics=['acc'])
    happymodel.fit(x=x,y=y,epochs=20,batch_size=32)
    preds = happymodel.evaluate(x=xtest,y=ytest)
    print('loss='+str(preds[0]))
    print('acc='+str(preds[1]))
    img = image.load_img('datasets/my_image.jpg',target_size=(64,64))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    print(happymodel.predict(img))


