# -*- coding:utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv2D,ZeroPadding2D,Activation,Input,concatenate 
from keras.layers.pooling import MaxPooling2D,AveragePooling2D 
from keras.layers.core import Lambda,Flatten,Dense 
from keras.initializers import glorot_uniform 
from keras import backend  as K 
K.set_image_data_format('channels_first')
import cv2 
import os  
from numpy import genfromtxt  
import pandas as pd 
import tensorflow as tf 
from utils.fr_utils import * 
from utils.inception_blocks_v2 import * 

def triplet_loss(y_true,y_pred,alpha=0.2):
    anchor,pos,neg = y_pred[0],y_pred[1],y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,pos)),axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,neg)),axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss,0))
    return loss 

def verify(image_path,identity,database,model):
    encoding = img_to_encoding(image_path,model)
    dist = np.linalg.norm(encoding-database[identity])
    if dist < 0.7:
        print("it's"+str(identity)+",welcome home")
        door_open = True 
    else:
        print("it's not"+str(identity)+"please go away")
        door_open = False 
    return dist,door_open 

def who_is_it(image_path,database,model):
    encoding = img_to_encoding(image_path,model)
    min_dist = 100
    for name,enco in database.items():
        dist = np.linalg.norm(encoding,enco)
        if dist < min_dist:
            min_dist = dist 
            id_name = name 
    if min_dist > 0.7:
        print("it's not in database")
    else:
        print("it's "+id_name)
    return id_name,min_dist 
    
if __name__=="__main__":
    FRmodel = faceRecoModel(input_shape=(3,96,96))
    FRmodel.compile(optimizer='adam',loss=triplet_loss,metrics=['accuracy'])
    load_weights_from_FaceNet(FRmodel)
    database = {}
    database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
    database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
    database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
    database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
    database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
    database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
    database["sebastiano"] = img_to_encoding("images/sebastiano.jpg",FRmodel)
    database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
    database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
    database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
    database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
    database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
    dist,door_open = verify('images/camera_0.jpg','younes',database,FRmodel)
