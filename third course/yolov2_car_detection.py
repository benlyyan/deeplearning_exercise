# -*- coding:utf-8 -*-

import argparse 
import os 
import matplotlib.pyplot as plt 
import scipy.io 
import scipy.misc 
import numpy as np 
import pandas as pd 
import PIL  
import tensorflow as tf 
from keras import backend as K 
from keras.layers import Input,Lambda,Conv2D
from keras.models import load_model,Model 
from utils.yolo_utils import * 
from utils.yad2k.models.keras_yolo import * 

def yolo_filter_box(box_confidence,boxes,box_class_probs,threshold=0.6):
    box_scores = box_confidence*box_class_probs
    score_index = K.argmax(box_scores,axis=-1)
    score_max = K.max(box_scores,axis=-1)
    box_filter = (score_max>=threshold)
    scores = tf.boolean_mask(score_max,box_filter)
    boxes = tf.boolean_mask(boxes,box_filter)
    classes = tf.boolean_mask(score_index,box_filter)
    return scores,boxes,classes 

def iou(box1,box2):
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    inter_area = (x2-x1)*(y2-y1)
    union_area = area1+area2-inter_area 
    iou_ratio = inter_area/union_area 
    return iou_ratio 

def yolo_non_max_superssion(scores,boxes,classes,max_boxes=10,iou_thresh=0.5):
    max_boxes_tensor = K.variable(max_boxes,dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_ind = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_thresh)
    scores = K.gather(scores,nms_ind)
    boxes = K.gather(boxes,nms_ind)
    classes = K.gather(classes,nms_ind)
    return scores,boxes,classes 

def yolo_eval(yolo_outputs,image_shape=(720.,1280.),max_boxes=10,score_thresh=0.6,iou_thresh=0.5):
    box_conf,box_xy,box_wh,box_class_p = yolo_outputs 
    boxes = yolo_boxes_to_corners(box_xy,box_wh)
    scores,boxes,classes = yolo_filter_box(box_conf,boxes,box_class_p,score_thresh)
    boxes = scale_boxes(boxes,image_shape)
    scores,boxes,classes = yolo_non_max_superssion(scores,boxes,classes,max_boxes)  
    return scores,boxes,classes

def predict(sess,image_file,class_names,scores,boxes,classes):
    image,image_data = preprocess_image('datasets/'+image_file,model_image_size=(416,416))
    out_scores,out_boxes,out_classes = sess.run([scores,boxes,classes],feed_dict={yolo_model.input:image_data,K.learning_phase():0})
    print('found {} boxes for {}'.format(len(out_boxes),image_file))
    colors = generate_colors(class_names)
    draw_boxes(image,out_scores,out_boxes,out_classes,class_names,colors)
    image.save(image_file,quality=90)
    return out_scores,out_boxes,out_classes
    
if __name__ =="__main__":
    # with tf.Session() as test_a:
    #     box_conf = tf.random_normal([19,19,5,1],mean=1,stddev=4,seed=1)
    #     boxes = tf.random_normal([19,19,5,4],mean=1,stddev=4,seed=1)
    #     box_class_p = tf.random_normal([19,19,5,80],mean=1,stddev=4,seed=1)
    #     scores,boxes,classes = yolo_filter_box(box_conf,boxes,box_class_p,threshold=0.6)
    #     print('scores[1]='+str(scores[1].eval()))
    #     print('boxes[1]='+str(boxes[1].eval()))
    #     print('classes[1]='+str(classes[1].eval()))
    #     print('scores.shape='+str(scores.shape))
    #     print('boxes.shape='+str(boxes.shape))
    #     print('classes.shape='+str(classes.shape))
    # box1 = (2,1,4,3)
    # box2 = (1,2,3,4)
    # print('iou='+str(iou(box1,box2)))
    # with tf.Session() as test_b:
    #     scores = tf.random_normal([54,],mean=1,stddev=4,seed=1)
    #     boxes = tf.random_normal([54,4],mean=1,stddev=4,seed=1)
    #     classes = tf.random_normal([54,],mean=1,stddev=4,seed=1)
    #     scores,boxes,classes = yolo_non_max_superssion(scores,boxes,classes,max_boxes=10)
    #     print('scores[1]='+str(scores[1].eval()))
    #     print('boxes[1]='+str(boxes[1].eval()))
    #     print('classes[1]='+str(classes[1].eval()))
    #     print('scores.shape='+str(scores.eval().shape))
    #     print('boxes.shape='+str(boxes.eval().shape))
    #     print('classes.shape='+str(classes.eval().shape))
    # with tf.Session() as test_b:
    #     yolo_outputs = (tf.random_normal([19,19,5,1],mean=1,stddev=4,seed=1),
    #             tf.random_normal([19,19,5,2],mean=1,stddev=4,seed=1),
    #             tf.random_normal([19,19,5,2],mean=1,stddev=4,seed=1),
    #             tf.random_normal([19,19,5,80],mean=1,stddev=4,seed=1))
    #     scores,boxes,classes = yolo_eval(yolo_outputs)    
    #     print('scores[1]='+str(scores[1].eval()))
    #     print('boxes[1]='+str(boxes[1].eval()))
    #     print('classes[1]='+str(classes[1].eval()))
    #     print('scores.shape='+str(scores.eval().shape))
    #     print('boxes.shape='+str(boxes.eval().shape))
    #     print('classes.shape='+str(classes.eval().shape))
    sess = K.get_session()
    class_names = read_classes('datasets/coco_classes.txt')
    anchors = read_anchors('datasets/yolo_anchors.txt')
    image_shape = (720.,1280.)
    yolo_model = load_model('yolo.h5')
    print(yolo_model.summary())
    yolo_outputs = yolo_head(yolo_model.output,anchors,len(class_names))
    scores,boxes,classes = yolo_eval(yolo_outputs,image_shape)
    out_scores,out_boxes,out_classes = predict(sess,'test.jpg',class_names,scores,boxes,classes)





