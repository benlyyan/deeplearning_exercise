# -*- coding:utf-8 -*-
# @author:zb
# date:2018.8

import numpy as np 

def zero_pad(x,pad):
    x_pad = np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values=0)
    return x_pad 

def conv_single_step(a,w,b):
    s = a*w + b 
    z = np.sum(s)
    return z 

def conv_forward(a,w,b,params):
    (m,h,w,c) = a.shape 
    (f,f,_,c_) = w.shape 
    stride = params['stride']
    pad = params['pad']
    n_h = int((h-f+2*pad)/stride)+1
    n_w = int((w-f+2*pad)/stride)+1
    z = np.zeros((m,n_h,n_w,c_))
    a_pad = zero_pad(a,pad)
    for i in range(m):
        pad_temp = a_pad[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(c_):
                    vs = h*stride
                    ve = vs+f 
                    hs = w*stride 
                    he = hs+f 
                    a_prev = pad_temp[vs:ve,hs:he,:]
                    z[i,h,w,c] = np.sum(np.multiply(a_prev,w[:,:,:,c])+b[:,:,:,c])
    cache = (a,w,b,params)
    return z,cache

def pool_forward(a,params,mode='max'):
    (m,h,w,c) = a.shape
    f = params['f']
    stride = params['stride']
    n_h = int(1+(h-f)/stride)
    n_w = int(1+(w-f)/stride)
    z = np.zeros((m,n_h,n_w,c))
    for i in range(m):
        temp_a = a[i]
        for h in range(n_h):
            for w in range(n_w):
                for c_ in range(c):
                    vs = h*stride
                    ve = vs+f 
                    hs = w*stride
                    he = hs+f 
                    a_slice = temp_a[vs:ve,hs:he,c_]
                    if mode =='max':
                        z[i,h,w,c_] = np.max(a_slice)
                    elif mode =='average':
                        z[i,h,w,c_] = np.mean(a_slice)
    cache = (a,params)
    return z,cache      

def conv_backward(dz,cache):
    (a,w,b,params) = cache 
    (m,h,w,c) = a.shape 
    (f,f,_,c_) = w.shape 
    stride = params['stride']
    pad = params['pad']
    (_,h_,w_,_) = dz.shape 
    da = np.zeros(a.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    a_pad = zero_pad(a,pad)
    da_pad = zero_pad(da,pad)

    for i in range(m):
        temp_a = a_pad[i]
        temp_da = da_pad[i]
        for h in range(h_):
            for w in range(w_):
                for c in range(c_):
                    vs = h*stride
                    ve = vs+f 
                    hs = w*stride
                    he = hs + f 
                    temp_a_slice = temp_a[vs:ve,hs:he,:]
                    temp_da[vs:ve,hs:he,:] += w[:,:,:,c]*dz[i,h,w,c] 
                    dw[:,:,:,c] += temp_a_slice*dz[i,h,w,c]
                    db[:,:,:,c] += dz[i,h,w,c]
        da_pad[i,:,:,:] = temp_da[pad:-pad,pad:-pad,:]
    return da_pad,dw,db 

def create_mask(x):
    mask =(x==np.max(x))
    return mask 

def dist_value(dz,shape):
    (h,w) = shape 
    ave = dz/(h*w)
    a = np.ones(shape)*ave 
    return a 

def pool_backward(da,cache,mode='max'):
    (a_prev,params) = cache 
    stride = params['stride']
    f = params['f']
    (m,h,w,c) = a_prev.shape
    (_,h_,w_,c_) = da.shape
    da_prev = np.zeros(a_prev.shape)
    for i in range(m):
        a_prev_temp = a_prev[i]
        for h in range(h_):
            for w in range(w_):
                for c in range(c_):
                    vs = h*stride
                    ve = vs+f 
                    hs = w*stride 
                    he = hs+f 
                    if mode == 'max':
                        a_prev_slice = a_prev_temp[vs:ve,hs:he,c]
                        mask = create_mask(a_prev_slice)
                        da_prev[i,vs:ve,hs:he,c] += mask*da[i,vs,hs,c]
                    elif mode =='average':
                        da_temp = da[i,vs,hs,c]
                        shape = (f,f)
                        da_prev[i,vs:ve,hs:he,c] += dist_value(da_temp,shape)
    return da_prev





if __name__ == "__main__":
    np.random.seed(1)
    x = np.random.randn(4,3,3,2)
    print(x.shape)
    x_pad = zero_pad(x,2)
    print(x_pad.shape)
    # print(x_pad)
