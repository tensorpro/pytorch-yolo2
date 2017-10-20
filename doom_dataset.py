from __future__ import division
import matplotlib.pyplot as plt
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import numpy as np
from scipy import misc
import cv2
from random import random, sample, randrange
from numpy.random import choice

import os
from os import path
from scipy import misc

def overlay_label(img, sprite, top_left, sprite_label):
    img = img.copy()
    r,c = top_left
    imh, imw = img.shape[:2]
    sh,sw = sprite.shape[:2]
    eh = sh-max(r+sh -imh,0)
    ew = sw-max(c+sw -imw,0)

    
    sprite = sprite.astype(np.float32)
    img = img.astype(np.float32)
    alpha = sprite[:eh, :ew, 3]
    print(img[r:r+eh, c:c+ew].shape)
    print(alpha.shape)
    img[r:r+eh, c:c+ew][alpha>=200]=sprite_label
    return img

def rescale(sprite, scale):
    h,w = sprite.shape[:2]
    return cv2.resize(sprite, max((int(scale*w),4)), max(int(scale*h), 4))

def load_sprites(dir_path):
    sprites = []
    for sprite_name in os.listdir(dir_path):
        sprite_path=path.join(dir_path,sprite_name)
        sprites.append(misc.imread(sprite_path))
    return sprites

def sprite_db(base_dir, label_names=None):
    if label_names:
        sprite_names = []
        for ln in label_names:
            sprite_names.append(ln)
    else:
        sprite_names = os.listdir(base_dir)
    sprites = []
    for sn in sprite_names:
        sprite_dir = path.join(base_dir, sn)
        sprites.append(load_sprites(sprite_dir))
    return sprites

def sprite_stream(sprite_db):
    while True:
        weights = np.log(map(len, sprite_db))
        ps = weights/np.sum(weights)
        sprite_id = choice(range(len(sprite_db)), 1, p=ps)[0]
        possible_sprites = sprite_db[sprite_id]
        sprite = sample(possible_sprites,1)[0]
        yield sprite_id, sprite
    
def sampler(data, fn=None):
    while True:
        smp= sample(data,1)[0]
        if fn:
            smp = fn(smp)
        yield smp

def overlaid(data_s, sprite_s, size=lambda : random()*2+.3, num_sprites=2):
    h,c = data_s.next().shape[:2]
    pos = lambda : (randrange(h), randrange(c))
    while True:
        rgb = data_s.next()
        
        for _ in range(num_sprites):
            lab, sprt = sprite_s.next()
            rgb= overlay_sprite(rgb, rescale(sprt, size()), pos())
        yield rgb


def center_to_corner(bs, h=None, w=None):
    xc, yc, dx, dy = bs
    x1 = xc-dx/2
    x2 = xc+dx/2
    y1 = yc-dy/2
    y2 = yc+dy/2
    if h is not None:
        x1*=w
        x2*=w
    if w is not None:
        y1*=h
        y2*=h
    return np.array([x1,y1,x2,y2])

def corner_to_center(bb):
    x1,y1,x2,y2 = bb
    xc = (x1+x2)/2
    yc = (y1+y2)/2
    dx = (x2-x1)
    dy = (y2-y1)
    return np.array([xc,yc, dx,dy])


def overlay_sprite(img, sprite, top_left):
    img_ = img
    # print('gay')
    # print(img.shape)
    r,c = top_left
    imh, imw = img.shape[:2]
    sh,sw = sprite.shape[:2]
    eh = sh-max(r+sh -imh,0)
    ew = sw-max(c+sw -imw,0)

    sprite = sprite.astype(np.float32)
    img = img.astype(np.float32)
    csprite=(sprite[:eh, :ew, :3])
    # print csprite.shape
    alpha = sprite[:eh, :ew, 3]
    # print alpha.shape
    alpha[alpha<200]=0
    csprite[alpha==0]=0
    scale=255
    for i in range(3):
        upd8 = (scale-alpha)/scale
        img[r:r+eh, c:c+ew,i]*=upd8
    csprite[:,:,0]*(alpha/scale)
    csprite[:,:,1]*(alpha/scale)
    csprite[:,:,2]*(alpha/scale)
    img[r:r+eh, c:c+ew]+=csprite
    sprite = sprite.astype(np.uint8)
    img = img.astype(np.uint8)
    y1,x1 = r/imh,c/imw
    y2,x2 = (r+eh)/imw, (c+eh)/imw
    c2c = corner_to_center([x1,y1,x2,y2])
    corner = np.array([x1,y1,x2,y2])
    x,y = c, r
    w,h = ew, eh
    corner = np.array([x,y,x+w,y+h])
    center = scale_vals(corner_to_center(corner), 1/imh, 1/imw)
    
    return img, center

def a2b(bb, h, w):
    bb = scale_vals(1/h,1/w)
    bb = corner_to_center(bb)
    return bb

def b2a(bb, h, w):
    bb = center_to_corner(bb)
    return np.round_(scale_vals(bb, h, w)).astype(np.uint16)

def normal(center=0, mul=1, minval=0, maxval=np.inf, shape=None):
    return np.clip(np.random.standard_normal(shape) *mul+center, minval, maxval)

class Overlayer:

    def __init__(self, sprite_path=path.expanduser('~/data/sprites'),
                 scale_fn=lambda:normal(1,.3, minval=.3),
                 sprite_count_fn=lambda:normal(3, .3, minval=1, maxval=49)):
        self.sdb = sprite_db(sprite_path)
        self.sprites = sprite_stream(self.sdb)
        self.sprite_count_fn = sprite_count_fn
        self.scale_fn = scale_fn

    def add_sprites(self, image):
        image = image.copy()
        labels = np.zeros(50*5)
        num_sprites = self.sprite_count_fn()

        h,w = image.shape[:2]
        pos_fn = lambda sh, sw: (randrange(h-sh), randrange(w-sw))
        for i in range(int(num_sprites)):
            class_id, sprite = next(self.sprites)
            scale = self.scale_fn()
            sprite = rescale(sprite, scale)
            sh, sw = sprite.shape[:2]
            if sh>= h or sw >= w:
                continue
            pos = pos_fn(sh, sw)
            labels[5*i]=class_id
            image, labels[5*i+1:5*i+5] =overlay_sprite(image, sprite, pos)
        return image, labels
            
        

class DetectData(Dataset):

    def __init__(self, rgb=None, datapath=path.expanduser('~/data/backgrounds.npy'),
                 max_sprites=10, overlayer=Overlayer(), subsize=None):
        if rgb is not None:
            self.rgb = rgb
        else:
            self.rgb = np.load(datapath)
        if subsize is not None:
            self.rgb = self.rgb[:subsize]
        self.size = len(self.rgb)
        self.overlayer = overlayer

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index <= len(self)
        img = self.rgb[index]
        return self.overlayer.add_sprites(img)

def scale_vals(bb, h, w):
    bb = bb.astype(np.float32)
    bb[::2]*=w
    bb[1::2]*=h
    return (bb)
    
def show_bbs(img, bbs):
    # img = img.copy()
    h,w = img.shape[:2]
    bbs = bbs.reshape(-1,5)
    for bb in bbs:
        bb = bb[1:]
        # print(bb)
        lel = scale_vals(center_to_corner(bb),h,w)
        # print('lel', lel)
        x1,y1,x2,y2 = lel.astype(np.uint16)
        cv2.rectangle(img, (x1, y1), (x2,y2), (255,0,0), 1)
    return img
