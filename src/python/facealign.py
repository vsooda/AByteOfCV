#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Philipp Wagner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the author nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys, math
from PIL import Image
import glob
import os
import cv2
import cv
import numpy as np

def Distance(p1,p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
  if (scale is None) and (center is None):
    return image.rotate(angle=angle, resample=resample)
  nx,ny = x,y = center
  sx=sy=1.0
  if new_center:
    (nx,ny) = new_center
  if scale:
    (sx,sy) = (scale, scale)
  cosine = math.cos(angle)
  sine = math.sin(angle)
  a = cosine/sx
  b = sine/sx
  c = x-nx*a-ny*b
  d = -sine/sy
  e = cosine/sy
  f = y-nx*d-ny*e
  return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
  # calculate offsets in original image
  offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
  offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
  # get the direction
  eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
  # calc rotation angle in radians
  rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
  # distance between them
  dist = Distance(eye_left, eye_right)
  # calculate the reference eye-width
  reference = dest_sz[0] - 2.0*offset_h
  # scale factor
  scale = float(dist)/float(reference)
  # rotate original around the left eye
  image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
  # crop the rotated image
  crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
  crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
  image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
  # resize it
  image = image.resize(dest_sz, Image.ANTIALIAS)
  return image

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE | 0 | cv.CV_HAAR_FIND_BIGGEST_OBJECT)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def readFileNames():
    try:
        inFile = open('path_to_created_csv_file.csv')
    except:
        raise IOError('There is no file named path_to_created_csv_file.csv in current directory.')
        return False

    picPath = []
    picIndex = []

    for line in inFile.readlines():
        if line != '':
            fields = line.rstrip().split(';')
            picPath.append(fields[0])
            picIndex.append(int(fields[1]))

    return (picPath, picIndex)


if __name__ == "__main__":
  #[images, indexes]=readFileNames()
    basepath = '/home/sooda/data/face/'
    images = glob.glob(basepath + '*.png')
    cv2.namedWindow('det',  cv2.WINDOW_AUTOSIZE)
    output_path = basepath + "result/"
    print output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    num = 1
    for img in images:
        num = num + 1
        #image =  Image.open(img)
        print num
        src = cv2.imread(img)
        cascade_fn = "../../data/haarcascade_frontalface_alt2.xml"
        lefteye_fn  = "../../data/haarcascade_mcs_lefteye.xml"
        righteye_fn  = "../../data/haarcascade_mcs_righteye.xml"
        cascade = cv2.CascadeClassifier(cascade_fn)
        lefteye = cv2.CascadeClassifier(lefteye_fn)
        righteye = cv2.CascadeClassifier(righteye_fn)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY);
        gray = cv2.equalizeHist(gray);
        rects = detect(gray, cascade);
        if len(rects) == 0:
             continue
        vis = src.copy()
        draw_rects(vis, rects, (0, 255, 0))
        left_position = []
        right_position = []
        for x1, y1, x2, y2 in rects:
            left_position = []
            right_position = []
            mid_left = 5 * (x1 + x2) / 9;
            mid_right = 4 * (x1 + x2) / 9;
            roi_left = gray[y1:y2, x1:mid_left]
            roi_right = gray[y1:y2, mid_right:x2]
            vis_roi = vis[y1:y2, x1: mid_left]
            subrects = detect(roi_left.copy(), lefteye)
            if len(subrects) == 0:
                continue
            draw_rects(vis_roi, subrects, (255, 0, 0))
            left_box = subrects[0];
            left_position = [(left_box[0] + left_box[2]) / 2, (left_box[1] + left_box[3])/ 2]
            left_position[0] = left_position[0] + x1
            left_position[1] = left_position[1] + y1
            vis_roi = vis[y1:y2, mid_right: x2]
            subrects = detect(roi_right.copy(), righteye)
            if len(subrects) == 0:
                continue
            right_box = subrects[0];
            right_position = [(right_box[0] + right_box[2]) / 2, (right_box[1] + right_box[3])/ 2]
            right_position[0] = right_position[0] + mid_right,
            right_position[1] = right_position[1] + y1

            draw_rects(vis_roi, subrects, (255, 255, 0))
        #cv2.imwrite(output_path+img.rstrip().split('/')[-1], vis)
        output_name = output_path+img.rstrip().split('/')[-1];
        if len(left_position) > 0  and len(right_position) > 0:
            print  left_position
            print  right_position
            imgg = Image.open(img)
            CropFace(imgg, left_position, right_position, offset_pct=(0.3,0.3), dest_sz=(250,300)).save(output_name)
