import tensorflow as tf
import numpy as np

def scale_box(matrix, h, w):
    return tf.stack([matrix[:, 0]*w,
                     matrix[:, 1]*h,
                     matrix[:, 2]*w,
                     matrix[:, 3]*h], axis=-1)


def encode_ground_truth(matched_box, feature_box):
    matched_box = tf.cast(matched_boxes, dtype=tf.float32)
    feature_box = tf.cast(feature_box, dtype=tf.float32)
    
    encoded_values = [
        (matched_box[:,0] - feature_box[:, 0]) / feature_box[:,2],
        (matched_box[:,1] - feature_box[:, 1]) / feature_box[:,3],
        tf.math.log(matched_box[:, 2] / feature_box[:, 2]),
        tf.math.log(matched_box[:, 3] / feature_box[:, 3])
    ]

    scaling_factors = tf.constant([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)

    return tf.stack(encoded_values, axis=-1) / scaling_factors

def decode_ground_truth(matched_box, feature_box):
    matched_box = tf.cast(matched_boxes, dtype=tf.float32)
    feature_box = tf.cast(feature_box, dtype=tf.float32)

    matched_boxes *= [0.1, 0.1, 0.2, 0.2]
    return tf.stack([matched_boxes[:,0] * feature_box[:, 2] + (feature_box[:, 0]),
                    matched_boxes[:,1] * feature_box[:, 3] + (feature_box[:, 1]),
          tf.math.exp(matched_boxes[:,2]) * feature_box[:, 2],
          tf.math.exp(matched_boxes[:,3]) * feature_box[:, 3]],
          axis=-1)
     
def calc_scale_of_default_boxes(k, m, s_max = 0.9, s_min = 0.2):
    # m - number of feature maps
    # k - feature map number

    return s_min + (s_max + s_min) * (k - 1) / (m - 1)

def generate_bboxes(fm_sizes):
    num_of_feature_maps = len(fm_sizes)
    #assert num_of_feature_maps == len(aspect_ratios)

    aspect_ratios = [1, 2, 3, 0.5, 0.333]
    feature_boxes = []

    # for k, fk in enumerate(fm_sizes):
    #     sk = calc_scale_of_default_boxes(k, num_of_feature_maps)
    #     sk_prime = np.sqrt(sk * calc_scale_of_default_boxes(k+1, num_of_feature_maps))
    #     for i in range(fk):
    #         for j in range(fk):
    #             cx = (i + 0.5) / fk
    #             cy = (j + 0.5) / fk
    #             boxes.append([cx, cy, skprime, skprime])

    
    for fm_size in fm_sizes:
        w_ar=[]
        h_ar=[]
        scale = calc_scale_of_default_boxes(fm_size, num_of_feature_maps)
        for i in aspect_ratios:
            if int(i) == 1:
                sk_prime = np.sqrt(scale * calc_scale_of_default_boxes(fm_size+1, num_of_feature_maps))
                w = sk_prime * np.sqrt(i)
                h = sk_prime / np.sqrt(i)
                w_ar.append(w)
                h_ar.append(h)
            w = scale * np.sqrt(i)
            h = scale / np.sqrt(i)
            w_ar.append(w)
            h_ar.append(h)

            
        x_axis = np.linspace(0,fm_size,fm_size+1)
        y_axis=np.linspace(0,fm_size,fm_size+1)
        xx,yy = np.meshgrid(x_axis,y_axis)
        x = [(i+0.5)/(fm_size) for i in xx[:-1,:-1]]
        y = [(i+0.5)/(fm_size) for i in yy[:-1,:-1]]  

        num_of_boxes = 6
        num_of_coordinates = 4
        ndf_boxes = fm_size * fm_size * num_of_boxes
        feature_box = np.zeros((ndf_boxes, num_of_coordinates))
        x = np.array(x).reshape(fm_size*fm_size)
        x = np.repeat(x,num_of_boxes)
        y = np.array(y).reshape(fm_size*fm_size)
        y = np.repeat(y,num_of_boxes)

        w_ar = np.tile(w_ar,fm_size*fm_size)
        h_ar = np.tile(h_ar,fm_size*fm_size)
        feature_box[:,0] = x
        feature_box[:,1] = y
        feature_box[:,2] = w_ar
        feature_box[:,3] = h_ar
        feature_boxes.append(feature_box)
    df_box = np.concatenate(feature_boxes, axis=0)
    return df_box

from shapely.geometry import Polygon
from PIL import Image, ImageDraw
import time
import random

def get_polygon(box):
    xmin, ymin, xmax, ymax = box
    polygon = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
    return polygon
def iou_old(box_1, box_2, pic_num):
    poly_1 = get_polygon(box_1)
    poly_2 = get_polygon(box_2)
    #image = Image.new("RGB", (224, 224), "yellow")
    #draw = ImageDraw.Draw(image)

    # Draw the first box in red
    # draw.polygon([(box_1[0], box_1[1]), (box_1[0], box_1[3]), 
    #               (box_1[2], box_1[3]), (box_1[2], box_1[1])], outline="red")

    # # Draw the second box in blue
    # draw.polygon([(box_2[0], box_2[1]), (box_2[0], box_2[3]), 
    #               (box_2[2], box_2[3]), (box_2[2], box_2[1])], outline="blue")

    # # Save the image with boxes drawn
    # timestamp = str(int(time.time()))
    
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    # draw.text((50, 50), f"IoU: {iou:.4f}", fill="black")
    # image.save(f"test/boxes_with_iou_{pic_num}.png") 
    #print(poly_1)
    #print(poly_2)
    return iou

def iou(box1,box2):
  box1 = tf.cast(box1,dtype=tf.float32)
  box2 = tf.cast(box2,dtype=tf.float32)
  
  x1 = tf.math.maximum(box1[:,None,0],box2[:,0])
  y1 = tf.math.maximum(box1[:,None,1],box2[:,1])
  x2 = tf.math.minimum(box1[:,None,2],box2[:,2])
  y2 = tf.math.minimum(box1[:,None,3],box2[:,3])
  
  #Intersection area
  intersectionArea = tf.math.maximum(0.0,x2-x1)*tf.math.maximum(0.0,y2-y1)

  #Union area
  box1Area = (box1[:,2]-box1[:,0])*(box1[:,3]-box1[:,1])
  box2Area = (box2[:,2]-box2[:,0])*(box2[:,3]-box2[:,1])
  
  unionArea = tf.math.maximum(1e-10,box1Area[:,None]+box2Area-intersectionArea)
  iou = intersectionArea/unionArea
  return tf.clip_by_value(iou,0.0,1.0)


def df_match(labels,iou_matrix):
  max_values = tf.reduce_max(iou_matrix,axis=1)
  max_idx = tf.math.argmax(iou_matrix,axis=1)
  matched = tf.cast(tf.math.greater_equal(max_values,0.5),
                  dtype=tf.float32)
  gt_box = tf.gather(labels,max_idx)
  return gt_box,matched