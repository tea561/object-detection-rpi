import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os

def create_label(row):
    return [row['xmin'], row['xmax'], row['ymin'], row['ymax'], row['class_id']]

def create_box_xywh(row):
    center_x = round((row['xmin'] + row['xmax']) / 2, 2)
    center_y = round((row['ymin'] + row['ymax']) / 2, 2)

    width = round(row['xmax'] - row['xmin'], 2)
    height = round(row['ymax'] - row['ymin'], 2)

    return [center_x, center_y, width, height]

def preprocess_data(path, org_img_size, resized_img_size):
    df = pd.read_csv(path)
    org_h, org_w = org_img_size
    resized_h, resized_w = resized_img_size

    width_ratio = resized_w / org_w
    height_ratio = resized_h / org_h

    df['xmin'] = df['xmin'] * width_ratio
    df['xmax'] = df['xmax'] * width_ratio
    df['ymin'] = df['ymin'] * height_ratio
    df['ymax'] = df['ymax'] * height_ratio 
    df = df.round({'xmin': 2, 'xmax': 2, 'ymin': 2, 'ymax': 2})
    df['label'] = df[['xmin', 'ymin', 'xmax', 'ymax', 'class_id']].values.tolist()
    df['box_xywh'] = df.apply(create_box_xywh, axis=1)
    df = df.groupby('frame').aggregate(lambda tdf: tdf.tolist())
    return df

def read_img(img):
  with tf.io.gfile.GFile(img, 'rb') as fp:
    image = fp.read()
  return image

def wrap_bytes(img):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))

def wrap_float(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# CHECK VALUES FOR BBOX!!
# CHECK IF VALUES SHOULD BE WRAPPED IN ARRAY
def create_tfrecord(images_dir, images, labels, out_path):
    with tf.io.TFRecordWriter(out_path) as writer:
        for i in range(len(images)):
            image_path = os.path.join(images_dir, images[i])
            img_bytes = read_img(image_path)
            features = {
                'image': wrap_bytes(img_bytes),
                'x': wrap_float(np.array(labels[i])[:,0]),
                'y':wrap_float(np.array(labels[i])[:,1]),
                'w':wrap_float(np.array(labels[i])[:,2]),
                'h':wrap_float(np.array(labels[i])[:,3]),
                'class':wrap_float(np.array(labels[i])[:,4])
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            serialized = example.SerializeToString()
            writer.write(serialized)
        

