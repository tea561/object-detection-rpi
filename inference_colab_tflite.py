import cv2
import os
import sys, getopt
import time
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner


import serial
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from tkinter_app import *
import tensorflow as tf



port = '/dev/ttyACM0'
baudrate = 115600
# Initialize serial port
ser = serial.Serial()
ser.port     = port
ser.baudrate = baudrate
ser.open()
ser.reset_input_buffer()

width = 320
height = 240
bytes_per_pixel = 2
bytes_per_frame = width * height * bytes_per_pixel


interpreter = tf.lite.Interpreter(model_path='./colab_models/model-effi_2.tflite')


def serial_readline():
    data = ser.readline()
    return data.decode("utf-8").strip()

def rgb565_to_rgb888(val):
    r = ((val[0] >> 3) & 0x1f) << 3
    g = (((val[1] >> 5) & 0x07) | ((val[0] << 3) & 0x38)) << 2
    b = (val[1] & 0x1f) << 3
    rgb = np.array([r,g,b], dtype=np.uint8)

    return rgb

def run_inference(image):
    img = cv2.resize(image, (448,448))
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img = img.astype(np.uint8)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    

    interpreter.set_tensor(input_details[0]['index'], img)

    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    inference_time = (end_time - start_time) * 1000
    print("Inference time: ", inference_time, "ms")

    scores = interpreter.get_tensor(output_details[0]['index'])
    boxes = interpreter.get_tensor(output_details[1]['index'])
    num = interpreter.get_tensor(output_details[2]['index'])
    labels = interpreter.get_tensor(output_details[3]['index'])

    

    for i in range (boxes.shape[1]):
        if scores[0, i] > 0.4 and labels[0, i] != 2:
            box = boxes[0, i, :]
            x0 = int(box[1] * image.shape[1])
            y0 = int(box[0] * image.shape[0])
            x1 = int(box[3] * image.shape[1])
            y1 = int(box[2] * image.shape[0])

            w = x1 - x0
            h = y1 - y0

            image = cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 1)

    current_time = time.strftime('%Y%m%d%H%M%S')
    img_name = f"./results/image_{current_time}.jpg"
    cv2.imwrite(img_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))        
            
app = App()
print("Running")


while True:

    data_str = b"0x82"
    while data_str != "<image>":
        data_str = ser.readline()
        try:
            data_str = data_str.decode('utf-8').strip()
        except UnicodeDecodeError:
        # Handle non-UTF-8 data differently
        # For example:
            data_str = ser.readline()

    if str(data_str) == "<image>":
        print("Reading frame")
        data = ser.read(bytes_per_frame)
        img = np.frombuffer(data, dtype=np.uint8)
        img_rgb565 = img.reshape((height, width, bytes_per_pixel))
        img_rgb888 = np.empty((height, width, 3), dtype=np.uint8)

        for y in range(0, height):
            for x in range(0, width):
                img_rgb888[y][x] = rgb565_to_rgb888(img_rgb565[y][x])
        
        data_str = serial_readline()
        if(str(data_str) == "</image>"):
            print("Captured frame")
            run_inference(img_rgb888)
            image_pil = Image.fromarray(img_rgb888)
            image_pil.save("out.bmp")
            test = ImageTk.PhotoImage(image_pil)
            label = Label(app.root, image=test)
            label.image = test
            label.place(x=0, y=0)
        else:
            print("Error: Unable to capture image")
                    
