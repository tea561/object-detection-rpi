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


model_file = './modelfile.eim'
image_path = 'test7.png'

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
    with ImageImpulseRunner(model_file) as runner:
            try:
                model_info = runner.init()
                print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
                labels = model_info['model_parameters']['labels']

                # img = cv2.imread(image)
                # if img is None:
                #     print('Failed to load image', image)
                #     exit(1)

                # # imread returns images in BGR format, so we need to convert to RGB
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # # print(img.shape)
                # # print(type(img[0,0,0]))

                # get_features_from_image also takes a crop direction arguments in case you don't have square images
                features, cropped = runner.get_features_from_image(image)

                res = runner.classify(features)

                if "classification" in res["result"].keys():
                    print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                    for label in labels:
                        score = res['result']['classification'][label]
                        print('%s: %.2f\t' % (label, score), end='')
                    print('', flush=True)

                elif "bounding_boxes" in res["result"].keys():
                    print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                    for bb in res["result"]["bounding_boxes"]:
                        ## TODO :check score 
                        print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                        cropped = cv2.rectangle(cropped, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)

                # the image will be resized and cropped, save a copy of the picture here
                # so you can see what's being passed into the classifier
                        
                current_time = time.strftime('%Y%m%d%H%M%S')
                img_name = f"./results/image_{current_time}.jpg"
                cv2.imwrite(img_name, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

            finally:
                if (runner):
                    runner.stop()

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
    
