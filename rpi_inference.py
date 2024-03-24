import tensorflow as tf
import cv2
import numpy as np
# import serial
# from tkinter import *
# from PIL import Image, ImageTk
# import threading

def run_inference(image_path):
    interpreter = tf.lite.Interpreter(model_path='ei-model.lite')

    img_org = cv2.imread(image_path)
    img = cv2.resize(img_org, (320, 320))
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img = img.astype(np.int8)


    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)

    print(output_details)

    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])
    labels = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num = interpreter.get_tensor(output_details[3]['index'])

    return boxes, labels, scores, num

def plot_result(boxes, labels, scores, num):
    fig, ax = plt.subplots()
    print(output_details)

    for i in range (boxes.shape[1]):
        if scores[0, i] > 0.2:
            box = boxes[0, i, :]
            x0 = int(box[1] * img_org.shape[1])
            y0 = int(box[0] * img_org.shape[0])
            x1 = int(box[3] * img_org.shape[1])
            y1 = int(box[2] * img_org.shape[0])

            rect = patches.Rectangle((x0, y0), (x1 - x0), (y1 - y0))
            ax.add_patch(rect)
            
            ax.text(x0, (y0 - 10), str(int(labels[0, i])), fontsize=12, color='r')

    ax.axis('off')
    plt.show()

image_path = './only_cars/test/images/frame_0441.jpg'

boxes, labels, scores, num = run_inference(image_path)
plot_result(boxes, labels, scores, num)

# class App(threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#         self.start()

#     def callback(self):
#         self.root.quit()

#     def run(self):
#         self.root = Tk()
#         self.root.protocol("WM_DELETE_WINDOW", self.callback)

#         self.root.title("Object detection")
#         self.root.geometry('600x600')

#         self.root.mainloop()

# port = '/dev/ttyACM1'
# baudrate = 115600
# # Initialize serial port
# ser = serial.Serial()
# ser.port     = port
# ser.baudrate = baudrate
# ser.open()
# ser.reset_input_buffer()

# width = 320
# height = 240
# bytes_per_pixel = 2
# bytes_per_frame = width * height * bytes_per_pixel

# image = np.empty((height, width, bytes_per_pixel), dtype=np.uint8)

# def serial_readline():
#     data = ser.readline()
#     return data.decode("utf-8").strip()

# def rgb565_to_rgb888(val):
#     r = ((val[0] >> 3) & 0x1f) << 3
#     g = (((val[1] >> 5) & 0x07) | ((val[0] << 3) & 0x38)) << 2
#     b = (val[1] & 0x1f) << 3
#     rgb = np.array([r,g,b], dtype=np.uint8)

#     return rgb

# app = App()

# while True:

#     data_str = serial_readline()

#     if str(data_str) == "<image>":
#         print("Reading frame")
#         data = ser.read(bytes_per_frame)
#         img = np.frombuffer(data, dtype=np.uint8)
#         img_rgb565 = img.reshape((height, width, bytes_per_pixel))
#         img_rgb888 = np.empty((height, width, 3), dtype=np.uint8)

#         for y in range(0, height):
#             for x in range(0, width):
#                 img_rgb888[y][x] = rgb565_to_rgb888(img_rgb565[y][x])
        
#         data_str = serial_readline()
#         if(str(data_str) == "</image>"):
#             print("Captured frame")
#             image_pil = Image.fromarray(img_rgb888)
#             image_pil.save("out.bmp")
#             test = ImageTk.PhotoImage(image_pil)
#             label = Label(app.root, image=test)
#             label.image = test
#             label.place(x=0, y=0)


