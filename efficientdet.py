import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


width = 384
height = 384

image_path = 'test5.jpeg'
interpreter = tf.lite.Interpreter(model_path='lite1-detection-default.tflite')

img_org = cv2.imread(image_path)
img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
img = cv2.resize(img_org, (width, height))
img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
img = img.astype(np.uint8)


interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], img)

interpreter.invoke()

boxes = interpreter.get_tensor(output_details[0]['index'])
labels = interpreter.get_tensor(output_details[1]['index'])
scores = interpreter.get_tensor(output_details[2]['index'])
num = interpreter.get_tensor(output_details[3]['index'])
	
print(labels)

fig, ax = plt.subplots()
print(output_details)

ax.imshow(img_org)

for i in range (boxes.shape[1]):
    if scores[0, i] > 0.4 and labels[0, i] != 2:
        box = boxes[0, i, :]
        x0 = int(box[1] * img_org.shape[1])
        y0 = int(box[0] * img_org.shape[0])
        x1 = int(box[3] * img_org.shape[1])
        y1 = int(box[2] * img_org.shape[0])

        rect = patches.Rectangle((x0, y0), (x1 - x0), (y1 - y0), fill=False, color='r')
        ax.add_patch(rect)
        ax.text(x0, (y0 - 10), str(round(scores[0,i], 2)), fontsize=12, color='b')

ax.axis('off')
plt.show()