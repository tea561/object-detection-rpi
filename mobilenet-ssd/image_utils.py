import cv2
import matplotlib.pyplot as plt

def load_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def show_image(image):
    plt.figure(figsize = (8,10))
    plt.imshow(image)

def show_image_with_bbox(img_path, label, image_size):
    image = cv2.imread(img_path)
    h, w = image_size
    for i, val in enumerate(label):
        xmin = label[i][0]
        ymin = label[i][1]
        xmax = label[i][2]
        ymax = label[i][3]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    show_image(image)

def pre_process_img(img,feature_box_conv,matched):
  img = cv2.imread(img)
  img = cv2.resize(img, (hImage,wImage), interpolation = cv2.INTER_AREA)
  color = (255,0,0)
  matched_idx = np.where(matched)
  for i in (matched_idx):
    for j in i:
      start = feature_box_conv[j,:2]
      end = feature_box_conv[j,2:4]
      start = tuple((start))
      end = tuple((end))
      cv2.rectangle(img,start,end,color,2)
  plt.title('Matched Boxes')
  imshow(img)  