import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

label_name=['shark','jay','frog','peacock','dog']

spacing = 5
images = []
for i in range(5):
    datapath = f'C:/Users/csydwu/Downloads/MDPI/ILSVRC/{label_name[i]}'
    image_files = [f for f in os.listdir(datapath) if os.path.isfile(os.path.join(datapath, f))]
    for image_file in image_files:
        image_path = os.path.join(datapath, image_file)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (64, 64))
        images.append(resized_image)

images = np.array(images)
image_height, image_width, _ = images[0].shape
num_rows = 5
num_cols = 5
collage_width = num_cols * (image_width + spacing) - spacing + 60
collage_height = num_rows * (image_height + spacing) - spacing 
collage = np.ones((collage_height, collage_width, 3), dtype=np.uint8) * 255

for i, image in enumerate(images):
    row = (i // num_cols) * (image_height + spacing) 
    col = (i % num_cols) * (image_width + spacing)
    if col == 0:
        class_name = label_name[i // num_cols]
        cv2.putText(collage, class_name, (col + spacing, row + image_height - 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    collage[row:row + image_height, col + 60:col + 60 + image_width] = image

cv2.imwrite("C:/Users/csydwu/Downloads/MDPI/ILSVRC/ILSVRC.png", collage)
cv2.imshow('Collage', collage)
cv2.waitKey(0)