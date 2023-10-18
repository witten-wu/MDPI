import numpy as np
import cv2
import matplotlib.pyplot as plt

# 打开cifar-10数据集文件目录
def unpickle(file):
    import pickle
    with open('C:/Users/csydwu/Downloads/cifar-10-batches-py/'+file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_batch=unpickle("data_batch_1")

cifar_label=data_batch[b'labels']
cifar_data=data_batch[b'data']

#把字典的值转成array格式，方便操作
cifar_label=np.array(cifar_label)
print(cifar_label.shape)
cifar_data=np.array(cifar_data)
print(cifar_data.shape)

label_name=['airplane','automobile','brid','cat','deer','dog','frog','horse','ship','truck']


def imwrite_images():
    num_classes = len(label_name)
    num_images_per_class = 10

    for class_id in range(num_classes):
        class_images = cifar_data[cifar_label == class_id]

        for image_id in range(num_images_per_class):
            image = class_images[image_id]
            label = cifar_label[image_id]
            image = image.reshape(3, 32, 32).transpose(1, 2, 0)

            filename = f"C:/Users/csydwu/Downloads/cifar-10-batches-py/picture/class_{class_id}_image_{image_id}.jpg"
            cv2.imwrite(filename, image)

    print(f"每个标签类别生成{num_images_per_class}张图像完毕")

imwrite_images()


spacing = 5
images = []
for i in range(10):
    for j in range(10):
        image_path = f'C:/Users/csydwu/Downloads/cifar-10-batches-py/picture/class_{i}_image_{j}.jpg'
        image = cv2.imread(image_path)
        images.append(image)

# 将图像列表转换为NumPy数组
images = np.array(images)

# 获取图像的尺寸
image_height, image_width, _ = images[0].shape

# 计算大图的尺寸
num_rows = 10
num_cols = 10  # 包括类别名称列
collage_width = num_cols * (image_width + spacing) - spacing + 80
collage_height = num_rows * (image_height + spacing) - spacing 

# 创建一个空的大图
collage = np.ones((collage_height, collage_width, 3), dtype=np.uint8) * 255

# 将图像复制到大图中
for i, image in enumerate(images):
    # 计算当前图像在大图中的位置
    row = (i // num_cols) * (image_height + spacing) 
    col = (i % num_cols) * (image_width + spacing)

    # 在每一行的最开始处添加类别名称
    if col == 0:
        class_name = label_name[i // num_cols]
        cv2.putText(collage, class_name, (col + spacing, row + image_height - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


    # 将图像复制到大图中的相应位置
    collage[row:row + image_height, col + 80:col + 80 + image_width] = image

    # collage[row:row+image_height, col:col+image_width] = image

save_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
cv2.imwrite("C:/Users/csydwu/Downloads/cifar-10-batches-py/CIFAR-10.png", collage)
cv2.imshow('Collage', collage)
cv2.waitKey(0)
