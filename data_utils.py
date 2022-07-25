import numpy as np
import argparse
import cv2


def get_digits_data(path):
    data = np.load(path, allow_pickle=True)
    total_nb_data = len(data)
    np.random.shuffle(data)
    data_train = []

    for i in range(total_nb_data):
        data_train.append(data[i])

    print("-------------DONE------------")
    print('The number of train digits data: ', len(data_train))

    return data_train


def get_alphas_data(path):
    data = np.load(path, allow_pickle=True)
    total_nb_data = len(data)

    np.random.shuffle(data)
    data_train = []

    for i in range(total_nb_data):
        data_train.append(data[i])

    print("-------------DONE------------")
    print('The number of train alphas data: ', len(data_train))

    return data_train


def get_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('-w', '--weight_path', help='link to weight path', default="./weights/lsplate.weights")
    args.add_argument('-i', '--image_path', help='link to image file')
    args.add_argument('-c', '--config_path', help='link to config path', default="./cfg/yolov4-custom.cfg")
    args.add_argument('-cl', '--classes_path', default="./cfg/yolo.names")

    return args.parse_args()


def get_labels(path):
    with open(path, 'r') as file:
        lines = file.readlines()

    return [line.strip() for line in lines]


def draw_labels_and_boxes(image, labels, confidence, boxes):
    x_min = round(boxes[0])
    y_min = round(boxes[1])
    x_max = round(boxes[0] + boxes[2])
    y_max = round(boxes[1] + boxes[3])

    confidence = '%.4f' % confidence

    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), thickness=2)
    image = cv2.putText(image, labels, (x_min - 30, y_min), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
    image = cv2.putText(image, confidence, (x_min, y_max + 20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.6, color=(255,255,255), thickness=2)

    return image

def get_output_layers(model):
    layers_name = model.getLayerNames()
    output_layers = [layers_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]

    return output_layers


def order_points(coordinates):
    rect = np.zeros((4, 2), dtype="float32")
    x_min, y_min, width, height = coordinates

    # top left - top right - bottom left - bottom right
    rect[0] = np.array([round(x_min), round(y_min)])
    rect[1] = np.array([round(x_min + width), round(y_min)])
    rect[2] = np.array([round(x_min), round(y_min + height)])
    rect[3] = np.array([round(x_min + width), round(y_min + height)])

    return rect


def convert2Square(image):
    """
    Resize non square image(height != width to square one (height == width)
    :param image: input images
    :return: numpy array
    """

    img_h = image.shape[0]
    img_w = image.shape[1]

    # if height > width
    if img_h > img_w:
        diff = img_h - img_w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = np.zeros(shape=(img_h, (diff//2) + 1))

        squared_image = np.concatenate((x1, image, x2), axis=1)
    elif img_w > img_h:
        diff = img_w - img_h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1

        squared_image = np.concatenate((x1, image, x2), axis=0)
    else:
        squared_image = image

    return squared_image
