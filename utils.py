import cv2


def load_image(filename):
    img = cv2.imread(filename)
    return img
