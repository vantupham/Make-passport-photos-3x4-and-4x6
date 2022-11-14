import numpy as np
import cv2
from PIL import Image


LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def crop_image_4x6(image, det,margin):
    left, top, right, bottom = rect_to_tuple(det)
    center = (left + right) // 2, (top + bottom) // 2
    half_width = (right - left) // 2
    half_height = (bottom - top) // 2
    left = int(center[0] - half_width * margin)
    right = int(center[0] + half_width * margin)
    top = int(center[1] - (half_height * margin)*1.5)
    bottom = int(center[1] + (half_height * margin)*1.5)
    image = Image.fromarray(image)
    return image.crop((left, top, right, bottom))
    # return image[top:bottom, left:right]
def crop_image_3x4(image, det,margin):
    left, top, right, bottom = rect_to_tuple(det)
    center = (left + right) // 2, (top + bottom) // 2
    half_width = (right - left) // 2
    half_height = (bottom - top) // 2
    left = int(center[0] - half_width * margin)
    right = int(center[0] + half_width * margin)
    top = int(center[1] - (half_height * margin)*1.33)
    bottom = int(center[1] + (half_height * margin)*1.33)
    image = Image.fromarray(image)
    return image.crop((left, top, right, bottom))