from PIL import Image, ImageFilter
import cv2
import os
import numpy as np
import mediapipe as mp
import torch

def testColorChanging():
    original_image = Image.open("Iris 100/OriginalData/S5000L00.jpg")
    mask_image = Image.open("Iris 100/SegmentationClass/S5000L00.png")
    iris_color_rgb = (102, 255, 102)
    modified_image = original_image.copy()
    original_pixels = modified_image.load()
    mask_pixels = mask_image.load()

    for y in range(mask_image.height):
        for x in range(mask_image.width):
            if mask_pixels[x, y] == iris_color_rgb:
                original_pixels[x, y] += 60
    edges = modified_image.filter(ImageFilter.FIND_EDGES)
    blurred_edges = edges.filter(ImageFilter.GaussianBlur(radius = 20))
    final_image = Image.composite(blurred_edges, modified_image, edges)
    final_image.save("C:/Users/adria/Desktop/S5000L00_modified.jpg")

def testRescale():
    image = cv2.imread("Iris 100/OriginalData/S5000L00.jpg")
    aspect_ratio = image.shape[1] / image.shape[0]
    new_width = 64
    new_height = int(new_width / aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height))
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def splitImagesInHalf(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok = True)
    files = os.listdir(source_folder)

    for file_name in files:
        file_path = os.path.join(source_folder, file_name)
        image = Image.open(file_path)
        width, height = image.size
        midpoint = width // 2
        left_half = image.crop((0, 0, midpoint, height))
        right_half = image.crop((midpoint, 0, width, height))
        left_half_filename = os.path.join(destination_folder, f"{os.path.splitext(file_name)[0]}_left.jpg")
        right_half_filename = os.path.join(destination_folder, f"{os.path.splitext(file_name)[0]}_right.jpg")
        left_half.save(left_half_filename)
        right_half.save(right_half_filename)
        print(f"Images saved as {left_half_filename} and {right_half_filename}")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
left_iris = {474, 475, 476, 477, 473}
left_eye_upper = {398, 384, 385, 386, 387, 388, 466}
left_eye_lower = {263, 249, 390, 373, 374, 380, 381, 382, 362}
right_iris = {470, 469, 472, 471, 468}
right_eye_upper = {173, 157, 158, 159, 160, 161, 246}
right_eye_lower = {33, 7, 163, 144, 145, 153, 154, 155, 133}
right_boundaries = {101, 143, 105, 168}
left_boundaries = {330, 265, 334, 168}

def getEyeBoundingAreaSize(landmarks, eye_indices, image_width, image_height, margin = 0.1):
    eye_landmarks = [landmarks[i] for i in eye_indices]
    x_coords = [lm.x for lm in eye_landmarks]
    y_coords = [lm.y for lm in eye_landmarks]
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    box_width = (max_x - min_x)
    box_height = (max_y - min_y)
    margin_x = box_width * margin
    margin_y = box_height * margin
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y
    min_x = max(0, min_x)
    max_x = min(1, max_x)
    min_y = max(0, min_y)
    max_y = min(1, max_y)
    return int(min_x * image_width), int(min_y * image_height), int(max_x * image_width), int(max_y * image_height)

def getEyeArea(image, landmarks, eye_indices, image_width, image_height, margin = 0.1):
    min_x, min_y, max_x, max_y = getEyeBoundingAreaSize(landmarks, eye_indices, image_width, image_height, margin = margin)
    eye_area = image[min_y:max_y, min_x:max_x]
    return eye_area, (min_x, min_y, max_x, max_y)

def preprocessImage(image, model):
    input_image = np.array(image)
    input_image_resized = cv2.resize(input_image, (256, 256), interpolation = cv2.INTER_CUBIC)
    output = torch.tensor(input_image_resized, dtype = torch.float32).unsqueeze(0).unsqueeze(0).to(model.device)
    return output

def postprocessPrediction(prediction, target_shape):
    target_height, target_width = target_shape
    prediction_resized = cv2.resize(prediction, (target_width, target_height), interpolation = cv2.INTER_LINEAR)
    return (prediction_resized > 0.5).astype(np.uint8) * 255