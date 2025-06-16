import os
import cv2
import numpy as np
from PIL import Image
from skimage import transform
import random

def zoom_image(image, target_size=(1024, 1024)):
    return transform.resize(image, target_size, mode='reflect', anti_aliasing=True)

def random_crop(image, crop_size=(224, 224)):
    h, w = image.shape
    top = random.randint(0, h - crop_size[0])
    left = random.randint(0, w - crop_size[1])
    return image[top:top+crop_size[0], left:left+crop_size[1]]

def generate_frames(image, num_frames=8):
    zoomed_image = zoom_image(image)
    frames = []
    for _ in range(num_frames):
        frames.append(random_crop(zoomed_image))
    return frames

def save_video(frames, output_path, fps=1):
    height, width = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=False)
    for frame in frames:
        out.write((frame * 255).astype(np.uint8))
    out.release()

def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path).convert('L')
            image = np.array(image) / 255.0

            frames = generate_frames(image)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.mp4')
            save_video(frames, output_path)

input_folder = '/media/pc/d/2025/BigData/Med_MEGA/finetune_images/medvqa/images'
output_folder = '/media/pc/d/2025/BigData/Med_MEGA/finetune_images/medvqa/videos'

main(input_folder, output_folder)