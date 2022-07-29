import numpy as np
import base64
import io
import glob
import os
import cv2

from PIL import Image

def convert_from_base64(base64_image):
    image_decode = base64.b64decode(str(base64_image))
    return np.array(Image.open(io.BytesIO(image_decode)))

def read_images_from_path(image_folder):
    query_folder = os.path.join(image_folder, "**/*")
    image_paths = glob.glob(query_folder, recursive=True)
    images = [cv2.imread(image_path) for image_path in image_paths]
    return images