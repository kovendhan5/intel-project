import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# Constants
DATASET_DIR = '/content/Image/Pixelated'  # Replace with your dataset directory
OUTPUT_DIR = '/content/Image/Output'  # Output directory for super-resolution images
TFLITE_MODEL_PATH = '/content/pixelated_image_detector_quantized.tflite'  # Path to your TFLite model 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def preprocess_image(image_path):
    """Loads image from path and preprocesses to make it model ready."""
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    hr_image = tf.expand_dims(hr_image, 0)
    hr_image = hr_image / 255.0  # Normalize image to [0, 1]
    return hr_image

def save_image(image, filename):
    """Saves unscaled Tensor Images."""
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 1) * 255.0
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s.jpg" % filename)
    print("Saved as %s.jpg" % filename)

def plot_image(image, title=""):
    """Plots images from image tensors."""
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 1) * 255.0
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)

def process_image(image_path, interpreter):
    """Processes a single image using the TFLite interpreter."""
    hr_image = preprocess_image(image_path)
    plot_image(tf.squeeze(hr_image), title="Original Image")
    save_image(tf.squeeze(hr_image), filename=os.path.join(OUTPUT_DIR, "Original_" + os.path.basename(image_path)))

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], hr_image)

    start = time.time()
    interpreter.invoke()
    fake_image = interpreter.get_tensor(output_details[0]['index'])
    print("Time Taken: %f" % (time.time() - start))

    fake_image = tf.squeeze(fake_image)
    plot_image(fake_image, title="Super Resolution")
    save_image(fake_image, filename=os.path.join(OUTPUT_DIR, "SR_" + os.path.basename(image_path)))

def main():
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    image_paths = [os.path.join(DATASET_DIR, x) for x in os.listdir(DATASET_DIR) if x.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in image_paths:
        print(f"Processing {image_path}")
        process_image(image_path, interpreter)

if _name_ == "_main_":
    main()
