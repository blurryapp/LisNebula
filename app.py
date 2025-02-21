import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Directory containing images
IMAGE_DIR = "herbig_haro_images"

def get_latest_images(directory, num_images=2):
    """Retrieve the latest N images from the directory."""
    images = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))],
        key=os.path.getmtime,
        reverse=True
    )
    return images[:num_images] if len(images) >= num_images else None

def preprocess_image(image_path):
    """Convert image to grayscale and resize to standard dimensions."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (512, 512))
    return resized_image

def analyze_images(directory):
    """Find the latest two images and compare them."""
    latest_images = get_latest_images(directory)

    if not latest_images or len(latest_images) < 2:
        print("Not enough images to compare.")
        return

    image1_path, image2_path = latest_images

    img1 = preprocess_image(image1_path)
    img2 = preprocess_image(image2_path)

    # Compute absolute difference
    diff = cv2.absdiff(img1, img2)

    # Display images and differences
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1), plt.imshow(img1, cmap="gray"), plt.title("Previous Image")
    plt.subplot(1, 3, 2), plt.imshow(img2, cmap="gray"), plt.title("Latest Image")
    plt.subplot(1, 3, 3), plt.imshow(diff, cmap="hot"), plt.title("Difference")
    plt.savefig('comparison.png')  # Save the plot instead of showing it
    plt.close()  # Close the figure to free memory

# Run the analysis
analyze_images(IMAGE_DIR)
