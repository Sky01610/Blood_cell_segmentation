import cv2
import numpy as np


def detect_borders(input_image_path, output_image_path):
    # Read the input image
    image = cv2.imread(input_image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Use the Canny edge detector to detect edges
    edges = cv2.Canny(blurred_image, 100, 200)

    # Save the resulting image
    cv2.imwrite(output_image_path, edges)


if __name__ == "__main__":
    for i in range(19):
        input_image_path = f"val_images/noisy/{i}.png"
        output_image_path = f"results_bourder/{i}.png"
        detect_borders(input_image_path, output_image_path)