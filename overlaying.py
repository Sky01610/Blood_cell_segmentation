import cv2
import numpy as np

def overlay_red_borders(input_image_path, border_image_path, output_image_path):
    # Read the input image
    input_image = cv2.imread(input_image_path)

    # Read the border image
    border_image = cv2.imread(border_image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure both images have the same dimensions
    if input_image.shape[:2] != border_image.shape[:2]:
        border_image = cv2.resize(border_image, (input_image.shape[1], input_image.shape[0]))

    # Convert the border image to a binary mask
    _, mask = cv2.threshold(border_image, 127, 255, cv2.THRESH_BINARY)

    # Create a red image of the same size as the input image
    red_image = np.zeros_like(input_image)
    red_image[:, :, 2] = 255  # Set the red channel to 255

    # Black-out the area of the border in the input image
    input_bg = cv2.bitwise_and(input_image, input_image, mask=cv2.bitwise_not(mask))

    # Take only the region of the red image where the border is
    red_fg = cv2.bitwise_and(red_image, red_image, mask=mask)

    # Overlay the red border on the input image
    result = cv2.add(input_bg, red_fg)

    # Save the resulting image
    cv2.imwrite(output_image_path, result)

if __name__ == "__main__":
    for i in range(19):
        input_image_path = f"val_images/noisy/{i}.png"
        border_image_path = f"results/{i}.png"
        output_image_path = f"overlay_results/{i}.png"
        overlay_red_borders(input_image_path, border_image_path, output_image_path)