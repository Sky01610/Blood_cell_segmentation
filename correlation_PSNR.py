import cv2
import numpy as np

def correlation(path_1, path_2):
    # Load the images
    img1 = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path_2, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded
    if img1 is None or img2 is None:
        print(f"Error: Could not load {path_1} or {path_2}")
        return

    # Ensure images have the same size
    if img1.shape != img2.shape:
        print(f"Error: Image dimensions do not match for {path_1} and {path_2}")
        return

    # Compute the Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)

    # Compute PSNR
    if mse == 0:
        print(f"PSNR for {path_1} and {path_2}: Infinite (Images are identical)")
    else:
        PIXEL_MAX = 255.0
        psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
        print(f"PSNR for {path_1} and {path_2}: {psnr:.2f} dB")
for i in range (1,19):
    # Run the function for the uploaded images
    image = cv2.imread(f"val_images/clean/{i}.png")
    # Resize to 256x256
    resized_image = cv2.resize(image, (256, 256))
    # Save the resized image
    cv2.imwrite(f"resize/{i}.png", resized_image)
    correlation(f"resize/{i}.png", f"result/{i}.png")
