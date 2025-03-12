import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop, Resize, Compose, Pad
import torchvision.utils as utils

from model import Generator

def test_model(model_path, test_image_path, output_path):
    # Load the trained generator model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(color_mode="L").to(device)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    # Read and preprocess the test image
    test_image = Image.open(test_image_path).convert('L')
    fixed_size = (256, 256)  # Define a fixed size for the images

    # Define the preprocessing operations
    transform = Compose([
        Resize(fixed_size),
        ToTensor()
    ])

    test_image = transform(test_image).unsqueeze(0).to(device)

    # Use the trained generator model to denoise the image
    with torch.no_grad():
        output_image = generator(test_image)

    # Post-process and save the denoised image
    utils.save_image(output_image, output_path)


if __name__ == "__main__":
    model_path = "checkpoints/netG_epoch_10.pth"  # Path to the trained generator model
    for i in range(1,19):
        test_image_path = f"val_images/noisy/{i}.png"  # Path to the test image
        output_path = f"result/{i}.png"  # Path to save the reconstructed image
        test_model(model_path, test_image_path, output_path)

    # 创建输出目录
    os.makedirs("results", exist_ok=True)

    # 测试模型
    #test_model(model_path, test_image_path, output_path)