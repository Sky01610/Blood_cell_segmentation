import os


def rename_images(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Sort files to ensure correct order
    files.sort()

    # Rename each file
    for i, filename in enumerate(files):
        new_name = f"{i}.png"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")


if __name__ == "__main__":
    directory = "dataset_restore/noisy"  # Replace with the path to your directory
    rename_images(directory)