from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size=(224, 224)):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith("jpeg"):
            input_path = os.path.abspath(os.path.join(input_folder, filename))
            output_path = os.path.abspath(os.path.join(output_folder, filename))
                
            try:
                # Open the image
                img = Image.open(input_path)

                if img.format == "JPEG" or img.format == "JPG":
                    img = img.convert("RGBA")

                # Get the maximum size while maintaining the aspect ratio
                max_size = max(img.size)
                new_size = (int(img.width * target_size[0] / max_size), int(img.height * target_size[1] / max_size))

                # Resize the image to the maximum size
                img = img.resize(new_size, Image.ANTIALIAS)

                # Create a new image with a white background
                new_img = Image.new("RGB", target_size, "white")

                # Calculate the position to paste the resized image
                paste_position = ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2)

                # Paste the resized image onto the white canvas
                new_img.paste(img, paste_position)

                # Save the resized image
                new_img.save(output_path, format="PNG")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    # Input folder containing the original images (absolute path)
    input_folder = input("Enter the absolute input folder path: ").strip('\"')

    # Output folder where resized images will be saved (absolute path)
    output_folder = input("Enter the absolute output folder path: ").strip('\"')
    resize_images(input_folder, output_folder)
    print("Image resizing complete.")