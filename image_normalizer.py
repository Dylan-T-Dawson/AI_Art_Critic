from PIL import Image

def resize_image(input_image, target_size=(224, 224)):
    try:
        # Open the input image
        img = input_image.copy()

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

        # Return the resized image
        return new_img

    except Exception as e:
        print(f"Error processing the image: {e}")
        return None