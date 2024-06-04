from PIL import Image
import os

# Function to create images with specified patterns
def generate_images(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    width, height = 512, 512
    
    # White image
    white_image = Image.new('L', (width, height), 255)
    white_image.save(os.path.join(output_folder, '01_white_image.jpeg'))
    
    # Black image
    black_image = Image.new('L', (width, height), 0)
    black_image.save(os.path.join(output_folder, '02_black_image.jpeg'))
    
    # Half white and half black image
    half_image = Image.new('L', (width, height))
    for x in range(width):
        for y in range(height):
            if x < width // 2:
                half_image.putpixel((x, y), 0)
            else:
                half_image.putpixel((x, y), 255)
    half_image.save(os.path.join(output_folder, '03_half_image.jpeg'))
    
    # Checkered pattern generation function
    def create_checkered_image(width, height, squares):
        img = Image.new('L', (width, height))
        square_size = width // squares
        for x in range(width):
            for y in range(height):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    img.putpixel((x, y), 255)
                else:
                    img.putpixel((x, y), 0)
        return img
    
    # Generate checkered images
    for i, squares in enumerate([2, 4, 8, 16, 32, 64, 128, 256], start=4):
        checkered_image = create_checkered_image(width, height, squares)
        checkered_image.save(os.path.join(output_folder, f'{i:02d}_{squares}x{squares}_checkered.jpeg'))

# Output folder to save images
output_folder = 'generated_images'

# Generate and save the images
generate_images(output_folder)

print(f"Images have been saved to the folder: {output_folder}")