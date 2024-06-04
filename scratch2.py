from PIL    import Image
import numpy as np
from glob   import glob
import os

images = glob("*circle.jpeg", root_dir="images/circles/")

unsorted_images = []

for i, image in enumerate(images):
    
    # print(f"Image {i}: {image}")

    unsorted_images.append({
        "original position":    i + 1,
        "image_name":           image,
        "std":                  np.std(np.array(Image.open(f"./images/circles/{image}")))
    })
    
for item in unsorted_images: print(item)

sorted_images = list(unsorted_images)

sorted_images.sort(key = lambda x: x["std"], reverse = True)

for i, item in enumerate(sorted_images): 
    os.rename(f"./images/circles/{item['image_name']}", f"./images/circles/{i + 1}_{item["image_name"]}")
    print(item)

# with Image.open("images/smiley_face_2.jpeg") as image:
    
#     image_array = np.array(image)
    
#     print(image_array.shape)
    
#     print(f"Channel 1 STD: {np.std(image_array[:][:][0])}") # 76.62007379924147
#     print(f"Channel 2 STD: {np.std(image_array[:][:][1])}") # 76.61552474289756
#     print(f"Channel 3 STD: {np.std(image_array[:][:][2])}") # 87.10304215719763