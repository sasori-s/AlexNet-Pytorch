import os
import cv2
from mtcnn.mtcnn import MTCNN
from PIL import Image

detector = MTCNN()

input_folder = '/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/Faces'  # Ensure this is a directory
output_folder = '/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/Cut_faces'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for person_name in os.listdir(input_folder):
    print(f"\t Now iterating through {person_name}")
    person_path = os.path.join(input_folder, person_name)

    if not os.path.isdir(person_path) or person_name == "images" or person_name == "Cut_faces":
        continue
    
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping {person_name}, folder could not be read.")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        faces = detector.detect_faces(image_rgb)
        
        for i, face in enumerate(faces):
            x, y, width, height = face['box']
            
            x, y = max(0, x), max(0, y)
            width, height = min(image.shape[1] - x, width), min(image.shape[0] - y, height)

            cropped_face = image[y:y+height, x:x+width]
            
            cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
            
            person_output_folder = os.path.join(output_folder, person_name)
            if not os.path.exists(person_output_folder):
                os.makedirs(person_output_folder)

            cropped_face_path = os.path.join(person_output_folder, f'{os.path.splitext(image_name)[0]}_face_{i+1}.jpg')
            cropped_face_pil.save(cropped_face_path)
            
            print(f"Saved cropped face: {cropped_face_path}")

print("Face cropping completed!")