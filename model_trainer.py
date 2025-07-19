import cv2
import numpy as np
import os
import pickle

def train_model():
    print("Model training is starting...")
    
    # Cascade classifier
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Get faces and labels from the dataset
    faces = []
    labels = []
    label_dict = {}
    current_label = 0
    
    print("Scanning dataset folder...")
    
    # For each person in the dataset folder
    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        print("Error: Dataset folder not found!")
        return
        
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            print(f"Processing: {person_name}")
            label_dict[current_label] = person_name
            
            # For each photo of the person
            for img_name in os.listdir(person_path):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(person_path, img_name)
                    print(f"Reading photo: {img_path}")
                    
                    # Read the image and convert to grayscale
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Warning: Could not read image {img_path}. Skipping.")
                        continue
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Detect face
                    face_rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                    
                    for (x, y, w, h) in face_rects:
                        face_roi = gray[y:y+h, x:x+w]
                        # Resize face to a standard size
                        face_roi = cv2.resize(face_roi, (100, 100))
                        faces.append(face_roi)
                        labels.append(current_label)
                        
            current_label += 1
    
    print(f"Total {len(faces)} faces detected.")
    
    if len(faces) == 0:
        print("No faces detected! Training cannot be performed.")
        return
        
    print("Training model...")
    
    # Create and train the face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    
    # Save the model
    recognizer.save('trainer.yml')
    print("Model saved as trainer.yml.")
    
    # Save the label dictionary
    with open('labels.pkl', 'wb') as f:
        pickle.dump(label_dict, f)
    print("Labels saved as labels.pkl.")
    
    print("\nModel training completed!")
    print(f"Recognized individuals: {list(label_dict.values())}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

