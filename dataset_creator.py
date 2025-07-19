import cv2
import os
import time

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera!")
    exit()

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create dataset folder if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Get user's name
name = input("Please enter your name: ")
user_path = os.path.join('dataset', name)
if not os.path.exists(user_path):
    os.makedirs(user_path)

count = 0
total_images = 30
last_capture_time = time.time()

print("Capturing face photos. Look at the camera and turn your face at different angles...")
print(f"Total {total_images} photos will be captured.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not retrieve frame!")
        break

    # Flip image horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    
    # Face Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    current_time = time.time()
    
    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Capture a photo every 0.5 seconds
        if current_time - last_capture_time >= 0.5 and count < total_images:
            img_path = os.path.join(user_path, f'{name}_{count}.jpg')
            face_img = frame[y:y+h, x:x+w]
            
            # Save the photo
            if face_img.size > 0:
                cv2.imwrite(img_path, face_img)
                print(f"Photo {count+1}/{total_images} saved!")
                count += 1
                last_capture_time = current_time
    
    # Show remaining photo count
    cv2.putText(frame, f"Remaining: {total_images-count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the image
    cv2.imshow('Face Recognition System', frame)
    
    # Exit if all photos are captured or 'q' is pressed
    if count >= total_images:
        print("All photos successfully saved!")
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print(f"Process completed. {count} photos saved to {user_path} folder.")

