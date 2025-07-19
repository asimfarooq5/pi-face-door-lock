import cv2
import pickle
import time
import sys
import os
from gpiozero import LED

# Qt/Wayland(Resolve Qt/Wayland error)
os.environ["QT_QPA_PLATFORM"] = "xcb"

# GPIO pin (GPIO pin settings)
# IMPORTANT: This part assumes you are running on a Raspberry Pi.
# If not, you will need to comment out or remove this section and related GPIO calls.
# Connect an LED or a relay to this pin (e.g., GPIO 17).
GPIO_PIN = 17
try:
    # Using LED for simple on/off control of a digital output pin
    # You can also use OutputDevice(GPIO_PIN) if you prefer a more generic output
    output_pin = LED(GPIO_PIN)
    print(f"GPIO pin {GPIO_PIN} successfully initialized!")
    # Ensure the pin starts in a low state (off)
    output_pin.off()
except Exception as e:
    print(f"Warning: Could not initialize GPIO pin {GPIO_PIN}. Functionality will be limited. Error: {e}")
    output_pin = None # Set output_pin to None if initialization fails

def unlock_door():
    if output_pin is None:
        print("GPIO pin not initialized. Cannot trigger door action.")
        return
    try:
        print("Door action: Pin going HIGH...")
        output_pin.on()  # Set the pin HIGH (e.g., turn on LED, activate relay)
        time.sleep(3) # Keep the pin HIGH for 3 seconds
        
        print("Door action: Pin going LOW...")
        output_pin.off()  # Set the pin LOW (e.g., turn off LED, deactivate relay)
        time.sleep(1) # Short delay after turning off
    except Exception as e:
        print(f"GPIO control error: {str(e)}")

# Face recognizer and cascade classifier
recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read('trainer.yml')
except:
    print("Error: trainer.yml file not found! Make sure you have trained the model first.")
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load labels
try:
    with open('labels.pkl', 'rb') as f:
        label_dict = pickle.load(f)
except:
    print("Error: labels.pkl file not found! Make sure you have trained the model first.")
    sys.exit(1)

print("Starting camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera!")
    sys.exit(1)

cap.set(3, 640) # Set width
cap.set(4, 480) # Set height
print("System ready! Press 'q' to exit.")

last_unlock_time = 0  # Last door unlock time
recognition_counter = 0  # Consecutive recognition counter
RECOGNITION_THRESHOLD = 3  # Number of consecutive recognitions required

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not retrieve image!")
            break
            
        # Flip image horizontally
        frame = cv2.flip(frame, 1)
        
        # Create info panel
        info_panel = frame.copy()
        cv2.rectangle(info_panel, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(info_panel, "Face Recognition System - Press 'q' to exit", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        frame = info_panel
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        
        current_time = time.time()
        face_recognized = False
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (100, 100))
            
            try:
                # Face recognition
                label, confidence = recognizer.predict(roi_gray)
                
                # Confidence value is between 0-100; lower value means better match
                if confidence < 85:  # Threshold value
                    name = label_dict[label]
                    match_text = f"Match: {100-confidence:.1f}%"
                    color = (0, 255, 0)  # Green
                    face_recognized = True
                    
                    # Increment consecutive recognition counter
                    recognition_counter += 1
                    
                    # Check for sufficient consecutive recognitions and time interval
                    if recognition_counter >= RECOGNITION_THRESHOLD and current_time - last_unlock_time > 5:
                        print(f"Welcome {name}! (Confidence: {100-confidence:.1f}%)")
                        unlock_door() # Call the modified unlock_door function
                        last_unlock_time = current_time
                        recognition_counter = 0  # Reset counter
                else:
                    name = "Unknown"
                    match_text = f"Match: {100-confidence:.1f}%"
                    color = (0, 0, 255)  # Red
                    recognition_counter = 0  # Reset counter if not recognized
                
                # Draw rectangle and info text
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-60), (x+w, y), color, -1)
                cv2.putText(frame, name, (x+5, y-35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, match_text, (x+5, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Recognition error: {str(e)}")
                recognition_counter = 0
        
        # Reset counter if no face is recognized in the current frame
        if not face_recognized:
            recognition_counter = 0
        
        # Check for exit key ('q')
        cv2.imshow('Face Recognition System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram terminated by user.")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
finally:
    print("\nShutting down system...")
    if output_pin is not None:
        output_pin.off() # Ensure the pin is off on exit
        output_pin.close() # Release the GPIO pin resource
    cap.release()
    cv2.destroyAllWindows()

