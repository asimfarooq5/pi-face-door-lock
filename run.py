def create_dataset_train_recognize():
    print("Creating dataset...")
    import dataset_creator
    print("Training model...")
    import model_trainer
    print("Running face recognizer...")
    import face_recognizer

def run_recognizer():
    print("Running face recognizer...")
    import face_recognizer

def main():
    print("Options:")
    print("1. Create dataset, train model, and run face recognizer")
    print("2. Run face recognizer")
    choice = input("Enter your choice (1/2): ")
    
    if choice == '1':
        create_dataset_train_recognize()
    elif choice == '2':
        run_recognizer()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
