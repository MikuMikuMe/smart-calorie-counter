Creating a sophisticated calorie tracking application that uses machine learning to recognize and categorize food from images is a multi-step process involving image recognition, a database of food items with nutritional information, and a front-end user interface. To demonstrate a simplified version of this application, I will outline a Python program that:

1. Uses a pre-trained deep learning model for image classification.
2. Checks recognized food items against a predefined database to fetch calorie information.
3. Provides a simple command-line interface for user interaction.

For the deep learning model, we can use a pre-trained model like `MobileNetV2` from the `tensorflow` library for image recognition. This will keep things straightforward without needing extensive training from scratch.

### Step-by-Step Python Program

```python
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = MobileNetV2(weights='imagenet')

# Database of food items with their respective calories
food_calories_db = {
    'apple': 52,
    'banana': 89,
    'orange': 47,
    'broccoli': 55,
    'carrot': 41,
    # Add more items as needed
}

def classify_image(img_path):
    """Classifies an image and returns the top prediction."""
    try:
        # Load the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions
        predictions = model.predict(img_array)
        labels = decode_predictions(predictions)

        # Get the top prediction
        top_prediction = labels[0][0]
        predicted_label = top_prediction[1]  # Label name
        confidence = top_prediction[2]  # Confidence score

        return predicted_label, confidence
    except Exception as e:
        print(f"Error during image classification: {e}")
        return None, None

def get_calorie_info(food_item):
    """Fetches calorie information for a given food item."""
    try:
        food_item = food_item.lower()
        if food_item in food_calories_db:
            return food_calories_db[food_item]
        else:
            print("Food item not found in the database.")
            return None
    except Exception as e:
        print(f"Error retrieving calorie information: {e}")
        return None

def main():
    """Main function to run the smart calorie counter."""
    print("Welcome to the Smart Calorie Counter!")

    while True:
        # Ask for the image file path
        img_path = input("Enter the path to the food image (or 'exit' to quit): ")

        if img_path.lower() == 'exit':
            break

        # Classify the image
        label, confidence = classify_image(img_path)
        if label and confidence:
            print(f"Predicted Food Item: {label} (Confidence: {confidence:.2f})")

            # Fetch calorie information
            calories = get_calorie_info(label)
            if calories is not None:
                print(f"Approximate Calories: {calories} kcal")
        else:
            print("Failed to process the image.")

if __name__ == "__main__":
    main()
```

### Key Points of the Program

- **Image Recognition**: The program uses a pre-trained MobileNetV2 model to detect food items in images. We leverage the `decode_predictions` function to translate raw prediction scores into human-readable labels.

- **Calorie Database**: A simple dictionary (`food_calories_db`) maps food items to their calorie content. This could be expanded or replaced by a more sophisticated database in a real application.

- **Error Handling**: We wrap the image classification and calorie fetching functions in `try-except` blocks to handle potential exceptions gracefully.

- **User Interface**: A command-line loop allows users to classify multiple images, providing an opportunity to continue or exit the application.

This simplified application provides a basic framework for implementing a smart calorie counter. In a production environment, the program would benefit from improved user interfaces, a more comprehensive food database, and possibly cloud-based processing to manage heavier computational loads and dynamic database updates.