import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# بارگیری مدل‌ها
species_model = load_model('models/species_model.h5')
dog_breed_model = load_model('models/dog_breed_model.h5')
cat_breed_model = load_model('models/cat_breed_model.h5')

# بارگذاری لیبل‌ها
with open('data/dog_breeds_labels.txt', 'r') as f:
    dog_labels = [line.strip() for line in f.readlines()]

with open('data/cat_breeds_labels.txt', 'r') as f:
    cat_labels = [line.strip() for line in f.readlines()]

# پیش‌بینی
def predict_animal(image_path):
    img = preprocess_image(image_path)
    prediction = species_model.predict(img)

    if prediction[0][0] > 0.5:
        animal_type = 'dog'
        breed_pred = dog_breed_model.predict(img)
        breed = dog_labels[np.argmax(breed_pred)]
    else:
        animal_type = 'cat'
        breed_pred = cat_breed_model.predict(img)
        breed = cat_labels[np.argmax(breed_pred)]

    return animal_type, breed_pred, breed

# مثال استفاده
image_path = 'test_image.jpg'
animal_type, breed_pred, breed = predict_animal(image_path)
print(f"Animal: {animal_type}")
print(f"Breed Probabilities: {breed_pred}")
print(f"Predicted Breed: {breed}")