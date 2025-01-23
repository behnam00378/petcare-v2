
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image


import os

print("Train GermanDog images:", len(os.listdir('dataset/train/GermanDog')))
print("Train PersianCat images:", len(os.listdir('dataset/train/PersianCat')))
print("Validation GermanDog images:", len(os.listdir('dataset/val/GermanDog')))
print("Validation PersianCat images:", len(os.listdir('dataset/val/PersianCat')))




IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2

# اصلاحات مربوط به تنظیم داده‌ها
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    directory='dataset/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    directory='dataset/val',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

base_model = MobileNetV2(
    input_shape=IMAGE_SIZE + (3,), 
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = False

# ساخت لایه‌های جدید
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
prediction = Dense(NUM_CLASSES, activation='softmax')(x)

# مدل نهایی
model = Model(inputs=base_model.input, outputs=prediction)

# کامپایل مدل
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# آموزش مدل
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# ذخیره مدل
model.save('animal_breed_model.h5')

# تابع پیش‌بینی نژاد
def predict_breed(image_path):
    model = tf.keras.models.load_model('animal_breed_model.h5')
    image = Image.open(image_path)
    image = image.resize(IMAGE_SIZE)
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    
    # پیش‌بینی با مدل
    prediction = model.predict(image_array)
    
    # دریافت برچسب کلاس‌ها از دایرکتوری‌ها
    class_labels = list(train_generator.class_indices.keys())
    
    # پیدا کردن کلاس پیش‌بینی شده
    class_idx = np.argmax(prediction)
    
    return class_labels[class_idx], prediction[0][class_idx]

# مسیر تصویر برای پیش‌بینی
image_path = 'dataset/train/GermanDog/n02106662_11620.jpg'

# انجام پیش‌بینی
breed, confidence = predict_breed(image_path)
print('Predicted breed:', breed)
print('Confidence:', confidence)

print("Done")