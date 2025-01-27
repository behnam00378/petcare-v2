import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential, save_model
import argparse
import os

# تنظیمات پیش‌فرض
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
BASE_MODEL = tf.keras.applications.MobileNetV2

def build_model(num_classes):
    """ساخت مدل با Transfer Learning"""
    base_model = BASE_MODEL(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

    return model

def load_data(animal_type):
    """بارگذاری داده‌های آموزشی و اعتبارسنجی"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # مسیرهای دیتاست
    train_dir = f"data/{animal_type}_breeds/train"
    val_dir = f"data/{animal_type}_breeds/validation"

    # بارگیری داده‌ها
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_data = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_data, val_data

def train(animal_type, epochs):
    """فرآیند آموزش مدل"""
    # بررسی وجود دیتاست
    if not os.path.exists(f"data/{animal_type}_breeds/train"):
        raise FileNotFoundError(f"پوشه آموزشی {animal_type} وجود ندارد!")

    # بارگیری داده‌ها
    train_data, val_data = load_data(animal_type)

    # ساخت مدل
    model = build_model(train_data.num_classes)

    # کامپایل مدل
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # آموزش مدل
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=1
    )

    # ذخیره مدل
    model_path = f"models/{animal_type}_breed_model.h5"
    save_model(model, model_path)
    print(f"\nمدل نژاد {animal_type} با موفقیت ذخیره شد: {model_path}")

    # ذخیره لیبل‌ها
    labels_path = f"data/{animal_type}_breeds_labels.txt"
    with open(labels_path, 'w') as f:
        for label, index in train_data.class_indices.items():
            f.write(f"{label}\n")
    print(f"لیبل‌های نژاد {animal_type} در فایل ذخیره شد: {labels_path}")

    return history

if __name__ == "__main__":
    # تنظیمات خط فرمان
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--animal',
        type=str,
        required=True,
        choices=['dog', 'cat'],
        help='نوع حیوان (dog/cat)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help='تعداد دوره‌های آموزش'
    )
    args = parser.parse_args()

    # ایجاد پوشه models اگر وجود نداشته باشد
    os.makedirs("models", exist_ok=True)

    # اجرای آموزش
    train(args.animal, args.epochs)