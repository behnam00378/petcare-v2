import tensorflow 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy

model = load_model('./model_v-03.h5')
preprocessed_image =load_img('./images (1).jpeg', target_size=(256, 256))
preprocessed_image =img_to_array(preprocessed_image)
pre_img=numpy.expand_dims(preprocessed_image, axis=0)
prediction = model.predict(pre_img)
predicted_class='Cat'if prediction[0][0] <0.5 else 'Dog'

print(predicted_class)