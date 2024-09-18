import numpy as np
import cv2
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model 
from .forms import ImageUploadForm  

model = load_model(r"C:/Users/mdafi/Desktop/AfifDjangoProject/env/imgClass_final/aiclassifier2.h5")

def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data['image']
            fs = FileSystemStorage()
            file_path = fs.save(img.name, img)
            file_url = fs.url(file_path)

            # Load and preprocess the image
            img_path = fs.path(file_path)
            img = cv2.imread(img_path)

            # Check if the image was loaded correctly
            if img is None:
                return render(request, 'classifier/index.html', {
                    'form': form,
                    'error': 'Error loading the image.'
                })

            # Resize and normalize the image
            resize = tf.image.resize(img, (256, 256))
            img_array = np.expand_dims(resize / 255.0, axis=0)

            # Debug: Print the shape and values of the preprocessed image
            print(f"Preprocessed image shape: {img_array.shape}")
            print(f"Preprocessed image values (first 5): {img_array.flatten()[:5]}")

            # Predict using the loaded model
            prediction = model.predict(img_array)

            # Debug: Print the prediction output
            print(f"Prediction output: {prediction}")

            # Assuming that the model outputs probabilities for each class, and class 1 is 'fake'
            is_fake = prediction[0][0] < 0.5

            context = {
                'form': form,
                'file_url': file_url,
                'is_fake': is_fake,
                'debug': {
                    'preprocessed_image_shape': img_array.shape,
                    'preprocessed_image_values': img_array.flatten()[:5],
                    'prediction_output': prediction
                }
            }
            return render(request, 'classifier/result.html', context)
    else:
        form = ImageUploadForm()
    return render(request, 'classifier/index.html', {'form': form})


    