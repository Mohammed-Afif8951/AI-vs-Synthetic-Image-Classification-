import tensorflow as tf

# Path to the model
model_path = r"C:/Users/Mafaa/OneDrive/Desktop/Django/imgClass/aiclassifier2.h5"

# Load the model
model = tf.keras.models.load_model(model_path)

# Print the model summary to understand its input and output
model.summary()
