import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.metrics import classification_report, confusion_matrix

# Define image dimensions and batch size
img_height, img_width = 1080, 1920
batch_size = 32

# Define paths for train and test directories
train_dir = '/content/train'
test_dir = '/content/test'

# Create image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'  # 'binary' for two classes: original and pixelated
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Define a more lightweight CNN model
model = Sequential([
    Conv2D(4, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(8, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(16, activation='relu'),  # Further reduced the number of neurons
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Single output for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model
model.save('/content/pixelated_image_detector.h5')

# Check the model size
model_size = os.path.getsize('/content/pixelated_image_detector.h5') / (1024 * 1024)
print(f'Model Size: {model_size:.2f} MB')

# Calculate and print F1 Score, Precision, and Recall
test_generator.reset()
preds = model.predict(test_generator)
preds = (preds > 0.5).astype(int)
y_true = test_generator.classes
y_pred = preds

print(classification_report(y_true, y_pred, target_names=['Original', 'Pixelated']))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix')
print(cm)

# Convert the model to TensorFlow Lite with more aggressive quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the quantized model 
with open('/content/pixelated_image_detector_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

# Check the quantized model size
quantized_model_size = os.path.getsize('/content/pixelated_image_detector_quantized.tflite') / (1024 * 1024)
print(f'Quantized Model Size: {quantized_model_size:.2f} MB')
