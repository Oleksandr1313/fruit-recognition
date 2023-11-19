import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random

#DATA PREPROCESSING
#folders to be used for data augmentation
train_dir = r"C:\Main_Folder\project\dataset\fruits-360_dataset\fruits-360\Training"
test_dir = r"C:\Main_Folder\project\dataset\fruits-360_dataset\fruits-360\Test"

# Define the image size and batch size
image_size = (100, 100)
batch_size = 32

# Create an ImageDataGenerator for training data with data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create an ImageDataGenerator for testing data with only rescaling
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=image_size,
    class_mode="categorical",
    batch_size=batch_size
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=image_size,
    class_mode="categorical",
    batch_size=batch_size
)

# Fetch a batch of training data
train_data, train_labels = train_generator.next()

# Fetch a batch of testing data
test_data, test_labels = test_generator.next()

# print(train_data.shape)
# print(test_data.shape)


# MODEL ARCHITECTURE
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(100, 100, 3), padding="same"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax"),
])

#compiling the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# model.summary()

# Train the model using fit method
history = model.fit(train_generator, epochs=8, validation_data=test_generator)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# TESTING
class_names = list(train_generator.class_indices.keys())

# Display 5 images with predicted classes
for i in range(5):
    # Choose a random class
    random_class = random.choice(class_names)

    # Choose a random image from the chosen class
    random_image_path = os.path.join(test_dir, random_class, random.choice(os.listdir(os.path.join(test_dir, random_class))))

    # Load and display the random image
    img = tf.keras.preprocessing.image.load_img(random_image_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    plt.imshow(img_array / 255.0)  # Normalize pixel values
    plt.axis('off')
    plt.show()

    # Preprocess the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)

    # Display the predicted label
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]

    print(f"Predicted Class: {predicted_class_name}")