from google.colab import drive
drive.mount('/content/drive')

import os
# Define the path to your main folder
main_folder_path = '/content/drive/MyDrive/main'
# Check if the main folder path exists
if not os.path.exists(main_folder_path):
    print("Main folder path does not exist.")
else:
    print("Main folder path exists.")
    # List all directories within the main folder
    folders = [f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))]
    if len(folders) == 0:
        print("No subfolders found in the main folder.")
    else:
        print("Subfolders in the main folder:")
        for folder in folders:
            folder_path = os.path.join(main_folder_path, folder)
            num_files = len(os.listdir(folder_path))
            print(f"Folder Name: {folder}, Number of Files: {num_files}")


 import os
import numpy as np
from PIL import Image

# Define the path to your dataset
dataset_path = '/content/drive/MyDrive/main'

# Function to preprocess image files
def preprocess_images_in_folder(folder_path):
    preprocessed_images = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img = img.convert('RGB')  # Convert image to RGB format (if it's not already)
                img = img.resize((224, 224))  # Resize image to desired dimensions
                img_array = np.array(img)  # Convert image to NumPy array
                img.close()  # Close the image file
                preprocessed_images.append(img_array)
            except Exception as e:
                print(f"Error processing image '{file_path}': {e}")
    return preprocessed_images
# Preprocess images in each folder (class) separately
preprocessed_data = {}
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        preprocessed_images = preprocess_images_in_folder(folder_path)
        preprocessed_data[folder] = preprocessed_images
        print(f"Success: Preprocessing completed for folder '{folder}'")
# Convert preprocessed images to NumPy arrays and normalize pixel values
for folder, images in preprocessed_data.items():
    preprocessed_data[folder] = np.array(images) / 255.0
# Display information about the preprocessed images
for folder, images in preprocessed_data.items():
    print(f"\nFolder: {folder}")
    print(f"Number of preprocessed images: {len(images)}")
    print(f"Shape of preprocessed images: {images.shape}")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# Define the path to your dataset
dataset_path = '/content/drive/MyDrive/main'

# Define parameters for image preprocessing and model training
batch_size = 32
image_size = (224, 224)
num_epochs = 10 # Adjust the number of epochs as needed
# Create an ImageDataGenerator for preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
# Load the dataset
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping]
)
# Visualize training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
# Evaluate the model on the test set
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
# Visualize predictions on sample images
sample_images, sample_labels = next(validation_generator)
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)
plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    plt.subplot(5, 5, i + 1)
    plt.imshow(sample_images[i])
    plt.title(f"True: {np.argmax(sample_labels[i])}, Predicted: {predicted_labels[i]}")
    plt.axis('off')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
# Get sample images and labels
sample_images, sample_labels = next(validation_generator)
# Make predictions on sample images
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)
# Visualize predictions
plt.figure(figsize=(10, 10))
num_samples = min(len(sample_images), 25)
for i in range(num_samples):
    plt.subplot(5, 5, i + 1)
    plt.imshow(sample_images[i])
    # Set title with true and predicted labels
    true_label = np.argmax(sample_labels[i])
    predicted_label = predicted_labels[i]
    title = f"True: {true_label}\nPredicted: {predicted_label}"
    if true_label == predicted_label:
        plt.title(title, color='green')  # Correct prediction
    else:
        plt.title(title, color='red')  # Incorrect prediction
    plt.axis('off')
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
# Get sample images and labels along with their corresponding folder names
sample_images, sample_labels = next(validation_generator)
sample_folder_names = validation_generator.filenames[:len(sample_images)]
# Make predictions on sample images
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)
# Define class indices and corresponding folder names
class_indices = validation_generator.class_indices
class_folder_names = {v: k for k, v in class_indices.items()}
# Visualize predictions
plt.figure(figsize=(10, 10))
num_samples = min(len(sample_images), 25)
for i in range(num_samples):
    plt.subplot(5, 5, i + 1)
    plt.imshow(sample_images[i])
    # Get true and predicted labels
    true_label_index = np.argmax(sample_labels[i])
    true_label = class_folder_names[true_label_index]
    predicted_label = class_folder_names[predicted_labels[i]]
    # Set title with true and predicted labels along with folder names
    true_folder = sample_folder_names[i].split('/')[0]
    title = f"True: {true_label} ({true_folder})\nPredicted: {predicted_label}"
    if true_label == predicted_label:
        plt.title(title, color='green')  # Correct prediction
    else:
        plt.title(title, color='red')  # Incorrect prediction
    plt.axis('off')
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Generate predictions on the validation set
validation_generator.reset()
predictions = model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = validation_generator.classes

# Classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels))

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix with values
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Choose a sample image from the validation set
sample_index = 0  # Change this to select a different sample
sample_image = sample_images[sample_index]

# Convert the sample image to a numpy array
sample_image_array = image.img_to_array(sample_image)
sample_image_array = np.expand_dims(sample_image_array, axis=0)

# Define a function to plot the activation maps of a given layer as heatmaps
def save_activation_maps(model, sample_image, layer_name, num_heatmaps, output_dir):
    # Extract the output of the specified layer
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    activations = activation_model.predict(sample_image)

    # Create output directory if it doesn't exist
    create_directory(output_dir)

    # Plot and save the activation maps as heatmaps
    for i in range(min(num_heatmaps, activations.shape[-1])):
        plt.imshow(activations[0, :, :, i], cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'heatmap_{i+1}.png'))
        plt.close()

# Specify the layer for which you want to visualize the activation maps as heatmaps
layer_name = 'conv2d_1'  # Change this to the name of the desired layer
num_heatmaps = 10  # Number of heatmaps to generate
output_dir = '/content/drive/MyDrive/heatmaps'  # Output directory where heatmaps will be saved

# Generate and save the activation maps for the selected layer as heatmaps
save_activation_maps(model, sample_image_array, layer_name, num_heatmaps, output_dir)


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Choose a sample image from the validation set
sample_index = 0  # Change this to select a different sample
sample_image = sample_images[sample_index]

# Convert the sample image to a numpy array
sample_image_array = image.img_to_array(sample_image)
sample_image_array = np.expand_dims(sample_image_array, axis=0)

# Define a function to plot the activation maps of a given layer as heatmaps
def save_activation_maps(model, sample_image, layer_name, num_heatmaps, output_dir):
    # Extract the output of the specified layer
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    activations = activation_model.predict(sample_image)

    # Create output directory if it doesn't exist
    create_directory(output_dir)

    # Plot and save the activation maps as heatmaps
    for i in range(min(num_heatmaps, activations.shape[-1])):
        plt.imshow(activations[0, :, :, i], cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'heatmap_{i+1}.png'))
        plt.close()

# Specify the layer for which you want to visualize the activation maps as heatmaps
layer_name = 'conv2d_1'  # Change this to the name of the desired layer
num_heatmaps = 10  # Number of heatmaps to generate
output_dir = '/content/drive/MyDrive/heatmaps'  # Output directory where heatmaps will be saved

# Generate and save the activation maps for the selected layer as heatmaps
save_activation_maps(model, sample_image_array, layer_name, num_heatmaps, output_dir)


import numpy as np
import matplotlib.pyplot as plt

# Get sample images and labels along with their corresponding folder names
sample_images, sample_labels = next(validation_generator)
sample_folder_names = validation_generator.filenames[:len(sample_images)]

# Make predictions on sample images
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

# Define class indices and corresponding folder names
class_indices = validation_generator.class_indices
class_folder_names = {v: k for k, v in class_indices.items()}

# Visualize predictions
plt.figure(figsize=(10, 10))
num_samples = min(len(sample_images), 25)
for i in range(num_samples):
    plt.subplot(5, 5, i + 1)
    plt.imshow(sample_images[i])

    # Get true and predicted labels
    true_label_index = np.argmax(sample_labels[i])
    true_label = class_folder_names[true_label_index]
    predicted_label = class_folder_names[predicted_labels[i]]

    # Set title with true and predicted labels along with folder names
    true_folder = sample_folder_names[i].split('/')[0]
    title = f"True: {true_label} ({true_folder})\nPredicted: {predicted_label}"
    if true_label == predicted_label:
        plt.title(title, color='green')  # Correct prediction
    else:
        plt.title(title, color='red')  # Incorrect prediction

    plt.axis('off')

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Generate predictions on the validation set
validation_generator.reset()
predictions = model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = validation_generator.classes

# Classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels))

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix with values
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Get sample images and labels along with their corresponding folder names
sample_images, sample_labels = next(validation_generator)
sample_folder_names = validation_generator.filenames[:len(sample_images)]

# Make predictions on sample images
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

# Define class indices and corresponding folder names
class_indices = validation_generator.class_indices
class_folder_names = {v: k for k, v in class_indices.items()}

# Create a function to plot images and labels without overlap
def plot_images_without_overlap(images, true_labels, predicted_labels, true_folders, predicted_folders):
    num_samples = len(images)
    num_rows = 2
    num_cols = num_samples

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    fig.subplots_adjust(hspace=0.5)

    for i in range(num_samples):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f"True: {true_folders[i]}\nPredicted: {predicted_folders[i]}", color='green' if true_labels[i] == predicted_labels[i] else 'red')
        axes[0, i].axis('off')

        axes[1, i].imshow(images[i])
        axes[1, i].set_title(f"True: {true_folders[i]}\nPredicted: {predicted_folders[i]}", color='green' if true_labels[i] == predicted_labels[i] else 'red')
        axes[1, i].axis('off')

    for ax in axes.flat:
        ax.label_outer()

    plt.show()

# Visualize images and labels without overlap
plot_images_without_overlap(sample_images, np.argmax(sample_labels, axis=1), predicted_labels, sample_folder_names, [class_folder_names[label] for label in predicted_labels])


import numpy as np
import matplotlib.pyplot as plt

# Get sample images and labels along with their corresponding folder names
sample_images, sample_labels = next(validation_generator)
sample_folder_names = validation_generator.filenames[:len(sample_images)]

# Make predictions on sample images
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

# Define class indices and corresponding folder names
class_indices = validation_generator.class_indices
class_folder_names = {v: k for k, v in class_indices.items()}

# Display true and predicted images vertically with folder names
num_samples = len(sample_images)
num_cols = 1
num_rows = min(5, num_samples)  # Display up to 5 images
fig, axes = plt.subplots(num_rows, num_cols, figsize=(5, 20))

for i in range(num_rows):
    true_label = sample_folder_names[i].split('/')[0]
    predicted_label = class_folder_names[predicted_labels[i]]

    axes[i].imshow(sample_images[i])
    axes[i].set_title(f"True: {true_label}\nPredicted: {predicted_label}", fontsize=12)
    axes[i].axis('off')

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Get sample images and labels along with their corresponding folder names
sample_images, sample_labels = next(validation_generator)
sample_folder_names = validation_generator.filenames[:len(sample_images)]

# Make predictions on sample images
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

# Define class indices and corresponding folder names
class_indices = validation_generator.class_indices
class_folder_names = {v: k for k, v in class_indices.items()}

# Find indices where predictions match true labels
correct_indices = np.where(predicted_labels == np.argmax(sample_labels, axis=1))[0]

# Display true and predicted images for correctly predicted samples
num_samples = min(50, len(correct_indices))  # Display up to 50 images
num_cols = 5
num_rows = -(-num_samples // num_cols)  # Calculate number of rows
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))

for i in range(num_samples):
    idx = correct_indices[i]
    true_label = sample_folder_names[idx].split('/')[0]
    predicted_label = class_folder_names[predicted_labels[idx]]

    row = i // num_cols
    col = i % num_cols
    axes[row, col].imshow(sample_images[idx])
    axes[row, col].set_title(f"True: {true_label}\nPredicted: {predicted_label}", fontsize=10)
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()


