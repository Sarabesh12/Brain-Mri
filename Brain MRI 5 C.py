#!/usr/bin/env python
# coding: utf-8

# In[2]:


import zipfile
import os
from tqdm import tqdm

# Unzipping with a progress bar
zip_path = 'data.zip'
extract_path = './data/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for file in tqdm(zip_ref.namelist(), desc='Extracting files'):
        zip_ref.extract(file, extract_path)

# Listing out the contents to ensure proper extraction
print(os.listdir(extract_path))


# In[5]:


pip install opencv-python


# In[6]:


import cv2
print(cv2.__version__)


# In[8]:


import cv2
import os

# Define the path to the main data directory
data_dir = './data/'

# Define the CLAHE function
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert the image to grayscale (CLAHE works on grayscale images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to the grayscale image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(gray_image)
    
    return enhanced_image

# Loop through all folders and files in the data directory
for root, dirs, files in os.walk(data_dir):
    for file in files:
        # Only process image files (assuming they have .png, .jpg, or .jpeg extension)
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, file)
            print(f"Processing: {image_path}")
            
            # Load the image
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error: Failed to load image {image_path}")
                continue  # Skip to the next image if loading failed
            
            # Apply CLAHE to the image
            enhanced_image = apply_clahe(image)
            
            # Optionally, save or display the enhanced image
            # Save the processed image (this will overwrite the original image; you can change the path if needed)
            output_path = os.path.join(root, f"enhanced_{file}")
            cv2.imwrite(output_path, enhanced_image)
            
            # To display the image (optional)
            # cv2.imshow("Enhanced Image", enhanced_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            print(f"Enhanced image saved at: {output_path}")


# In[9]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator for data augmentation
data_gen_args = dict(rotation_range=30,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args, rescale=1./255)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Normalize images and augment using ImageDataGenerator
def preprocess_and_augment(image, mask):
    image = image_datagen.random_transform(image)
    mask = mask_datagen.random_transform(mask)
    return image, mask


# In[13]:


import os

# Inspect the directory structure
for root, dirs, files in os.walk('./data/'):
    print(f"Root: {root}")
    print(f"Directories: {dirs}")
    print(f"Files: {files}")
    print('-' * 40)


# In[14]:


image_paths = sorted(glob.glob('./data/Data/**/images/*.png', recursive=True))
mask_paths = sorted(glob.glob('./data/Data/**/masks/*.png', recursive=True))


# In[15]:


import glob
from sklearn.model_selection import train_test_split

# Update the glob patterns based on the folder structure you observed
image_paths = sorted(glob.glob('./data/Data/**/images/*.png', recursive=True))
mask_paths = sorted(glob.glob('./data/Data/**/masks/*.png', recursive=True))

# Print the number of images and masks found
print(f"Number of images found: {len(image_paths)}")
print(f"Number of masks found: {len(mask_paths)}")

# Check if there are any images and masks found
if len(image_paths) == 0 or len(mask_paths) == 0:
    raise ValueError("No images or masks found. Check the folder structure and file extensions.")

# Check that the number of images matches the number of masks
assert len(image_paths) == len(mask_paths), "Mismatch between the number of images and masks."

# Split the dataset into training and testing sets (80% train, 20% test)
train_images, test_images, train_masks, test_masks = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

# Output the size of the training and testing sets
print(f"Number of training images: {len(train_images)}")
print(f"Number of testing images: {len(test_images)}")


# In[18]:


import os

# Inspect the directory structure
for root, dirs, files in os.walk('./data/'):
    print(f"Root: {root}")
    print(f"Directories: {dirs}")
    print(f"Files: {files}")
    print('-' * 40)


# In[21]:


import glob
from sklearn.model_selection import train_test_split

# Update the glob patterns to search for .tif files
image_paths = sorted(glob.glob('./data/**/*.tif', recursive=True))  # Load all TIFF images
mask_paths = sorted(glob.glob('./data/**/*.tif', recursive=True))   # Load all TIFF masks

# Print the number of images and masks found
print(f"Number of images found: {len(image_paths)}")
print(f"Number of masks found: {len(mask_paths)}")

# Check if there are any images and masks found
if len(image_paths) == 0 or len(mask_paths) == 0:
    raise ValueError("No images or masks found. Check the folder structure and file extensions.")

# Check that the number of images matches the number of masks
assert len(image_paths) == len(mask_paths), "Mismatch between the number of images and masks."

# Split the dataset into training and testing sets (80% train, 20% test)
train_images, test_images, train_masks, test_masks = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

# Output the size of the training and testing sets
print(f"Number of training images: {len(train_images)}")
print(f"Number of testing images: {len(test_images)}")


# In[22]:


import tensorflow as tf
from tensorflow.keras import layers, models

def unet_plus_plus(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Add more layers as required for Nested U-Net

    # Decoder and skip connections
    up3 = layers.UpSampling2D(size=(2, 2))(conv2)
    merge3 = layers.concatenate([conv1, up3], axis=3)
    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge3)
    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv3)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv3)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Initialize the model
nested_unet_model = unet_plus_plus()
nested_unet_model.summary()


# In[23]:


def attention_unet(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Add attention gates
    def attention_block(g, x):
        wg = layers.Conv2D(1, 1, activation='sigmoid')(g)
        wx = layers.Conv2D(1, 1, activation='sigmoid')(x)
        alpha = layers.multiply([wg, wx])
        return alpha

    # Decoder and attention blocks
    up3 = layers.UpSampling2D(size=(2, 2))(conv2)
    attention3 = attention_block(up3, conv1)
    merge3 = layers.concatenate([conv1, attention3], axis=3)
    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge3)
    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv3)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv3)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Initialize the Attention U-Net model
attention_unet_model = attention_unet()
attention_unet_model.summary()


# In[21]:


import cv2
import numpy as np

def load_images_and_masks(image_paths, mask_paths):
    images = []
    masks = []
    
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        if img is not None:
            images.append(img)
    
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask in grayscale
        if mask is not None:
            masks.append(mask)
    
    return np.array(images), np.array(masks)

# Load the images and masks
train_images_np, train_masks_np = load_images_and_masks(train_images, train_masks)
test_images_np, test_masks_np = load_images_and_masks(test_images, test_masks)

# Print the shapes to verify
print(f"Train images shape: {train_images_np.shape}")
print(f"Train masks shape: {train_masks_np.shape}")
print(f"Test images shape: {test_images_np.shape}")
print(f"Test masks shape: {test_masks_np.shape}")


# In[31]:


print("Train Images Shape:", train_images_np.shape)  # Should be (num_samples, height, width, channels)
print("Train Masks Shape:", train_masks_np.shape)      # Should be (num_samples, height, width, num_classes)


# In[32]:


print("Unique values in train masks:", np.unique(train_masks_np))


# In[34]:


# Optional: A quick forward pass to check for output shape
test_output = nested_unet_model.predict(train_images_np[0:1])  # Test prediction on the first image
print("Test Output Shape:", test_output.shape)


# In[13]:


# Import necessary libraries for model building
from tensorflow.keras import layers, models

# Model Definitions
def create_nested_unet(input_size=(256, 256, 1)):
    # Define your U-Net architecture here
    # (This is a simplified version)
    inputs = layers.Input(input_size)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    return models.Model(inputs, outputs)

def create_attention_unet(input_size=(256, 256, 1)):
    # Define your Attention U-Net architecture here
    # (This is a simplified version)
    inputs = layers.Input(input_size)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    return models.Model(inputs, outputs)

# Create model instances
nested_unet_model = create_nested_unet()
attention_unet_model = create_attention_unet()

# Compile models
nested_unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
attention_unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Continue with your previous training code...


# In[14]:


from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class DataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=16, image_size=(256, 256), shuffle=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.image_paths[k] for k in indexes]
        batch_masks = [self.mask_paths[k] for k in indexes]

        X, y = self.__data_generation(batch_images, batch_masks)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_images, batch_masks):
        X = np.empty((self.batch_size, *self.image_size, 1))  # Adjust channels if needed
        y = np.empty((self.batch_size, *self.image_size, 1))  # Adjust channels if needed

        for i, (img_path, mask_path) in enumerate(zip(batch_images, batch_masks)):
            img = img_to_array(load_img(img_path, target_size=self.image_size)) / 255.0
            mask = img_to_array(load_img(mask_path, target_size=self.image_size, color_mode="grayscale")) / 255.0

            # Ensure the mask is binary
            mask = np.where(mask > 0.5, 1.0, 0.0)

            X[i,] = np.expand_dims(img, axis=-1)  # Add channel dimension
            y[i,] = np.expand_dims(mask, axis=-1)  # Add channel dimension

        return X, y


# In[15]:


# Parameters
batch_size = 16  # Set your desired batch size
image_size = (256, 256)  # Adjust this according to your model's input size

# Create data generators
train_generator = DataGenerator(train_images, train_masks, batch_size=batch_size, image_size=image_size)
test_generator = DataGenerator(test_images, test_masks, batch_size=batch_size, shuffle=False)


# In[17]:


from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class DataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=16, image_size=(256, 256), shuffle=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.image_paths[k] for k in indexes]
        batch_masks = [self.mask_paths[k] for k in indexes]

        X, y = self.__data_generation(batch_images, batch_masks)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_images, batch_masks):
        X = np.empty((self.batch_size, *self.image_size, 1))  # Expecting grayscale images
        y = np.empty((self.batch_size, *self.image_size, 1))  # Expecting binary masks

        for i, (img_path, mask_path) in enumerate(zip(batch_images, batch_masks)):
            img = img_to_array(load_img(img_path, target_size=self.image_size, color_mode='grayscale')) / 255.0
            mask = img_to_array(load_img(mask_path, target_size=self.image_size, color_mode='grayscale')) / 255.0

            # Ensure the mask is binary
            mask = np.where(mask > 0.5, 1.0, 0.0)

            # Add channel dimension
            X[i,] = np.expand_dims(img, axis=-1)  # Ensure shape is (256, 256, 1)
            y[i,] = np.expand_dims(mask, axis=-1)  # Ensure shape is (256, 256, 1)

        return X, y

    


# In[23]:


from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class DataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=16, image_size=(256, 256), shuffle=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.image_paths[k] for k in indexes]
        batch_masks = [self.mask_paths[k] for k in indexes]

        X, y = self.__data_generation(batch_images, batch_masks)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_images, batch_masks):
        X = np.empty((self.batch_size, *self.image_size, 1))  # Expecting grayscale images
        y = np.empty((self.batch_size, *self.image_size, 1))  # Expecting binary masks

        for i, (img_path, mask_path) in enumerate(zip(batch_images, batch_masks)):
            # Load image and mask in grayscale mode
            img = img_to_array(load_img(img_path, target_size=self.image_size, color_mode='grayscale')) / 255.0
            mask = img_to_array(load_img(mask_path, target_size=self.image_size, color_mode='grayscale')) / 255.0

            # Ensure the mask is binary
            mask = np.where(mask > 0.5, 1.0, 0.0)

            # Add channel dimension if needed
            if img.ndim == 2:  # If the image is 2D (H, W)
                img = np.expand_dims(img, axis=-1)  # Shape becomes (H, W, 1)
            if mask.ndim == 2:  # If the mask is 2D (H, W)
                mask = np.expand_dims(mask, axis=-1)  # Shape becomes (H, W, 1)

            X[i,] = img  # Shape should be (256, 256, 1)
            y[i,] = mask  # Shape should be (256, 256, 1)

        return X, y


# In[24]:


# Train the models
epochs = 25  # Total number of epochs
steps_per_epoch = min(15, len(train_generator))  # Limit to 15 batches

history_nested_unet = nested_unet_model.fit(train_generator, validation_data=test_generator, 
                                             epochs=epochs, steps_per_epoch=steps_per_epoch)

history_attention_unet = attention_unet_model.fit(train_generator, validation_data=test_generator, 
                                                  epochs=epochs, steps_per_epoch=steps_per_epoch)


# In[ ]:




