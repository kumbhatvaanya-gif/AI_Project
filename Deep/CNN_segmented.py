import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks

# ==========================================
# 1. Configuration & Paths
# ==========================================
ORIGINAL_DATA_DIR = "./leukemia_dataset"        # Your raw data
SEGMENTED_DATA_DIR = "./segmented_dataset"      # Where cropped images will be saved

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
EPOCHS = 50
CLASSES = ['benign', 'early', 'pre', 'pro']

# ==========================================
# 2. Automated Image Segmentation (OpenCV)
# ==========================================
def segment_wbc(image_path):
    """
    Reads an image, isolates the prominent purple WBC nucleus using HSV 
    color masking, crops the image to the cell, and blacks out the background.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 1. Convert to HSV color space to easily separate colors
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. Define color range for the purple/blue Giemsa stain of WBCs
    # (These values may need slight tweaking depending on the exact stain lighting)
    lower_purple = np.array([110, 40, 40])
    upper_purple = np.array([170, 255, 255])
    
    # 3. Create a mask isolating the WBC
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # 4. Clean up the mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 5. Find contours to locate the cell
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) # Fallback if no cell found
        
    # Find the largest contour (assuming the main WBC is the largest purple object)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add a small padding box around the cell (e.g., 20 pixels)
    pad = 20
    x, y = max(0, x - pad), max(0, y - pad)
    w, h = min(img.shape[1] - x, w + 2*pad), min(img.shape[0] - y, h + 2*pad)
    
    # 6. Apply mask to original image to black out the background
    segmented_img = cv2.bitwise_and(img, img, mask=mask)
    
    # 7. Crop strictly to the cell
    cropped_img = segmented_img[y:y+h, x:x+w]
    
    # Resize to final CNN input size
    final_img = cv2.resize(cropped_img, (IMG_WIDTH, IMG_HEIGHT))
    
    return final_img

def process_and_save_dataset():
    """Iterates through original folders, segments images, and saves them."""
    print("--- Starting Segmentation Process ---")
    if not os.path.exists(SEGMENTED_DATA_DIR):
        os.makedirs(SEGMENTED_DATA_DIR)
        
    for category in CLASSES:
        orig_folder = os.path.join(ORIGINAL_DATA_DIR, category)
        seg_folder = os.path.join(SEGMENTED_DATA_DIR, category)
        
        if not os.path.exists(seg_folder):
            os.makedirs(seg_folder)
            
        if not os.path.exists(orig_folder):
            print(f"Directory not found: {orig_folder}")
            continue
            
        images = os.listdir(orig_folder)
        print(f"Segmenting {len(images)} images in '{category}'...")
        
        for img_name in images:
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                orig_path = os.path.join(orig_folder, img_name)
                save_path = os.path.join(seg_folder, img_name)
                
                # Segment and save
                seg_img = segment_wbc(orig_path)
                if seg_img is not None:
                    cv2.imwrite(save_path, seg_img)
                    
    print("--- Segmentation Complete ---\n")

# ==========================================
# 3. Data Loading (Using Segmented Images)
# ==========================================
def load_segmented_data():
    """Loads the pre-segmented images for the CNN."""
    print("--- Loading Segmented Datasets ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        SEGMENTED_DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        SEGMENTED_DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    return train_ds.cache().prefetch(buffer_size=AUTOTUNE), val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 4. CNN Architecture
# ==========================================
def build_model():
    """Builds the CNN architecture."""
    # Note: We still use spatial augmentation (flipping/rotation) because 
    # cells can be oriented in any direction, even after segmentation.
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
    ])

    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        data_augmentation,
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(len(CLASSES), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    # 1. Run the segmentation logic on the raw folder to generate clean data
    process_and_save_dataset()
    
    # 2. Load the newly segmented data
    train_ds, val_ds = load_segmented_data()
    
    # 3. Build & Train
    model = build_model()
    
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]
    
    print("\n--- Starting Training on Segmented Data ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_list
    )
