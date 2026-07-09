import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. Configuration & Paths
# ==========================================
# Update this to the path where your folders (benign, early, pre, pro) are located
DATA_DIR = "./leukemia_dataset" 

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
EPOCHS = 50
CLASSES = ['benign', 'early', 'pre', 'pro']

# ==========================================
# 2. Dataset Diagnostic (Check Image Sizes)
# ==========================================
def check_image_sizes(data_directory):
    """Scans the dataset to check if images are uniform in size or need resizing."""
    print("--- Diagnosing Image Sizes ---")
    sizes = set()
    total_images = 0
    
    for category in CLASSES:
        folder_path = os.path.join(data_directory, category)
        if not os.path.exists(folder_path):
            print(f"Warning: Directory not found - {folder_path}")
            continue
            
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    sizes.add(img.shape) # returns (height, width, channels)
                    total_images += 1
                    
    if len(sizes) == 1:
        print(f"All {total_images} images are already a uniform size: {list(sizes)[0]}.")
        print("Resizing will still be applied in the pipeline to ensure tensor compatibility.")
    else:
        print(f"Found {len(sizes)} different image sizes in the dataset.")
        print("Resizing is REQUIRED and will be handled automatically by the data loader.")
        print(f"Sample of sizes found: {list(sizes)[:5]}")
    print("------------------------------\n")

# ==========================================
# 3. Data Loading & Splitting
# ==========================================
def create_datasets(data_directory):
    """Loads images, automatically resizes them, and splits into Train/Val/Test."""
    print("--- Loading Datasets ---")
    
    # 80% Training Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_directory,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    # 20% Validation Data (We will split this further into Val and Test)
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_directory,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    # Split the 20% validation into 10% validation and 10% testing
    val_batches = tf.data.experimental.cardinality(val_test_ds)
    test_ds = val_test_ds.take(val_batches // 2)
    val_ds = val_test_ds.skip(val_batches // 2)

    # Optimize performance with prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds

# ==========================================
# 4. Model Architecture Design
# ==========================================
def build_model():
    """Builds a custom CNN with Data Augmentation and Regularization."""
    
    # Built-in Data Augmentation layer (Only active during training)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ], name="data_augmentation")

    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        data_augmentation,
        
        # Normalize pixel values to [0, 1]
        layers.Rescaling(1./255),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier Head
        layers.GlobalAveragePooling2D(), # Better than Flatten() for reducing overfitting
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), # Prevent overfitting
        layers.Dense(len(CLASSES), activation='softmax') # 4 classes
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==========================================
# 5. Training, Evaluation & Plotting
# ==========================================
def plot_training_history(history):
    """Plots Training/Validation Accuracy and Loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='o')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='o')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.show()

def evaluate_and_report(model, test_ds):
    """Generates the Classification Report Table and Confusion Matrix."""
    print("\n--- Evaluating Model on Test Set ---")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {accuracy*100:.2f}%\n")

    # Extract true labels and predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Output Classification Table
    print("\n================== CLASSIFICATION REPORT ==================")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    print("===========================================================\n")

    # 2. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# ==========================================
# 6. Main Execution Block
# ==========================================
if __name__ == "__main__":
    # 1. Check images
    check_image_sizes(DATA_DIR)
    
    # 2. Load data
    train_ds, val_ds, test_ds = create_datasets(DATA_DIR)
    
    # 3. Build Model
    model = build_model()
    model.summary()
    
    # 4. Callbacks (Early stopping and dynamic learning rate reduction)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    # 5. Train
    print("\n--- Starting Training ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr]
    )
    
    # 6. Evaluate & Plot
    plot_training_history(history)
    evaluate_and_report(model, test_ds)
