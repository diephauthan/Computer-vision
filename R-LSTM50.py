import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.applications import ResNet50
# from keras.models import Model
# from keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, Input
# from keras.optimizers import Adam
# from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
# from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cv2
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
dataset_path = "/media/thandiep/New Volume/MSc_Materials/vision_research/Resnet-LSTM/MATWI-Dataset/MATWI-Dataset/MATWI/Set1/images"

# dataset_path = r"D:\MSc_Materials\vision_research\Resnet-LSTM\MATWI-Dataset\MATWI-Dataset\MATWI\Set1\images"
model_save_path = "r_lstm50_model.h5"

# Define hyperparameters (as shown in Table 5)
INITIAL_LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 80
DROPOUT_RATE = 0.5
PATIENCE = 10
SEQUENCE_LENGTH = 5  # Number of frames to process in each sequence

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3

def create_r_lstm50_model(num_classes=2):
    """
    Create the R-LSTM50 model architecture as shown in the diagram
    """
    # Input shape for a sequence of images
    input_shape = (SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    inputs = Input(shape=input_shape)
    
    # Create ResNet50 base model (without top layers)
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS),
        pooling='avg'
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Apply the base model to each time step using TimeDistributed
    encoded_frames = TimeDistributed(base_model)(inputs)
    
    # Apply LSTM layers
    lstm_1 = LSTM(256, return_sequences=True)(encoded_frames)
    lstm_2 = LSTM(256, return_sequences=True)(lstm_1)
    lstm_3 = LSTM(256, return_sequences=True)(lstm_2)
    lstm_4 = LSTM(256)(lstm_3)
    
    # Flatten and Fully Connected layers
    x = Dense(512, activation='relu')(lstm_4)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Classification layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def load_and_prepare_data(dataset_path):
    """
    Load images from the dataset path and prepare for training
    """
    print("Loading and preparing data...")
    
    # Get list of subdirectories (classes)
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Found {len(classes)} classes: {classes}")
    
    images = []
    labels = []
    
    # Load images and labels
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        print(f"Loading images from {class_path}...")
        
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(class_path, img_file)
            
            try:
                # Load and preprocess image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                img = img / 255.0  # Normalize to [0,1]
                
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Create sequences for LSTM
    # This is a simple approach - in a real scenario, you might want to create sequences
    # based on temporal relationships between images
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - SEQUENCE_LENGTH + 1):
        X_sequences.append(X[i:i+SEQUENCE_LENGTH])
        # Use the label of the last image in the sequence
        y_sequences.append(y[i+SEQUENCE_LENGTH-1])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    return X_sequences, y_sequences, classes

def data_augmentation(X_train):
    """
    Create augmented data
    """
    augmented_X = []
    
    for sequence in tqdm(X_train):
        augmented_sequence = []
        for img in sequence:
            # Random horizontal flip
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)
            
            # Random rotation
            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                img = cv2.warpAffine(img, M, (w, h))
            
            # Random brightness adjustment
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                img = np.clip(img * brightness, 0, 1)
                
            augmented_sequence.append(img)
        
        augmented_X.append(augmented_sequence)
    
    return np.array(augmented_X)

def train_model():
    """
    Train the R-LSTM50 model
    """
    # Load and prepare data
    X, y, classes = load_and_prepare_data(dataset_path)
    num_classes = len(classes)
    
    # Convert labels to one-hot encoding
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    
    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_onehot, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Perform data augmentation on training set
    print("Performing data augmentation...")
    X_train_augmented = data_augmentation(X_train)
    X_train_combined = np.concatenate([X_train, X_train_augmented])
    y_train_combined = np.concatenate([y_train, y_train])
    
    print(f"Training set after augmentation: {X_train_combined.shape}")
    
    # Create the model
    model = create_r_lstm50_model(num_classes=num_classes)
    
    # Calculate class weights to handle imbalance
    class_counts = np.sum(y_train, axis=0)
    total_samples = np.sum(class_counts)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Compile the model with weighted loss
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Define callbacks
    callbacks = [
        # Learning rate scheduler with cosine annealing
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: INITIAL_LEARNING_RATE * 0.5 * (1 + np.cos(epoch / EPOCHS * np.pi))
        ),
        # Early stopping
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        # Model checkpoint
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss')
    ]
    
    # Train the model
    print("Training the model...")
    history = model.fit(
        X_train_combined, y_train_combined,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_classes, y_pred)
    report = classification_report(y_test_classes, y_pred, target_names=classes)
    conf_matrix = confusion_matrix(y_test_classes, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return model, history

if __name__ == "__main__":
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPUs: {gpus}")
        # Set memory growth to avoid allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found. Using CPU.")
    
    # Train the model
    model, history = train_model()
    
    print("Training completed successfully!")