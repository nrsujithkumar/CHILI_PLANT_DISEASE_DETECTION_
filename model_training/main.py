import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Use MobileNetV2, a faster model compared to EfficientNetB0
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
base_model.trainable = False # Freeze the base model layers


# Build the new model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(5, activation='softmax')  # Assuming 5 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up data generators with augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Test/Validation set without augmentation (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow from directories
train_generator = train_datagen.flow_from_directory(
    '/home/sujith/Documents/Projects/CHILI_PLANT_DISEASE_DETECTION/model_training/data_set/train',  # Path to your training data
    target_size=(160, 160),  # Reduced image size
    batch_size=32,
    class_mode='categorical'
)

val_generator = test_datagen.flow_from_directory(
    '/home/sujith/Documents/Projects/CHILI_PLANT_DISEASE_DETECTION/model_training/data_set/val',  # Path to your validation data
    target_size=(160, 160),  # Reduced image size
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    '/home/sujith/Documents/Projects/CHILI_PLANT_DISEASE_DETECTION/model_training/data_set/test',  # Path to your testing data
    target_size=(160, 160),  # Reduced image size
    batch_size=32,
    class_mode='categorical'
)

# Use EarlyStopping to stop training if validation loss doesn't improve
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Reduced number of epochs for faster training
    validation_data=val_generator,
    callbacks=[early_stop],
    verbose=2
)
fine_tune_at = 100  # Fine-tune from a specific layer

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

model.save('chili_disease_model.h5')
print("MODEL SAVED TO 'chili_disease_mode.h5'")

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy*100:.2f}%")