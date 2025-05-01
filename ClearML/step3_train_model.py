import os
os.environ["MPLBACKEND"] = "Agg" 
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from clearml import Task, Logger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from clearml import Task
from PIL import Image

Task.set_credentials(
    web_host='https://app.clear.ml',
    api_host='https://api.clear.ml',
    files_host='https://files.clear.ml',
    key='Q9JQUW7NG3L7UIGUQ99CH6APXN7CWX',
    secret='ZDyUBTNev2TSyADi6gkFMEPRwMUBughNu4uVKUPjH7UsImaBOTLh2B6nVU2CwvYQTcw'
)

task = Task.init(project_name="VGG16", task_name="Pipeline Step 3 - Train Pneumonia Model")

task = Task.init(project_name="VGG16", task_name="Pipeline Step 3 - Train Pneumonia Model")
logger = Logger.current_logger()

args = {
    'dataset_task_id': '405caed14d034630b33cf083a9fcc28d',  
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs': 20,
}
task.connect(args)
# # Remote execution
task.execute_remotely()
# Step 3: Get Uploaded Dataset Paths
dataset_task = Task.get_task(task_id=args['dataset_task_id'])
train_dir = dataset_task.artifacts['train_dir'].get()
val_dir = dataset_task.artifacts['val_dir'].get()
class_indices = dataset_task.artifacts['class_indices'].get()

IMG_SIZE = args['img_size']
BATCH_SIZE = args['batch_size']

# Step 4: Setup ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Step 5: Build Lightweight CNN
model = models.Sequential([
    Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_indices), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Add Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True)

# Step 7: Train Model
history = model.fit(
    train_generator,
    epochs=args['epochs'],
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint]
)
# Step 8: Upload best model as Artifact
task.upload_artifact(name="best_model", artifact_object="best_model.keras")

# Step 9: Plot training curves
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
img = Image.open("training_curves.png")
logger.report_image("training_curves", "Accuracy and Loss", iteration=0, image=img)
print("✅ Model training and logging done." )