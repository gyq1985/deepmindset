import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from PIL import Image
from clearml import Task, Logger
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

Task.set_credentials(
    web_host='https://app.clear.ml',
    api_host='https://api.clear.ml',
    files_host='https://files.clear.ml',
    key='Q9JQUW7NG3L7UIGUQ99CH6APXN7CWX',
    secret='ZDyUBTNev2TSyADi6gkFMEPRwMUBughNu4uVKUPjH7UsImaBOTLh2B6nVU2CwvYQTcw'
)

task = Task.init(project_name="VGG16-v2", task_name="Pipeline Step 3 - Train Pneumonia Model")
logger = Logger.current_logger()

args = {
    'dataset_task_id': '93ffb2ec41694fa893c24775a7503cfb',
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs_stage1': 20,
    'epochs_stage2': 10,
    'learning_rate_stage1': 0.01,
    'learning_rate_stage2': 1e-5,
    'dropout_rate': 0.2,
    'dense1_units': 198,
    'dense2_units': 128,
    'freeze_until_layer': -4,
    'earlystop_patience': 3,
    'num_classes': 4
}
task.connect(args)
task.execute_remotely()

# Load dataset directories
dataset_task = Task.get_task(task_id=args['dataset_task_id'])
train_dir = dataset_task.artifacts['train_dir'].get()
val_dir = dataset_task.artifacts['val_dir'].get()
test_dir = dataset_task.artifacts['test_dir'].get()
class_indices = dataset_task.artifacts['class_indices'].get()

IMG_SIZE = args['img_size']
BATCH_SIZE = args['batch_size']

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
val_test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load VGG16
model = models.vgg16(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(25088, args['dense1_units']),
    nn.ReLU(),
    nn.Linear(args['dense1_units'], args['dense2_units']),
    nn.ReLU(),
    nn.Dropout(args['dropout_rate']),
    nn.Linear(args['dense2_units'], args['num_classes']),
    nn.Sigmoid()
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args['learning_rate_stage1'])

# Stage 1 Training
early_stopping_counter = 0
best_val_loss = float('inf')

for epoch in range(args['epochs_stage1']):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = nn.functional.one_hot(labels, num_classes=args['num_classes']).float()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            labels = nn.functional.one_hot(labels, num_classes=args['num_classes']).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model_stage1.pth")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= args['earlystop_patience']:
            break

# Stage 2 Fine-tuning
for param in list(model.features.parameters())[args['freeze_until_layer']:]:
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=args['learning_rate_stage2'])
early_stopping_counter = 0
best_val_loss = float('inf')

for epoch in range(args['epochs_stage2']):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = nn.functional.one_hot(labels, num_classes=args['num_classes']).float()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            labels = nn.functional.one_hot(labels, num_classes=args['num_classes']).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model_stage2.pth")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= args['earlystop_patience']:
            break

task.upload_artifact("best_model_stage2", "best_model_stage2.pth")

# Evaluation
model.load_state_dict(torch.load("best_model_stage2.pth"))
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
class_labels = list(class_indices.keys())

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig("confusion_matrix_vgg16.png")
logger.report_image("confusion_matrix_vgg16", "Confusion Matrix", iteration=0, image=Image.open("confusion_matrix_vgg16.png"))

report = classification_report(all_labels, all_preds, target_names=class_labels)
with open("classification_report_vgg16.txt", "w") as f:
    f.write(report)
task.upload_artifact("classification_report_vgg16", artifact_object="classification_report_vgg16.txt")
