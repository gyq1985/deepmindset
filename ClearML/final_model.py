import os
os.environ["MPLBACKEND"] = "Agg"

import json
import torch
import logging
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from clearml import Task, Logger, Dataset
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

task = Task.init(
    project_name='VGG16-v2',
    task_name='Final VGG16 Training (from HPO)',
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False
)
logger = Logger.current_logger()

# Connect HPO result and dataset ID
args = {
    'dataset_task_id': '405caed14d034630b33cf083a9fcc28d',
    'hpo_task_id': None,
    'img_size': (224, 224),
    'batch_size': 32,
    'learning_rate_stage2': 1e-5,
    'epochs_stage2': 10,
    'freeze_until_layer': 24,
    'dropout_rate': 0.2,
    'dense1_units': 198,
    'dense2_units': 128,
    'num_classes': 4
}
args = task.connect(args)
task.execute_remotely()

# Retrieve HPO best parameters
hpo_task_id = args['hpo_task_id']
if not hpo_task_id:
    raise ValueError("Missing hpo_task_id for retrieving best hyperparameters")
hpo_task = Task.get_task(task_id=hpo_task_id)
logger.report_text(f"Using HPO Task: {hpo_task.name}")

best_params = hpo_task.get_parameter('best_parameters')
if not best_params:
    logger.report_text("Loading best_parameters from artifact...")
    artifact_path = hpo_task.artifacts['best_parameters'].get_local_copy()
    with open(artifact_path, 'r') as f:
        best_params = json.load(f)['parameters']

args['batch_size'] = int(best_params.get('batch_size', args['batch_size']))
args['learning_rate_stage2'] = float(best_params.get('learning_rate_stage2', args['learning_rate_stage2']))

logger.info(f"Using best parameters: batch_size={args['batch_size']}, learning_rate_stage2={args['learning_rate_stage2']}")

# Load dataset
dataset_task = Task.get_task(task_id=args['dataset_task_id'])
train_dir = dataset_task.artifacts['train_dir'].get()
val_dir = dataset_task.artifacts['val_dir'].get()
test_dir = dataset_task.artifacts['test_dir'].get()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.Resize(args['img_size']),
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(args['img_size'], scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_test = transforms.Compose([
    transforms.Resize(args['img_size']),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_test)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)

# Define model
model = models.vgg16_bn(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
    nn.Linear(25088, args['dense1_units']),
    nn.ReLU(),
    nn.Linear(args['dense1_units'], args['dense2_units']),
    nn.ReLU(),
    nn.Dropout(args['dropout_rate']),
    nn.Linear(args['dense2_units'], args['num_classes'])
)
model = model.to(device)

# Unfreeze selected layers
for param in model.features[args['freeze_until_layer']:].parameters():
    param.requires_grad = True

# Training setup
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['learning_rate_stage2'])
loss_fn = nn.CrossEntropyLoss()

# Training loop
logger.info("Starting final training...")
best_val_acc = 0.0
for epoch in range(args['epochs_stage2']):
    model.train()
    running_loss = 0.0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / len(train_loader.dataset)
    task.get_logger().report_scalar("train", "loss", epoch_loss, iteration=epoch)
    task.get_logger().report_scalar("train", "accuracy", epoch_acc, iteration=epoch)
    logger.info(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Validation
    model.eval()
    val_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    val_acc = val_correct / total
    task.get_logger().report_scalar("validation", "accuracy", val_acc, iteration=epoch)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_final_model.pt")

# Final test evaluation
model.load_state_dict(torch.load("best_final_model.pt"))
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig("confusion_matrix_final.png")
logger.report_image("Final Confusion Matrix", "Confusion Matrix", iteration=0, image=Image.open("confusion_matrix_final.png"))

report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
with open("classification_report_final.txt", "w") as f:
    f.write(report)
task.upload_artifact("final_model_report", "classification_report_final.txt")
task.upload_artifact("final_model_weights", "best_final_model.pt")

logger.info("Final training completed. Model and report uploaded.")
print("🎉 Final model training done!")
