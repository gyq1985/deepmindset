import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
from clearml import Task, Logger
from utils import load_transformed_datasets

# ClearML Task init
Task.set_credentials(
    web_host='https://app.clear.ml',
    api_host='https://api.clear.ml',
    files_host='https://files.clear.ml',
    key='Q9JQUW7NG3L7UIGUQ99CH6APXN7CWX',
    secret='ZDyUBTNev2TSyADi6gkFMEPRwMUBughNu4uVKUPjH7UsImaBOTLh2B6nVU2CwvYQTcw'
)

task = Task.init(project_name="VGG16-v2", task_name="Pipeline Step 3 - Train Pneumonia Model")
logger = Logger.current_logger()

# Hyperparameters
args = {
    'dataset_task_id': '405caed14d034630b33cf083a9fcc28d',
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs_stage1': 20,
    'epochs_stage2': 10,
    'learning_rate_stage1': 0.01,
    'learning_rate_stage2': 1e-5,
    'dropout_rate': 0.2,
    'dense1_units': 198,
    'dense2_units': 128,
    'freeze_until_layer': 24,
    'earlystop_patience': 3,
    'num_classes': 4
}
task.connect(args)
task.execute_remotely()

# Get dataset artifact paths
dataset_task = Task.get_task(task_id=args['dataset_task_id'])
train_dir = dataset_task.artifacts['train_dir'].get()
val_dir = dataset_task.artifacts['val_dir'].get()
test_dir = dataset_task.artifacts['test_dir'].get()
class_indices = dataset_task.artifacts['class_indices'].get()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:",torch.cuda.is_available())
# Data preprocessing
train_loader, val_loader, test_loader, _, train_dataset = load_transformed_datasets(
    train_dir, val_dir, test_dir,
    img_size=args['img_size'],
    batch_size=args['batch_size']
)
# Model definition
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
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args['learning_rate_stage1'])

# Helper function for training
def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs, earlystop_patience):
    best_val_acc = 0.0
    best_model_wts = None
    patience = 0
    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct = 0.0, 0
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
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")
        task.get_logger().report_scalar("train", "accuracy", epoch_acc, iteration=epoch)
        task.get_logger().report_scalar("train", "loss", epoch_loss, iteration=epoch)
        task.get_logger().report_scalar("validation", "accuracy", val_acc, iteration=epoch)
        task.get_logger().report_scalar("validation", "accuracy", val_acc, iteration=0)
        task.get_logger().report_scalar("validation", "loss", val_loss, iteration=epoch)
        print(f"[DEBUG] Reported val_acc={val_acc:.4f} to ClearML at iteration=0 & TRAIN,VAL")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= earlystop_patience:
                print("Early stopping")
                break

    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history

# Stage 1 Training
model, acc1, val_acc1, loss1, val_loss1 = train_model(
    model, train_loader, val_loader, optimizer, loss_fn,
    args['epochs_stage1'], args['earlystop_patience']
)

# Stage 2 Fine-tuning
for param in model.features[args['freeze_until_layer']:].parameters():
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['learning_rate_stage2'])
model, acc2, val_acc2, loss2, val_loss2 = train_model(
    model, train_loader, val_loader, optimizer, loss_fn,
    args['epochs_stage2'], args['earlystop_patience']
)

# Save best model
torch.save(model.state_dict(), "best_model_stage2.pt")
task.upload_artifact("best_model_stage2", artifact_object="best_model_stage2.pt")

# Plot accuracy and loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(acc1 + acc2, label="Train Acc")
plt.plot(val_acc1 + val_acc2, label="Val Acc")
plt.title("Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss1 + loss2, label="Train Loss")
plt.plot(val_loss1 + val_loss2, label="Val Loss")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_curves_vgg16_pytorch.png")
logger.report_image("training_curves_vgg16_pytorch", "Training Curves", iteration=0, image=Image.open("training_curves_vgg16_pytorch.png"))

# Evaluation on test set
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
task.get_logger().report_scalar("validation", "accuracy", test_acc, iteration=0)
print(f"[DEBUG] Reported test_acc={test_acc:.4f} to ClearML as objective metric")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig("confusion_matrix_vgg16_pytorch.png")
logger.report_image("confusion_matrix_vgg16_pytorch", "Confusion Matrix", iteration=0, image=Image.open("confusion_matrix_vgg16_pytorch.png"))

report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
with open("classification_report_vgg16_pytorch.txt", "w") as f:
    f.write(report)
task.upload_artifact("classification_report_vgg16_pytorch", artifact_object="classification_report_vgg16_pytorch.txt")

