from clearml import Task
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import os
import shutil

# ClearML 连接
Task.set_credentials(
    web_host='https://app.clear.ml',
    api_host='https://api.clear.ml',
    files_host='https://files.clear.ml',
    key='Q9JQUW7NG3L7UIGUQ99CH6APXN7CWX',
    secret='ZDyUBTNev2TSyADi6gkFMEPRwMUBughNu4uVKUPjH7UsImaBOTLh2B6nVU2CwvYQTcw'
)


task = Task.init(project_name="VGG16-v2", task_name="Pipeline step 2 process image dataset")
args = {
    'dataset_task_id': '70d3e1b45c89407c9f19ffdb0b4e35b8',
    'img_size': (224, 224),
    'num_augmented_per_image': 1  
}
args = task.connect(args)
task.execute_remotely()


dataset_task = Task.get_task(task_id=args['dataset_task_id'])
data_path = dataset_task.artifacts['dataset'].get_local_copy()
train_dir = os.path.join(data_path, "train")
val_dir = os.path.join(data_path, "val")
test_dir = os.path.join(data_path, "test")


IMG_SIZE = args['img_size']
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

aug_train_dir = os.path.join(data_path, "aug_train")
os.makedirs(aug_train_dir, exist_ok=True)

raw_dataset = datasets.ImageFolder(root=train_dir)

for class_idx, class_name in enumerate(raw_dataset.classes):
    class_input_dir = os.path.join(train_dir, class_name)
    class_output_dir = os.path.join(aug_train_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    for idx, (img_path, _) in enumerate(raw_dataset.samples):
        img = Image.open(img_path).convert("RGB")
        for aug_id in range(args['num_augmented_per_image']):
            transformed_img = transform(img)
            save_path = os.path.join(class_output_dir, f"{idx}_{aug_id}.png")
            transforms.ToPILImage()(transformed_img).save(save_path)

task.upload_artifact('processed_train_dir', aug_train_dir)
task.upload_artifact('val_dir', val_dir)
task.upload_artifact('test_dir', test_dir)
task.upload_artifact('class_indices', raw_dataset.class_to_idx)

print("✅ Step 2 over")
