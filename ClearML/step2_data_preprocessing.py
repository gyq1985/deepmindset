from clearml import Task
import os
from torchvision import datasets, transforms

web_server = 'https://app.clear.ml'
api_server = 'https://api.clear.ml'
files_server = 'https://files.clear.ml'
access_key = 'Q9JQUW7NG3L7UIGUQ99CH6APXN7CWX'
secret_key = 'ZDyUBTNev2TSyADi6gkFMEPRwMUBughNu4uVKUPjH7UsImaBOTLh2B6nVU2CwvYQTcw'

Task.set_credentials(web_host=web_server,
                    api_host=api_server,
                    files_host=files_server,
                    key=access_key,
                    secret=secret_key
                    )   

# Initialization Task
task = Task.init(project_name="VGG16-v2", task_name="Pipeline step 2 process image dataset")

# args
args = {
    'dataset_task_id': 'cbec31a282d84bdabacf8e44190bf1c4',
    'img_size': (224, 224),
    'batch_size': 32,
}

task.connect(args)
print('Arguments: {}'.format(args))

# Remote execution
task.execute_remotely()


# Obtain the dataset artifact uploaded in the first step of the pipeline
if args['dataset_task_id']:
    dataset_upload_task = Task.get_task(task_id=args['dataset_task_id'])
    print('Input task id={} artifacts {}'.format(args['dataset_task_id'], list(dataset_upload_task.artifacts.keys())))
    data_path = dataset_upload_task.artifacts['dataset'].get_local_copy()
else:
    raise ValueError("Missing dataset link")

print(f'Dataset local path: {data_path}')

# Dataset folder structure
train_dir = os.path.join(data_path, "train")
val_dir = os.path.join(data_path, "val")
test_dir = os.path.join(data_path, "test")

# Image transforms
IMG_SIZE = args['img_size']
BATCH_SIZE = args['batch_size']

train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_test_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

# Save class indices for later use
class_indices = train_dataset.class_to_idx

# Uploading artifacts
print('Uploading processed dataset metadata')
task.upload_artifact('class_indices', class_indices)
task.upload_artifact('train_dir', train_dir)
task.upload_artifact('val_dir', val_dir)
task.upload_artifact('test_dir', test_dir)

print('Artifacts uploaded!')
print('Done.')
