from clearml import Task
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# 连接 ClearML
Task.set_credentials(
    web_host='https://app.clear.ml',
    api_host='https://api.clear.ml',
    files_host='https://files.clear.ml',
    key='Q9JQUW7NG3L7UIGUQ99CH6APXN7CWX',
    secret='ZDyUBTNev2TSyADi6gkFMEPRwMUBughNu4uVKUPjH7UsImaBOTLh2B6nVU2CwvYQTcw'
)

# 初始化任务
task = Task.init(project_name="VGG16-v2", task_name="Pipeline step 2 process image dataset")
args = {
    'dataset_task_id': '70d3e1b45c89407c9f19ffdb0b4e35b8',
    'img_size': (224, 224),
    'batch_size': 32,
}
task.connect(args)
task.execute_remotely()

# 获取数据路径
if args['dataset_task_id']:
    dataset_upload_task = Task.get_task(task_id=args['dataset_task_id'])
    print(f"Input task id={args['dataset_task_id']} artifacts: {list(dataset_upload_task.artifacts.keys())}")
    data_path = dataset_upload_task.artifacts['dataset'].get_local_copy()
else:
    raise ValueError("Missing dataset link")

print(f'Dataset local path: {data_path}')

# 数据集结构
train_dir = os.path.join(data_path, "train")
val_dir = os.path.join(data_path, "val")
test_dir = os.path.join(data_path, "test")

# 设置图像变换
IMG_SIZE = args['img_size']
BATCH_SIZE = args['batch_size']

# 保持一致的数据增强
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),  # Resize 类似 target_size
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 平移
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),  # 缩放
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # 等价于 rescale=1./255 的中心化
])

val_test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# 加载图像数据集
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

# 构建 DataLoader（等价于 flow_from_directory）
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 上传 class indices（等价于 train_generator.class_indices）
task.upload_artifact('class_indices', train_dataset.class_to_idx)

# 上传三个路径信息（作为 downstream pipeline 的输入）
task.upload_artifact('train_dir', train_dir)
task.upload_artifact('val_dir', val_dir)
task.upload_artifact('test_dir', test_dir)

print("Artifacts uploaded!")
print("Done.")
