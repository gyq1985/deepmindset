from torchvision.datasets import ImageFolder
from clearml import Task
from clearml import Dataset
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
    'dataset_task_id': '70d3e1b45c89407c9f19ffdb0b4e35b8'
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

# 上传三个路径信息（作为 downstream pipeline 的输入）
task.upload_artifact('train_dir', train_dir)
task.upload_artifact('val_dir', val_dir)
task.upload_artifact('test_dir', test_dir)

# 提取并上传类索引
dummy_transform = lambda x: x  # 无转换加载图像类名
train_dataset = ImageFolder(train_dir, transform=None)
task.upload_artifact('class_indices', train_dataset.class_to_idx)

task.set_parameter("General/processed_dataset_id", task.id)

print("Artifacts uploaded and processed_dataset_id set!")
print("Done.")
