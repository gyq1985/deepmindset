from clearml import Task, Dataset

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

# 初始化 Task（这会是 pipeline 的第一步 task）
task = Task.init(project_name="VGG16", task_name="Pipeline Step 1: Dataset Loader")

# # 远程执行
# task.execute_remotely()

# 加载已经上传好的 Dataset 系统中的数据集
dataset = Dataset.get(dataset_name="DeepmindsetDataset", dataset_project="VGG16")

# 下载数据集到本地缓存
local_path = dataset.get_local_copy()

# 上传为 task artifact，让 pipeline 的后续步骤可以使用
task.upload_artifact(name='dataset', artifact_object=local_path)

print(f"Dataset path {local_path} uploaded as artifact to task!")

print('Done.')
