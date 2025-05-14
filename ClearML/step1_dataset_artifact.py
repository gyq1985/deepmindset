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

# Initialize the Task (this will be the first step task of the pipeline)
task = Task.init(project_name="VGG16-v2", task_name="Pipeline Step 1: Dataset Loader")

# # Remote execution
task.execute_remotely()

# Load the Dataset in the dataset system that has been uploaded
dataset = Dataset.get(dataset_name="DeepmindsetDataset", dataset_project="VGG16")

# Download the dataset to the local cache
local_path = dataset.get_local_copy()

# Upload it as a task artifact to enable the subsequent steps of the pipeline to use it
task.upload_artifact(name='dataset', artifact_object=local_path)

print(f"Dataset path {local_path} uploaded as artifact to task!")

print('Done.')
