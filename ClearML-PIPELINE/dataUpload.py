from clearml import Task, Dataset

data_path = "ClearML\dataset\split_dataset\split_dataset"

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

# dataset 上传数据，dataset可复用
dataset = Dataset.create(dataset_name="DeepmindsetDataset", dataset_project="VGG16")
dataset.add_files(path=data_path)
dataset.upload()
dataset.finalize()

#dataset = Dataset.get(dataset_name="MyDataset", dataset_project="MyProject")
# local_path = dataset.get_local_copy()

print("Dataset folder uploaded to ClearML Task!")
