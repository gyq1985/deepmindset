from clearml import Task
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
task.set_base_docker("tensorflow/tensorflow:2.18.0-gpu")
# args
args = {
    'dataset_task_id': '70d3e1b45c89407c9f19ffdb0b4e35b8',  # It is filled through pipeline at runtime
    'img_size': (224, 224),
    'batch_size': 32,
}

task.connect(args)
print('Arguments: {}'.format(args))

# Remote execution
# task.execute_remotely()

# Obtain the dataset artifact uploaded in the first step of the pipeline
if args['dataset_task_id']:
    dataset_upload_task = Task.get_task(task_id=args['dataset_task_id'])
    print('Input task id={} artifacts {}'.format(args['dataset_task_id'], list(dataset_upload_task.artifacts.keys())))
    data_path = dataset_upload_task.artifacts['dataset'].get_local_copy()
else:
    raise ValueError("Missing dataset link")

print(f'Dataset local path: {data_path}')

# Dataset menu structure
train_dir = f"{data_path}/train"
val_dir = f"{data_path}/val"
test_dir = f"{data_path}/test"

#  ImageDataGenerator
IMG_SIZE = args['img_size']
BATCH_SIZE = args['batch_size']

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Uploading artifacts
print('Uploading processed dataset generators (train, val, test)')

# It can not upload the generator directly! It should be saved as a file or an intermediate variable.
# Upload the mapping of class indices for use in subsequent steps
task.upload_artifact('class_indices', train_generator.class_indices)

# Upload folder path (the next task reads the file directly)
task.upload_artifact('train_dir', train_dir)
task.upload_artifact('val_dir', val_dir)
task.upload_artifact('test_dir', test_dir)

print('Artifacts uploaded!')
print('Done.')