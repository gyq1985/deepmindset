from clearml import Task
from clearml.automation import PipelineController
import os

# 设置环境变量
os.environ["CLEARML_API_ACCESS_KEY"] = os.getenv("CLEARML_API_ACCESS_KEY", "Q9JQUW7NG3L7UIGUQ99CH6APXN7CWX")
os.environ["CLEARML_API_SECRET_KEY"] = os.getenv("CLEARML_API_SECRET_KEY", "ZDyUBTNev2TSyADi6gkFMEPRwMUBughNu4uVKUPjH7UsImaBOTLh2B6nVU2CwvYQTcw")
os.environ["CLEARML_API_HOST"] = os.getenv("CLEARML_API_HOST", "https://api.clear.ml")

def run_pipeline():
    pipe = PipelineController(
        name="DeepMind",
        project="VGG16",
        version="2.0.0",
        add_pipeline_tags=True
    )

    EXECUTION_QUEUE = "vgg16"
    pipe.set_default_execution_queue(EXECUTION_QUEUE)

    # Step 1: Dataset Artifact
    pipe.add_step(
        name="step1_dataset_artifact",
        base_task_project="VGG16-v2",
        base_task_name="Pipeline Step 1: Dataset Loader"
    )

    # Step 2: Process Dataset
    pipe.add_step(
        name="step2_preprocess_data",
        parents=["step1_dataset_artifact"],
        base_task_project="VGG16-v2",
        base_task_name="Pipeline step 2 process image dataset",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/dataset_task_id": "${step1_dataset_artifact.id}"
        },
    )

    # Step 3: Train Model
    pipe.add_step(
        name="step3_train_model",
        parents=["step2_preprocess_data"],
        base_task_project="VGG16-v2",
        base_task_name="Pipeline Step 3 - Train Pneumonia Model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/dataset_task_id": "${step2_preprocess_data.id}",
            "General/train_dir": "${step2_preprocess_data.artifacts.train_dir}",
            "General/val_dir": "${step2_preprocess_data.artifacts.val_dir}",
            "General/test_dir": "${step2_preprocess_data.artifacts.test_dir}"
        }
    )

    pipe.start(queue=EXECUTION_QUEUE)
    print("✅ Pipeline started.")

if __name__ == "__main__":
    run_pipeline()
