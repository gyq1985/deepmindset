from clearml import Task
from clearml.automation import PipelineController
import os

# 设置环境变量
os.environ["CLEARML_API_ACCESS_KEY"] = os.getenv("CLEARML_API_ACCESS_KEY", "Q9JQUW7NG3L7UIGUQ99CH6APXN7CWX")
os.environ["CLEARML_API_SECRET_KEY"] = os.getenv("CLEARML_API_SECRET_KEY", "ZDyUBTNev2TSyADi6gkFMEPRwMUBughNu4uVKUPjH7UsImaBOTLh2B6nVU2CwvYQTcw")
os.environ["CLEARML_API_HOST"] = os.getenv("CLEARML_API_HOST", "https://api.clear.ml")

# 可选：回调函数
def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    print(f"Cloning Task id={a_node.base_task_id} with parameters: {current_param_override}")
    return True

def post_execute_callback_example(a_pipeline, a_node):
    print(f"Completed Task id={a_node.executed}")
    return

def run_pipeline():
    # 初始化 pipeline
    pipe = PipelineController(
        name="DeepMind", 
        project="VGG16",
        version="2.0.0",
        add_pipeline_tags=True
    )

    pipe.set_default_execution_queue("vgg16")  # 替换成你的 ClearML queue 名

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
        parameter_override={
            "General/dataset_task_id": "${step2_preprocess_data.id}"
        }
    )


    # 启动 pipeline
    pipe.start(queue="vgg16")  # 用 pipeline 控制器队列
    print("✅ Pipeline started.")

if __name__ == "__main__":
    run_pipeline()
