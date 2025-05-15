from clearml import Task
from clearml.automation import PipelineController
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Queue configuration - using same queue for everything
EXECUTION_QUEUE = "vgg16"

def run_pipeline():
    # Connecting ClearML with the current pipeline
    pipe = PipelineController(
        name="DeepMind", 
        project="VGG16", 
        version="3.0.0", 
        add_pipeline_tags=True
    )

    # Set default queue for pipeline control
    pipe.set_default_execution_queue(EXECUTION_QUEUE)
    logger.info(f"Set default execution queue to: {EXECUTION_QUEUE}")

    # Step 1: Dataset Artifact
    pipe.add_step(
        name="stage_data",
        base_task_project="VGG16-v2",
        base_task_name="Pipeline Step 1: Dataset Loader",
        execution_queue=EXECUTION_QUEUE
    )

    # Step 2: Preprocess Dataset
    pipe.add_step(
        name="stage_process",
        parents=["stage_data"],
        base_task_project="VGG16-v2",
        base_task_name="Pipeline step 2 process image dataset",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/dataset_task_id": "${stage_data.id}"
        }
    )

    # Step 3: Initial Training Task
    pipe.add_step(
        name="stage_train",
        parents=["stage_process"],
        base_task_project="VGG16-v2",
        base_task_name="Pipeline Step 3 - Train Pneumonia Model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/dataset_task_id": "${stage_process.id}"
        }
    )

    # Step 4: Hyperparameter Optimization
    pipe.add_step(
        name="stage_hpo",
        parents=["stage_train", "stage_process"],
        base_task_project="VGG16-v2",
        base_task_name="HPO: batch_size & lr_stage2",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_dataset_id": "97d3fcafa2cd491d9da5298e89b35bf8",
            "General/test_queue": EXECUTION_QUEUE,
            "General/num_trials": 4,
            "General/time_limit_minutes": 60,
            "General/run_as_service": False,
            "General/dataset_task_id": "${stage_data.id}",
            "General/base_train_task_id": "${stage_train.id}"
        }
    )

    # Step 5: Final Model Training
    pipe.add_step(
        name="stage_final_model",
        parents=["stage_hpo", "stage_process"],
        base_task_project="VGG16-v2",
        base_task_name="Final VGG16 Training (from HPO)",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_dataset_id": "${stage_process.parameters.General/processed_dataset_id}",
            "General/hpo_task_id": "${stage_hpo.id}",
            "General/test_queue": EXECUTION_QUEUE
        }
    )

    # Start pipeline locally
    logger.info("Starting pipeline locally with tasks on queue: %s", EXECUTION_QUEUE)
    pipe.start_locally()
    logger.info("Pipeline started successfully")

if __name__ == "__main__":
    run_pipeline()
