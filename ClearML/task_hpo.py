from clearml import Task, Dataset
from clearml.automation import HyperParameterOptimizer
from clearml.automation import DiscreteParameterRange
import logging
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the HPO task
task = Task.init(
    project_name='VGG16-v2',
    task_name='HPO: batch_size & lr_stage2',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# Connect arguments
args = {
    'base_train_task_id': '${step3_train_model.id}',
    'num_trials': 4,
    'time_limit_minutes': 240,
    'test_queue': 'vgg16',
    'dataset_task_id': '${step2_preprocess_data.id}',
    'batch_size': 32,
    'learning_rate_stage2': 1e-5,
    'learning_rate_stage1': 0.01,
    'epochs_stage1': 20,
    'epochs_stage2': 10,
    'dropout_rate': 0.2,
    'dense1_units': 198,
    'dense2_units': 128,
    'freeze_until_layer': 24,
    'earlystop_patience': 3,
    'num_classes': 4
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

task.execute_remotely()

# 获取 base 训练任务和数据集
BASE_TRAIN_TASK_ID = args['base_train_task_id']
dataset_id = args['dataset_task_id']
logger.info(f"Using base train task: {BASE_TRAIN_TASK_ID}")

# 定义搜索空间（离散值搜索）
hpo = HyperParameterOptimizer(
    base_task_id=BASE_TRAIN_TASK_ID,
    hyper_parameters=[
        DiscreteParameterRange('batch_size', values=[32, 64]),
        DiscreteParameterRange('learning_rate_stage2', values=[5e-6, 1e-5]),
    ],
    objective_metric_title='validation',
    objective_metric_series='accuracy',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=args['time_limit_minutes'] * 60,
    total_max_jobs=args['num_trials'],
    min_iteration_per_job=1,
    max_iteration_per_job=args['epochs_stage2'],
    execution_queue=args['test_queue'],
    save_top_k_tasks_only=2,
    parameter_override={
        'dataset_task_id': dataset_id,
        'img_size': (224, 224),
        'learning_rate_stage1': args['learning_rate_stage1'],
        'epochs_stage1': args['epochs_stage1'],
        'epochs_stage2': args['epochs_stage2'],
        'dropout_rate': args['dropout_rate'],
        'dense1_units': args['dense1_units'],
        'dense2_units': args['dense2_units'],
        'freeze_until_layer': args['freeze_until_layer'],
        'earlystop_patience': args['earlystop_patience'],
        'num_classes': args['num_classes']
    }
)

# 启动 HPO 优化任务
logger.info("Starting Hyperparameter Optimization...")
hpo.start()

# 等待任务结束
logger.info(f"Waiting for optimization to complete ({args['time_limit_minutes']} minutes)...")
time.sleep(args['time_limit_minutes'] * 60)

# 获取最优试验
try:
    top_exp = hpo.get_top_experiments(top_k=1)
    if top_exp:
        best_exp = top_exp[0]
        best_params = best_exp.get_parameters()
        metrics = best_exp.get_last_scalar_metrics()
        best_acc = metrics['validation']['accuracy'][-1] if 'validation' in metrics and 'accuracy' in metrics['validation'] else None

        logger.info("Best parameters found:")
        logger.info(best_params)
        logger.info(f"Best validation accuracy: {best_acc}")

        result = {
            'parameters': best_params,
            'accuracy': best_acc
        }
        with open('best_parameters.json', 'w') as f:
            json.dump(result, f, indent=4)

        task.upload_artifact('best_parameters', 'best_parameters.json')
        task.set_parameter('best_parameters', best_params)
        task.set_parameter('best_accuracy', best_acc)
    else:
        logger.warning("No experiments completed.")
except Exception as e:
    logger.error(f"Error retrieving best experiment: {e}")

# 关闭优化器
hpo.stop()
logger.info("HPO task completed.")
