# deepmindset    

This repository presents a deep learning-based medical imaging project for pneumonia detection from chest X-ray images. The project was developed through three Agile Sprints, resulting in two model versions and a complete end-to-end pipeline integrated with ClearML and a Streamlit-based GUI.

## Project Demo
- Project Demo Link:
[Demo Link](https://deepmindset.streamlit.app/)
- Project Demo Presentation Slide:
[Presentation Slide](https://docs.google.com/presentation/d/1rdrTdre-7CA-u4UVo7l_XfYYtmS4l56R/edit?usp=drive_link&ouid=105613076003132628943&rtpof=true&sd=true)

##  Project Overview
We developed two models over three development Sprints:

- v1.0 – DNN Model (Sprint 1):
A basic deep neural network built as the initial prototype.

- v2.0 – Fine-Tuned VGG16 Model (Sprint 2 & 3):
We adopted transfer learning with a pretrained VGG16 model and applied Hyperparameter Optimization (HPO) for further improvement.

We used ClearML for pipeline orchestration and Streamlit Cloud to deploy a user-friendly web interface.

##  Repository Structure
```sql
.
├── DataPreparation/
│   └── Scripts for dataset downloading, splitting, and preprocessing.
│
├── v1/
│   └── DNN model (v1.0) and its training pipeline developed in Sprint 1.
│       Contains modified files after v2.0 branched from v1.0.
│
├── ClearML/
│   ├── step1_data_upload.py
│   ├── step2_preprocessing.py
│   ├── step3_train_model.py
│   ├── task_hpo.py
│   ├── final_model.py
│   ├── pipeline_from_tasks.py                ← Final Sprint 3 pipeline (with HPO)
│   ├── pipeline_from_tasks_non_HPO.py        ← Sprint 2 pipeline (without HPO)
│   ├── dataset/                              ← Dataset used for ClearML artifact uploads
│   ├── utils/                                ← Utility scripts
│   └── (tensorflow)ignore it/                ← Earlier TensorFlow implementation (deprecated)
│
├── GUI Interface/
│   └── Streamlit GUI code for model deployment and user interaction.

```

> The GUI Interface should contain the compressed package of the best model. However, the compressed package of the model is too large to be uploaded to github. Download address of the compressed package of the best model:
https://files.clear.ml/VGG16/.pipelines/DeepMind/stage_final_model.f80dba5f8f104bd9830dd87303c7652e/artifacts/final_model_weights/best_final_model.pt

## Key Feature
- Agile-based Iterative Development (3 Sprints)

- Two Model Versions:

- - Baseline DNN

- - Fine-tuned VGG16 with HPO

- ClearML Pipeline:
Fully automated pipeline including data loading, preprocessing, training, HPO, and evaluation.

- Streamlit GUI:
A simple web interface for clinicians to upload X-ray images and view predictions.

- Transfer from TensorFlow to PyTorch:
Final implementation rebuilt in PyTorch for improved flexibility and integration.