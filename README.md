# Introduction
A machine learning model to predict salary range deployed in [render](https://render.com/) with a public API.
- See [model card](docs/model_card.md) to read about the model and the ethical considerations.
- See [usage](#usage) to use the features of the repository
# Links

- [Repository](https://github.com/abdulazizab2/CensusClassifier)
- [WandB](https://wandb.ai/abdulazizab/census-classification?workspace=user-abdulazizab) for viewing experiments and exploratory data analysis code artifacts

# Results
- View [summary](https://wandb.ai/abdulazizab/census-classification/runs/z9f34iok/overview?workspace=user-abdulazizab) of model metrics in WandB
- View model slices output in WandB [table](https://wandb.ai/abdulazizab/census-classification/runs/h83eqmjg?workspace=user-abdulazizab)

# Usage

## Making predictions using HTTP requests on Render
App is deployed on Render and you may try it out by using Postman or any other tool of your convenience. A sample usage is in ```live_post_request.py```
App is available at: https://census-classifier-api.onrender.com/
## Using CensusClassifier as a library

### Requirements
[WandB](wandb.ai) account, as we will log model metrics using WandB

### Installation

```bash
pip install -r requirements.txt
# Optional for downloading EDA notebook
wandb artifact get abdulazizab/census-classification/job-https___github.com_abdulazizab2_CensusClassifier.git_explore_census_dataset.ipynb:latest
dvc pull # fetches data and model
```

### Training
```bash
PYTHONPATH=. python CensusClassifier/train_model.py
```
After training a new model, you can version it by:
```bash
dvc add model/{YOUR_MODEL}
dvc add model/{YOUR_ENCODER}
```
And push it to your remote !
### Inference
1. Launch API server
```bash
uvicorn main:app
```
2. View docs at ```{IP}:{PORT}/docs``` on how to make inference using ```/predict```
### Unit Tests
```bash
PYTHONPATH=. pytest
```