# Links

- [Repository](https://github.com/abdulazizab2/CensusClassifier)
- [WandB](https://wandb.ai/abdulazizab/census-classification?workspace=user-abdulazizab) for viewing experiments and exploratory data analysis code artifacts

# Results
- View [summary](https://wandb.ai/abdulazizab/census-classification/runs/z9f34iok/overview?workspace=user-abdulazizab) of model metrics in WandB
- View model slices output in WandB [table](https://wandb.ai/abdulazizab/census-classification/runs/h83eqmjg?workspace=user-abdulazizab)
# Requirements
[WandB](wandb.ai) account, as we will log model metrics using WandB

# Installation

```bash
pip install -r requirements.txt
# Optional for downloading EDA notebook
wandb artifact get abdulazizab/census-classification/job-https___github.com_abdulazizab2_CensusClassifier.git_explore_census_dataset.ipynb:v0
```

# Usage
```bash
dvc pull # fetched data and model
```

## Training
```bash
PYTHONPATH=. python CensusClassifier/train_model.py
```
After training a new model, you can version it by:
```bash
dvc add model/{YOUR_MODEL}
```
And push it to your remote !
## Inference
```bash
pass
```
## Unit Tests
```bash
pass
```