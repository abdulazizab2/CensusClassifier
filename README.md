# Links

- [Repository](https://github.com/abdulazizab2/CensusClassifier)
- [WandB](https://wandb.ai/abdulazizab/census-classification?workspace=user-abdulazizab) for viewing experiments and exploratory data analysis code artifacts
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

## Inference
```bash
pass
```
## Unit Tests
```bash
pass
```