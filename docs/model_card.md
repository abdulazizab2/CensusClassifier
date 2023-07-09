# Model Card

## Model Details
A binary classification model to predict wether a person icon is over 50K per year or not. Random Forest classifer from sklearn is used with default parameters except:
- random_state=: 42
## Intended Use
The model can be used to predict a salary range based on many attributes
## Training Data
[Census Dataset](https://archive.ics.uci.edu/ml/datasets/census+income) obtained from the link. It has many attributes with categorical (ordinal and non-ordinal) and numerical features. Raw data is messy as it contains many leading and trailing whitespaces in column names and values. It was cleaned using an EDA notebook which can be retrieved using [WandB](wandb.ai)
```bash
wandb artifact get abdulazizab/census-classification/job-https___github.com_abdulazizab2_CensusClassifier.git_explore_census_dataset.ipynb:v0
```

An 80/20 train/test split was used without stratification. sklearn OneHotEncoder was applied on all categorical features.
## Evaluation Data
The test split was used for evaluation. The categorical encoder fitted on the train split is used to transform the categorical features of the test set. The encoder can be obtained from the remote DVC:
```bash
dvc pull
```
## Metrics
The metrics for evaluations for the model are precision, recall, fbeta and a  confusion matrix

On the test set:
- precision: 0.76
- recall: 0.63
- fbeta: 0.69
- [Confusion Matrix](https://wandb.ai/abdulazizab/census-classification/runs/z9f34iok/workspace?workspace=user-abdulazizab) from WandB by viewing Custom Charts section

Metrics slices are computed for the data using all of the categorical features reporting:
- samples of the slice
- precision of the slice
- recall of the slice
- fbeta of the slice

The slices are formatted as a [table](https://wandb.ai/abdulazizab/census-classification/runs/h83eqmjg?workspace=user-abdulazizab) logged in WandB. View it using the link provided
**Note**: You might need to click ```Reset Table``` in the bottom right of the table if the table looks empty

## Ethical Considerations
The dataset contains features about race and gender which has the potential to be biased towards a gender or some races. It is critical to study metrics on slices of such features to ensure model fairness.
## Caveats and Recommendations
The data was collected in 1994 leading to features/attributes in the data that may had a significant drift. However, it can be used for a proof of concept work of machine learning experimentation. Also, the dataset is imbalanced.
