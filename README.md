# DiffPrep

This repository contains the source code for SIGMOD 2023 paper: [DiffPrep: Differentiable Data Preprocessing Pipeline Search for Learning over Tabular Data](https://arxiv.org/pdf/2308.10915.pdf)

## Installation
Run the following command to setup a [conda](https://www.anaconda.com) environment for DiffPrep and install the required packages.

```
conda env create -f environment.yml
conda activate diffprep
```

## Run experiments
The following command will preprocess a dataset with a preprocessing method, then train and evaluate an ML model on the preprocessed data.

```
python main.py --dataset <dataset_name> --method <method_name> --model <model_name> --result_dir <result_dir>
```

### Parameters
**dataset_name**: The available dataset name can be found in the `data` folder, where each folder correpsond to one dataset. If this is not specifed, the command will run all datasets in the folder.

**method_name**: There are 4 available preprocessing methods.

- `default`: This approach uses a default pipeline that first imputes numerical missing values with the mean value of the column and categorical missing values with the most frequent value of the column. Then, it normalizes each feature using standardization. 
- `random`: This approach searches for a pipeline by training ML models with randomly sampled pipelines 20 times and selecting the one with the best validation accuracy.
- `diffprep_fix`: This is our approach with a pre-defined fixed transformation order.
- `diffprep_flex`: This is our approach with a flexible trans- formation order.

**model_name**: There are 2 available ML models.
- `log`: Logistic Regression
- `two`: two-layer NN

**result_dir**: The directory where the results will be saved. Default: `result`.

### Experiment Setup
**Hyperparameter Tuning**: The hyperparameters of the experiment are specifed in the `main.py`. By default, we tune the learning rate in our experiments using the hold-out validation set. 

**Early Stopping**: We adopt earling stopping in our training process, where we keep track of the validation accuracy in our experiments and terminate training when the validation accuracy cannot be improved significantly.

### Results
The results will be saved in the folder `<result_dir>/<method_name>/<dataset_name>`. There are two resulting files:

- `result.json`:  This file stores the test accuracy of the best epoch selected by the validation loss.

- `params.json`: This file stores the model hyperparameters.






