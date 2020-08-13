# Kaggle Credit Risk Competition

## Getting Started

 * Install [pipenv](https://pipenv-fork.readthedocs.io/en/latest/) so that your Python life won't be a complete disaster.
 * Set up a kaggle account and [configure your credentials for downloading data](https://github.com/Kaggle/kaggle-api#api-credentials). You may also [install the `kaggle` command line tool](https://github.com/Kaggle/kaggle-api) if you wish but it is installed in this virtualenv.
 * Execute the `download_data.sh` script to download the archive.
 * Unzip the above data file in a parallel folder named `input`. In other words, the input folder will be in the same folder as the credit-risk folder.
 * Execute `run_jupyter.sh` to launch Jupyter.
 * Open the `baseline.ipynb` to see the baseline model or `Start Here_ A Gentle Introduction.ipynb` for a full rundown.
 * Suggested: Make a notebook under your username in this repo. Feel free to fork, PR, merge, etc.
 
## Code Organization
The following strategy will assist in fighting the Jupyter notebook dumpster fire. For the inexperienced, Jupyter notebooks are very hard to collaborate on and don't diff well for git commits. Therefore, you don't want to collaborate directly with them.

Here's what does work:

 * Create two sets of files:
   * Your notebooks. These will be unshared with others (unless they wish to read them.)
   * Library code. These will be Python files, currently located in the models package in this project. Being plain old source files, they are easy to modify, merge, and collaborate on.
 * Experiment in your notebook as desired and only run other people's notebooks.
 * Any code you find useful and/or shareable, lift from your notebook into a Python file in your models package.
 * Import the shared code in your notebook like so:
 
```python
import models.baselinelgb as lgb

submission, fi, metrics = lgb.model(app_train, app_test)
print('Baseline metrics')
print(metrics)
```
