# [Kaggle Credit Risk Competition](https://www.kaggle.com/c/home-credit-default-risk/)
A bunch of files, notebooks, projects, etc. for the [Kaggle Credit Risk Competition](https://www.kaggle.com/c/home-credit-default-risk/).

## Competitors
Feel free to steal our ideas and work. You're still going down.

## Getting Started

 * Install [pipenv](https://pipenv-fork.readthedocs.io/en/latest/) so that your Python life won't be a complete disaster.
 * Set up a kaggle account and [configure your credentials for downloading data](https://github.com/Kaggle/kaggle-api#api-credentials). You may also [install the `kaggle` command line tool](https://github.com/Kaggle/kaggle-api) if you wish but it is installed in this virtualenv.
 * Execute the `download_data.sh` script to download the archive.
 * Unzip the above data file in a parallel folder named `input`. In other words, the input folder will be in the same folder as the credit-risk folder. The data files will be in the input folder in a folder named `home-credit-default-risk`.
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

## Notes, Tips, Etc.

 * [This is a video](https://www.youtube.com/watch?v=7665INW4I5g&feature=youtu.be) from the competition winners.
 * Most winners are NOT using Neural Networks. They still use ensemble classifiers like:
   * [lightgbm.LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)
   * [XGBoost](https://xgboost.readthedocs.io/en/latest/)
 * The winners suggest using a combination of Stacking and Voting [ensembles](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble):
   * [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
   * [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
 * We should check out auto [Hyperparameter Optimizers](https://neptune.ai/blog/scikit-optimize):
   * [forest_minimize](https://scikit-optimize.github.io/stable/modules/generated/skopt.forest_minimize.html)
   * [gp_minimize](https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html)
