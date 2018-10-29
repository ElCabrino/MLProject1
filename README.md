# Higgs Boson Machine Learning Project

## Participants

- Julien Perrenoud: julien.perrenoud@epfl.ch
- Loïc Serafin: loic.serafin@epfl.ch
- Vincent Cabrini: vincent.cabrini@epfl.ch

## Results

Currently **31th** with **83.237%** of correct predictions.

## Code

| File | Description |
|:---:|---|
| `cache.py`  | implements a cache, used to store errors according to the parameters of a model |
|  `costs.py` | implements different costs functions |
| `evaluate.py` | implements functions to find the best parameters for a model |
|  `features.py` | implements differents features manipulation (polynomial expansion, standardization...) |
|  `gradients.py` | implements differents regression models based on the gradient descent |
|  `helpers.py` | provided helpers file with added function to encode the model's weights |
| `implementations.py` | implements required functions and some others functions to create analytical and descent models |
| `predict.py` | implements functions to predict labels (0 or 1) with given weights|
|  `run.py` | required script of our best submission |
| `validate.py`| implements functions to use cross-validation over a model |

## Folders
 
- `test/cache`: contains cache results of models with different parameters when use the sub sample mode.
- `cache`: contains cache results of models with different parameters when use the sub sample mode.
- `submissions`: contains results of prediction we did on the test dataset.

## Tutorial

Take a look at the `Tutorial.ipynb` for a detailed step-by-step tutorial.




