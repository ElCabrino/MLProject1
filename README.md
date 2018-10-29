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
| `algebra.py` | implements analytical regression models least squares and ridge regression |
| `cache.py`  | implements a cache, used to store errors according to the parameters of a model |
|  `costs.py` | implements different costs functions |
|  `features.py` | implements differents features manipulation (polynomial expansion, standardization...) |
|  `gradients.py` | implements differents regression models based on the gradient descent |
|  `helpers.py` | provided helpers file with added function to encode the model's weights |
|  `model.py` | implements higher order functions to run model with vectors of parameters |
|  `splits.py` | implements all function related to data spliting for cross validation and batch_iter |


## Folders
 
- `test/cache`: contains cache results of models with different parameters when use the sub sample mode.
- `cache`: contains cache results of models with different parameters when use the sub sample mode.
- `submissions`: contains results of prediction we did on the test dataset.

## Tutorial

Take a look at the `Tutorial.ipynb` for a detailed step-by-step tutorial.




