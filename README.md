# Training, Architecture, and Prior for Deterministic Uncertainty Methods
This work is the repository for the ICLR 2023 workshop paper [Training, Architecture, and Prior for Deterministic Uncertainty Methods](https://arxiv.org/abs/2303.05796). 

Abstract: Accurate and efficient uncertainty estimation is crucial to build reliable Machine Learning (ML) models capable to provide calibrated uncertainty estimates, generalize and detect Out-Of-Distribution (OOD) datasets. To this end, Deterministic Uncertainty Methods (DUMs) is a promising model family capable to perform uncertainty estimation in a single forward pass. This work investigates important design choices in DUMs: (1) we show that training schemes decoupling the core architecture and the uncertainty head schemes can significantly improve uncertainty performances. (2) we demonstrate that the core architecture expressiveness is crucial for uncertainty performance and that additional architecture constraints to avoid feature collapse can deteriorate the trade-off between OOD generalization and detection. (3) Contrary to other Bayesian models, we show that the prior defined by DUMs do not have a strong effect on the final performances.

### Install
```
conda env create -n dum --file environment.yml
python setup.py develop
```

### Run
For simple but complete examples, run a notebook in `notebook/run_*.ipynb`. From these notebooks, you can change the `dataset` and the DUM's hyperparameters to run more complex tasks.

### Citation
If you use the code in this repository, consider citing our work:
```
@misc{dums-components,
  title = {Training, Architecture, and Prior for Deterministic Uncertainty Methods},
  author = {Charpentier, Bertrand and Zhang, Chenxiang and Günnemann, Stephan},
  publisher = {ICLR Workshop on Pitfalls of limited data and computation for Trustworthy ML},
  year = {2023},
}
```
