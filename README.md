
### Paper
Paper is here: `link comes here after publication`

Cite the following article to refer to this work.
```BibTeX
@article{wo2023,
  title = {Stoichiometric growth of SrTiO$_3$ films via {Bayesian} optimization with adaptive prior mean},
  author = {Yuki K. Wakabayashi and Takuma Otsuka and Yoshiharu Krockenberger and Hiroshi Sawada and Yoshitaka Taniyasu and Hideki Yamamoto},
  journal = {APL Machine Learning},
  volume = {},
  number = {},
  doi = {},
  year = {2023}
}
```

### How to run
Use `run_BO.py` to carry out BO and display the result of a certain method. 
Use `run_BO_all_methods.py` to run all methods for comparing respective methods. 

1. Example 1: just run

```$ python run_BO.py --num 100 --init 5 --repeat 3```
executes BO until having `100` observations with `5` initial observations for `3` times to average the results. 

For running all methods, the following line works. 
```$ python run_BO_all_methods.py --num 100 --init 5 --repeat 3```
Five methods (baseline `base`, data averagin `DA`, adaptive leveling `AL`, empirical Bayes `EB` and empirical Bayes with uniform sampling `EBu`) are carried out sequentially. 

2. Example 2: optimization of Rosenbrock function

```$ python run_BO.py --obj rb```

`--obj` option specifies the function. Choose from `ackley` (`ac` for short) and `rosenbrock` (`rb`). 

3. Example 3: run AL method

```$ python run_BO.py --mode al``` 
`--mode` option specifies the method. Choose from `base`, `DA` (data averaging), `AL` (adaptive leveling), `EB` (empirical Bayes), `EBu` (empirical Bayes with uniform sampling). 

4. Example 4: put Gaussian observation noise

```$ python run_BO.py --obsvar 0.005``` 
adds a Gaussian noise to the observation with its variance being `0.005`.

### Software version
Codes are confirmed to run with the following libraries. Likely to be compatible with newer versions. 

* `python`: `3.7.9`
* `numpy`: `1.19.2`
* `scipy`: `1.5.2`
* `sklearn`: `1.0.2`
* `matplotlib`: `3.5.0`
* `seaborn`: `0.11.1`

### Files
* `README.md`: This file. 
* `LICENSE.md`: Document of agreement for using this sample code. Read this carefully before using the code. 
* `run_BO.py`: Script to execute BO sequence with a specific method. 
* `run_BO.py`: Script to execute BO of all methods. 
* `BO_core.py`: Implements BO class. 
* `obj_func.py`: Implements objective functions. 
* `visualize.py`: Contains some functions to plot optimization results. 
* `utils.py`: Contains internal functions. 
* `lhsmdu.py`: Latin hypercube sampling package for acquisition function. Repository: https://dx.doi.org/10.5281/zenodo.2578780  Paper:http://dx.doi.org/10.1016%2Fj.jspi.2011.09.016
