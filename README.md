# Eigen component analysis (ECA) introduction

This is the repository for paper *Eigen component analysis: A quantum theory incorporated machine learning technique to find linearly maximum separable components.* It includes two main parts for the experiments, Eigen component analysis (ECA) and eigen component analysis network (ECAN). Either ECA or ECAN can be trained with vanilla eigen copomnent analysis (VECA) or approximated eigen copomnent analysis (AECA). As the article mentioned, VECA often result in a sparse result and better option for dimension reduction. 

ECA, in my humble opinion which I cannot say in the article,  is a top-ranking feature extraction or dimension reduction algorithm. The obtained eigenfeature matrix (EFM) and eigenfeature-class mapping matrix (ECMM) could be used to conduct the concrete dimension reduction. The concrete number of dimension reduction is near to the rank of whole data set if without background or noise. In addition, the variance ratio is close to 1 with this concrete dimension reduction. **For example, the concrete number of dimension reduction  for MNIST data set is 110 using VECA (or 328 using AECA) neither more or less. This could mean that the MNIST data set or the background/noise-free data set  only occupy a subspace with dimension 110. The difference between the result of VECA and AECA is that VECA ignored some less important information.**  In ECAN, with the dimension operator, the nonlinear dimension reduction outperforms many classical algorithms which will be reported in our future work.  

[//]: # "I will upload the enviroment requirements later. I know the code is kind of messy, since I created many branches in this project and this repository is just one branch I chosen. I will merge the code and  add some comments to help you understand this project. "

- [Eigen component analysis (ECA) introduction](#eigen-component-analysis--eca--introduction)
- [Quick start](#quick-start)
  * [Directory structure](#directory-structure)
  * [Train VECA](#train-veca)
  * [Train AECA](#train-aeca)
  * [Train the 2-fold ECAN with AECA on MNIST data set](#train-the-2-fold-ecan-with-aeca-on-mnist-data-set)
  * [Train the 2-fold ECAN with VECA on MNIST data set](#train-the-2-fold-ecan-with-veca-on-mnist-data-set)
- [Dimension reduction](#dimension-reduction)
  * [ECA](#eca)
  * [2-fold ECAN](#2-fold-ecan)

# Quick start

## Directory structure

The core algorithm is implemented in *real_eigen.py* (*complex_eigen.py* is independently implemented). The base model for training is named with prefix *base*, which could be run independently or work as a module. And ECAN related file are suffixed with *network*. Data loading is implemented in *load_data.py*. The comparison with other models is implemented in *other_models.py*. The obtained EFM, ECMM, RaDO or ReDO are stored in directory *history.* 

+   Analytic ECA: analytic_eca.py, which can find an analytic solution for full rank data sets
+   Approximated ECA: base_approx.py
+   Complex ECA: complex_eigen.py, base_complex_eigen.py

All the *data_tag* in analytic ECA and base model could be changed to train other data sets. The history and checkpoints are managed by MAGIC_CODE in *real_eigen.py/complex_eigen.py* and WORK_MAGIC_CODE in each to-be-executed file.

## Train VECA

+   The files include twodim.py  (```data_tag="2d"```), threedim.py (```data_tag="3d"```), bc.py (```data_tag="breast_cancer"```), wis.py (```data_tag="wis"```), mnist.py (```data_tag="mnist"```) correspoinding to 2D, 3D, Wis1992, Wis1995, MNIST data set mentioned in the article. 

+   Set the *to_train* option to be **True** otherwise it will just test on previous saved model.

+   Then training on Wis1992 should be 

    ```bash
    python bc.py
    ```

## Train AECA

[//]: # "I will upload this part of code later. "

-   One could change the data_tag to test other data sets.

    ```
    python base_approx.py
    ```

    



## Train the 2-fold ECAN with AECA on MNIST data set

+   The code for ECAN is in base_network.py. One could change the data_tag to these one mentioned in load_data.py

+   Set the dimension operator to be used

    ```python
    # dimension operator, set quadratic false to use ReLU and neural network (not fully connected)
    model = build_model_do(
        state_len, num_classes, 
        to_raise=True, to_reduce=False, 
        raise_quadratic=True, reduce_quadratic=True)
    
    # using fully connected neural networks as dimension operator
    # model = build_model_dnn(state_len, num_classes, to_raise=True, to_reduce=True)
    ```

+   Train with AECA

    ```python
      # vanilla
      # ECMM = EigenDist
      # vanilla
      # model.compile(loss=[categorical_bernoulli_crossentropy, categorical_bernoulli_crossentropy],
                    # loss_weights=[0.5, 0.5],
                    # optimizer=keras.optimizers.Adadelta(),
                    # metrics=['accuracy'])
    
      # approx
      ECMM = EigenDistApprox
      # model.compile(loss=keras.losses.categorical_crossentropy,
      # model.compile(loss=keras.losses.mean_squared_error,
      # model.compile(loss=categorical_bernoulli_crossentropy,
      # approx
      model.compile(loss=[categorical_crossentropy, categorical_crossentropy],
                    loss_weights=[0.5, 0.5],
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
    ```

+   Set *to_train* to be **True** and training on MNIST data set

    ```bash
    python base_network.py
    ```

    



## Train the 2-fold ECAN with VECA on MNIST data set

+   The only difference from training with AECA is in this block of code

    ```python
    # vanilla
    ECMM = EigenDist
    # vanilla
    model.compile(loss=[categorical_bernoulli_crossentropy, categorical_bernoulli_crossentropy],
                  loss_weights=[0.5, 0.5],
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    # approx
    # ECMM = EigenDistApprox
    # model.compile(loss=keras.losses.categorical_crossentropy,
    # model.compile(loss=keras.losses.mean_squared_error,
    # model.compile(loss=categorical_bernoulli_crossentropy,
    # approx
    # model.compile(loss=[categorical_crossentropy, categorical_crossentropy],
    #               loss_weights=[0.5, 0.5],
    #               optimizer=keras.optimizers.Adadelta(),
    #               metrics=['accuracy'])
    ```

    



# Dimension reduction

In the history folder, with the corresponding MAGIC and WORK MAGIC code, we can find the obtained  EFM *P* , ECMM *LL*. In the 2-fold ECAN, EFM and ECMM has suffix a number indicating the corresponding fold. The RaDO or ReDO all belong to the 1st fold. In ECAN, the identity operator is installed every other fold since two consecutive dimension operators in a row are trival. 

## Dimension reduction with ECA

```python
LL = np.round(LL)

x_norm = np.linalg.norm(x, axis=1, keepdims=True)
x /= x_norm
x1 = np.matmul(x,P[:,np.sum(LL, axis=1)==1])
```

## Dimension reduction with 2-fold ECAN

```python
LL = np.round(LL)

x_norm = np.linalg.norm(x, axis=1, keepdims=True)
x /= x_norm
# change of basis
psi = np.matmul(x,P1)

# nonlinear dimension reduction
x1 = ReDO(redo, psi)

x1_norm = np.linalg.norm(x1, axis=1, keepdims=True)
x1 /= x1_norm
# linear dimension reduction
x2 = np.matmul(x1,P2[:,np.sum(LL2, axis=1)==1])
```

The reducing dimension operator (ReDO) is defined as 

```python
def ReDO(redo, psi, real_eca=True):
	x = np.concatenate([np.ones(psi.shape[0],), psi],axis=1)
	x = x * x
  
	# guarantee the square root being legal in real ECA
	if real_eca:
		redo = redo * redo 					
    
	return np.sqrt(np.matmul(x, redo))
  
  
```







[//]: # "The code is kind of messy cuz of commented code, yet, I am still a perfect progrmamer. As my code is often self-explainable, so, marginal comments."
[//]: # "Email: rzchen2014@gmail.com"

