# Eigen component analysis (ECA) introduction

This is the repository for paper *Eigen component analysis: A quantum theory incorporated machine learning technique to find linearly maximum separable components.* It includes two main parts for the experiments, Eigen component analysis (ECA) and eigen component analysis network (ECAN). Either ECA or ECAN can be trained with vanilla eigen copomnent analysis (VECA) or approximated eigen copomnent analysis (AECA). As the article mentioned, VECA often result in a sparse result and better option for dimension reduction. 

ECA, in my humble opinion which I cannot say in the article,  is a top-ranking feature extraction or dimension reduction algorithm. The obtained eigenfeature matrix (EFM) and eigenfeature-class mapping matrix (ECMM) could be used to conduct the concrete dimension reduction. The concrete number of dimension reduction is near to the rank of whole data set if without background or noise. In addition, the variance ratio is close to 1 with this concrete dimension reduction. **For example, the concrete number of dimension reduction  for MNIST data set is 110 using VECA (or 328 using AECA) neither more or less. This could mean that the MNIST data set or the background/noise-free data set  only occupy a subspace with dimension 110. The difference between the result of VECA and AECA is that VECA ignored some less important information.**  In ECAN, with the dimension operator, the nonlinear dimension reduction outperforms many classical algorithms which will be reported in our future work.  

 I will upload the enviroment requirements later. I know the code is kind of messy, since I created many branches in this project and this repository is just one branch I choose. I will merge the code and  add some comments to help you understand this project. 

# Quick start

## Train VECA

+   The files include twodim.py, threedim.py, bc.py, wis.py, mnist.py correspoinding to 2D, 3D, Wis1992, Wis1995, MNIST data set mentioned in the article. 

+   Set the *to_train* option to be True otherwise it will just test on previous saved model.

+   Then training on Wis1992 should be 

    ```bash
    python bc.py
    ```

## Train AECA

I will upload this part of code later. 

## Train the 2-fold ECAN with AECA on MNIST data set

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

+   Use VECA or AECA

    ```python
      # vanilla
      # ECMM = EigenDist
      # approx
      ECMM = EigenDistApprox
    
      # model.compile(loss=keras.losses.categorical_crossentropy,
      # model.compile(loss=keras.losses.mean_squared_error,
      # model.compile(loss=categorical_bernoulli_crossentropy,
      # vanilla
      # model.compile(loss=[categorical_bernoulli_crossentropy, categorical_bernoulli_crossentropy],
      # approx
      model.compile(loss=[categorical_crossentropy, categorical_crossentropy],
                    loss_weights=[0.5, 0.5],
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
    ```

+   Set *to_train* to be True and training on MNIST data set

```bash
python base_network.py
```



## Train the 2-fold ECAN with VECA on MNIST data set

+   The only difference is in this block of code

```python
  # vanilla
  ECMM = EigenDist
  # approx
  # ECMM = EigenDistApprox

  # model.compile(loss=keras.losses.categorical_crossentropy,
  # model.compile(loss=keras.losses.mean_squared_error,
  # model.compile(loss=categorical_bernoulli_crossentropy,
  # vanilla
  model.compile(loss=[categorical_bernoulli_crossentropy, categorical_bernoulli_crossentropy],
  # approx
  # model.compile(loss=[categorical_crossentropy, categorical_crossentropy],
                loss_weights=[0.5, 0.5],
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
```





[//]: # "The code is kind of messy cuz of commented code, yet, I am still a perfect progrmamer. As my code is often self-explainable, so, marginal comments."
[//]: # "Email: rzchen2014@gmail.com"

