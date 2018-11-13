# HW2 Deep Learning and Machine Learning Basics

## Task
* ROC curve (in report)
* PCA and LDA
* Object Recognition using CNN

## Requirement
* Python 3.6.4
* torch 0.4.1
* scipy 1.0.1
* numpy 1.15.2
* matplotlib 3.0.0

## PCA and LDA
* Implement PCA and LDA manually
* PCA Usage
     ```
        python3 hw2-2_pca.py [whole dataset] [input test image] [output test image]
     ```
     refer to results in directory `result`
     
* LDA Usage
     ```
        python3 hw2-2_pca.py [whole dataset] [output first fisherface]
     ```
     refer to results in directory `result`

## Object Recognition using CNN
* Implement simple CNN for recognizing MNIST dataset
* Usage
    * For training
    ```
        python3 hw2-3_train.py [image directory] 
    ```
    * For testing
    ```
        python3 hw2-3_test.py [image directory] [output csv] 
    ```
* Learning curve

    ![learning curve](https://github.com/fanoping/Computer-Vision/blob/master/hw2/result/curve.png)
