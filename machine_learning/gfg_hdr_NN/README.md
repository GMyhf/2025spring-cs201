# Handwritten Digit Recognition using Neural Network

Last Updated : 30 Sep, 2024

https://www.geeksforgeeks.org/handwritten-digit-recognition-using-neural-network/



### Introduction:

Handwritten digit recognition using MNIST dataset is a major project made with the help of Neural Network. It basically detects the scanned images of handwritten digits. 

We have taken this a step further where our handwritten digit recognition system not only detects scanned images of handwritten digits but also allows writing digits on the screen with the help of an integrated GUI for recognition. 

### Approach: 

We will approach this project by using a three-layered Neural Network. 

- ***\*The input layer:\**** It distributes the features of our examples to the next layer for calculation of activations of the next layer.
- ***\*The hidden layer:\**** They are made of hidden units called activations providing nonlinear ties for the network. A number of hidden layers can vary according to our requirements.
- ***\*The output layer:\**** The nodes here are called output units. It provides us with the final prediction of the Neural Network on the basis of which final predictions can be made.

A neural network is a model inspired by how the brain works. It consists of multiple layers having many activations, this activation resembles neurons of our brain. A neural network tries to learn a set of parameters in a set of data which could help to recognize the underlying relationships. Neural networks can adapt to changing input; so the network generates the best possible result without needing to redesign the output criteria.

### Methodology:

We have implemented a Neural Network with 1 hidden layer having **100** activation units (excluding bias units). The data is loaded from a **.mat** file, features(X) and labels(y) were extracted. Then features are divided by **255** to rescale them into a range of **[0,1]** to avoid overflow during computation. Data is split up into **60,000** training and **10,000** testing examples. Feedforward is performed with the training set for calculating the hypothesis and then backpropagation is done in order to reduce the error between the layers. The regularization parameter lambda is set to 0.1 to address the problem of overfitting. Optimizer is run for 70 iterations to find the best fit model. 

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20210610171121/Screenshot532.png" alt="img" style="zoom:50%;" />

Layers of  Neural Network

***\*Note:\****

- Save all **.py** files in the same directory.
- Download dataset from Kaggle.

### Main.py

Importing all the required libraries, extract the data from **mnist-original.mat** file. Then features and labels will be separated from extracted data. After that data will be split into training (60,000) and testing (10,000) examples. Randomly initialize Thetas in the range of [-0.15, +0.15] to break symmetry and get better results. Further, the optimizer is called for the training of weights, to minimize the cost function for appropriate predictions. We have used the “**minimize**” optimizer from “**scipy.optimize**” library with “**L-BFGS-B**” method. We have calculated the test, the “training set accuracy and precision using “predict” function.



```python
from scipy.io import loadmat
import numpy as np
from Model import neural_network
from RandInitialize import initialise
from Prediction import predict
from scipy.optimize import minimize


# Loading mat file
data = loadmat('mnist-original.mat')

# Extracting features from mat file
X = data['data']
X = X.transpose()

# Normalizing the data
X = X / 255

# Extracting labels from mat file
y = data['label']
y = y.flatten()

# Splitting data into training set with 60,000 examples
X_train = X[:60000, :]
y_train = y[:60000]

# Splitting data into testing set with 10,000 examples
X_test = X[60000:, :]
y_test = y[60000:]

m = X.shape[0]
input_layer_size = 784  # Images are of (28 X 28) px so there will be 784 features
hidden_layer_size = 100
num_labels = 10  # There are 10 classes [0, 9]

# Randomly initialising Thetas
initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
initial_Theta2 = initialise(num_labels, hidden_layer_size)

# Unrolling parameters into a single column vector
initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
maxiter = 100
lambda_reg = 0.1  # To avoid overfitting
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)

# Calling minimize function to minimize cost function and to train weights
results = minimize(neural_network, x0=initial_nn_params, args=myargs, 
          options={'disp': True, 'maxiter': maxiter}, method="L-BFGS-B", jac=True)

nn_params = results["x"]  # Trained Theta is extracted

# Weights are split back to Theta1, Theta2
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (
                              hidden_layer_size, input_layer_size + 1))  # shape = (100, 785)
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
                      (num_labels, hidden_layer_size + 1))  # shape = (10, 101)

# Checking test set accuracy of our model
pred = predict(Theta1, Theta2, X_test)
print('Test Set Accuracy: {:f}'.format((np.mean(pred == y_test) * 100)))

# Checking train set accuracy of our model
pred = predict(Theta1, Theta2, X_train)
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y_train) * 100)))

# Evaluating precision of our model
true_positive = 0
for i in range(len(pred)):
    if pred[i] == y_train[i]:
        true_positive += 1
false_positive = len(y_train) - true_positive
print('Precision =', true_positive/(true_positive + false_positive))

# Saving Thetas in .txt file
np.savetxt('Theta1.txt', Theta1, delimiter=' ')
np.savetxt('Theta2.txt', Theta2, delimiter=' ')
```



```
/usr/bin/python3 /Users/hfyan/git/2025spring-cs201/lihang_code/tmp/Main.py 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        79510     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  6.72261D+00    |proj g|=  5.16344D-01
 This problem is unconstrained.

At iterate    1    f=  3.39049D+00    |proj g|=  5.85247D-02

At iterate    2    f=  3.27044D+00    |proj g|=  3.49894D-02

At iterate    3    f=  3.22147D+00    |proj g|=  2.44817D-02

At iterate    4    f=  3.17578D+00    |proj g|=  2.53452D-02

At iterate    5    f=  3.02798D+00    |proj g|=  3.83729D-02

At iterate    6    f=  2.67217D+00    |proj g|=  6.19362D-02

At iterate    7    f=  1.92126D+00    |proj g|=  7.02050D-02

At iterate    8    f=  1.48330D+00    |proj g|=  3.24052D-02

At iterate    9    f=  1.37794D+00    |proj g|=  2.49396D-02

At iterate   10    f=  1.29857D+00    |proj g|=  1.98121D-02

At iterate   11    f=  1.15678D+00    |proj g|=  3.94783D-02

At iterate   12    f=  1.06770D+00    |proj g|=  4.03991D-02

At iterate   13    f=  1.00937D+00    |proj g|=  1.69286D-02

At iterate   14    f=  9.59515D-01    |proj g|=  1.51583D-02

At iterate   15    f=  9.24319D-01    |proj g|=  1.66716D-02

At iterate   16    f=  8.41916D-01    |proj g|=  1.83286D-02

At iterate   17    f=  7.99573D-01    |proj g|=  2.22313D-02

At iterate   18    f=  7.51630D-01    |proj g|=  1.00434D-02

At iterate   19    f=  7.21871D-01    |proj g|=  8.08918D-03

At iterate   20    f=  6.86696D-01    |proj g|=  1.15758D-02

At iterate   21    f=  6.46718D-01    |proj g|=  8.55255D-03

At iterate   22    f=  6.19391D-01    |proj g|=  1.67903D-02

At iterate   23    f=  5.97221D-01    |proj g|=  4.26187D-03

At iterate   24    f=  5.81481D-01    |proj g|=  5.04949D-03

At iterate   25    f=  5.55705D-01    |proj g|=  9.42475D-03

At iterate   26    f=  5.42856D-01    |proj g|=  2.67651D-02

At iterate   27    f=  5.16896D-01    |proj g|=  7.20683D-03

At iterate   28    f=  5.04609D-01    |proj g|=  2.84284D-03

At iterate   29    f=  4.90895D-01    |proj g|=  6.67786D-03

At iterate   30    f=  4.74485D-01    |proj g|=  8.60338D-03

At iterate   31    f=  4.44353D-01    |proj g|=  9.83179D-03

At iterate   32    f=  4.28460D-01    |proj g|=  1.00148D-02

At iterate   33    f=  4.13922D-01    |proj g|=  3.85916D-03

At iterate   34    f=  4.04145D-01    |proj g|=  3.22304D-03

At iterate   35    f=  3.93153D-01    |proj g|=  5.76074D-03

At iterate   36    f=  3.78687D-01    |proj g|=  6.10178D-03

At iterate   37    f=  3.66389D-01    |proj g|=  3.57860D-03

At iterate   38    f=  3.57067D-01    |proj g|=  3.28725D-03

At iterate   39    f=  3.44243D-01    |proj g|=  3.21115D-03

At iterate   40    f=  3.35383D-01    |proj g|=  6.96796D-03

At iterate   41    f=  3.26534D-01    |proj g|=  2.69886D-03

At iterate   42    f=  3.16030D-01    |proj g|=  2.64054D-03

At iterate   43    f=  3.08557D-01    |proj g|=  3.13082D-03

At iterate   44    f=  2.99351D-01    |proj g|=  4.80460D-03

At iterate   45    f=  2.90844D-01    |proj g|=  1.89403D-03

At iterate   46    f=  2.80295D-01    |proj g|=  2.36775D-03

At iterate   47    f=  2.73601D-01    |proj g|=  3.83986D-03

At iterate   48    f=  2.66698D-01    |proj g|=  2.40862D-03

At iterate   49    f=  2.60690D-01    |proj g|=  1.80146D-03

At iterate   50    f=  2.54512D-01    |proj g|=  3.04735D-03

At iterate   51    f=  2.48697D-01    |proj g|=  5.53581D-03

At iterate   52    f=  2.42118D-01    |proj g|=  1.76182D-03

At iterate   53    f=  2.37117D-01    |proj g|=  2.10025D-03

At iterate   54    f=  2.30835D-01    |proj g|=  2.91770D-03

At iterate   55    f=  2.28595D-01    |proj g|=  8.48376D-03

At iterate   56    f=  2.20246D-01    |proj g|=  2.18143D-03

At iterate   57    f=  2.16404D-01    |proj g|=  9.61618D-04

At iterate   58    f=  2.11913D-01    |proj g|=  1.89967D-03

At iterate   59    f=  2.06766D-01    |proj g|=  2.35737D-03

At iterate   60    f=  2.02905D-01    |proj g|=  3.37709D-03

At iterate   61    f=  1.97549D-01    |proj g|=  9.11567D-04

At iterate   62    f=  1.94524D-01    |proj g|=  9.45538D-04

At iterate   63    f=  1.90316D-01    |proj g|=  2.81252D-03

At iterate   64    f=  1.86091D-01    |proj g|=  3.23584D-03

At iterate   65    f=  1.81876D-01    |proj g|=  1.60017D-03

At iterate   66    f=  1.77110D-01    |proj g|=  1.16247D-03

At iterate   67    f=  1.73939D-01    |proj g|=  1.85731D-03

At iterate   68    f=  1.70196D-01    |proj g|=  1.59616D-03

At iterate   69    f=  1.66457D-01    |proj g|=  4.11412D-03

At iterate   70    f=  1.62750D-01    |proj g|=  1.17716D-03

At iterate   71    f=  1.60224D-01    |proj g|=  1.43845D-03

At iterate   72    f=  1.55390D-01    |proj g|=  1.95279D-03

At iterate   73    f=  1.52625D-01    |proj g|=  2.30710D-03

At iterate   74    f=  1.49035D-01    |proj g|=  8.03972D-04

At iterate   75    f=  1.46238D-01    |proj g|=  8.55805D-04

At iterate   76    f=  1.42650D-01    |proj g|=  1.96298D-03

At iterate   77    f=  1.38982D-01    |proj g|=  2.04080D-03

At iterate   78    f=  1.35885D-01    |proj g|=  1.01397D-03

At iterate   79    f=  1.32411D-01    |proj g|=  8.80980D-04

At iterate   80    f=  1.29993D-01    |proj g|=  1.20817D-03

At iterate   81    f=  1.25923D-01    |proj g|=  1.53575D-03

At iterate   82    f=  1.22578D-01    |proj g|=  1.62065D-03

At iterate   83    f=  1.19410D-01    |proj g|=  1.04529D-03

At iterate   84    f=  1.16578D-01    |proj g|=  8.09510D-04

At iterate   85    f=  1.13953D-01    |proj g|=  1.23306D-03

At iterate   86    f=  1.10813D-01    |proj g|=  1.33200D-03

At iterate   87    f=  1.07920D-01    |proj g|=  9.39741D-04

At iterate   88    f=  1.05007D-01    |proj g|=  7.79462D-04

At iterate   89    f=  1.02922D-01    |proj g|=  1.09914D-03

At iterate   90    f=  1.00413D-01    |proj g|=  5.38215D-04

At iterate   91    f=  9.79954D-02    |proj g|=  6.15804D-04

At iterate   92    f=  9.52291D-02    |proj g|=  5.14250D-04

At iterate   93    f=  9.27205D-02    |proj g|=  1.21535D-03

At iterate   94    f=  9.04982D-02    |proj g|=  7.63553D-04

At iterate   95    f=  8.80600D-02    |proj g|=  5.19053D-04

At iterate   96    f=  8.60511D-02    |proj g|=  8.27906D-04

At iterate   97    f=  8.26643D-02    |proj g|=  5.25276D-04

At iterate   98    f=  8.05408D-02    |proj g|=  1.73722D-03

At iterate   99    f=  7.83393D-02    |proj g|=  8.41542D-04

At iterate  100    f=  7.71113D-02    |proj g|=  6.16883D-04

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
79510    100    105      1     0     0   6.169D-04   7.711D-02
  F =   7.7111296009525376E-002

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
Test Set Accuracy: 97.420000
Training Set Accuracy: 99.386667
Precision = 0.9938666666666667

```

