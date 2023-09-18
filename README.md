# ML_1

There are two parts to this assignment namely part1.py and part2.py. Before proceeding further install all the required libraries.

Libraries - pandas, numpy, scikit-learn, seaborn, matplotlib

# Part_1

**Important Note: while executing the Part_1.py therw will be some depecration warnings in the terminal, ignore them and wait for some time.**

Part 1 consists of implementing the gradient descent algorithm for the multivariate data from scratch instead of using the libraries.

All the code is properly managed by using the classes and functions. Dataset used for this assignment is wine quality dataset which is downloaded and accessed from github using this following [URL](https://raw.githubusercontent.com/saitharun051/ML_1/main/winequality-red.csv)

Class Name is Assignment1

Functions defined in the class are
- pre_process --- Does the pre_processing such as importing the data, checking for NA values, etc.
- Normalization(data) --- This function takes data frame as input divides the X and Y (target) and Normalizes the whole data using MinMax Scaling.
- Split(X, Y, Iterations, Learning_Rate) --- splits the data into training and testing set using train_test_split method, and finally a call to the next function called gradient_descent is made.
- gradient_descent(pred_fn, x, y, X_test, Y_test, iterations = 10000, learning_rate = 0.0001, stopping_threshold = 1e-6) --- This function takes multiple inputs such as predictive_function which is used to predict the output, other inputs are iterations, tolerance, learning_rate.
- predictive_function(x,w) --- takes input 2 parameters namely Input variables X and weights of these parameters.

Code from Line **97** is executed automatically and the required functions are called with the class objects. 

For finding the optimal solution multiple combinations of iterations and learning rates are applied using a for loop. And an if coniditon at the end is added to show the final best optimal values of learning rate and iterations.


# Part_2

Class Name is Assignment1_Part2

Part_2 completely deals with the implementing the Linear regression model using the scikit-learn library.

There are similar functions like pre_process(), Normalization() and a function called pre_trained() that does the creating the Model object, fitting the data and prediction.

