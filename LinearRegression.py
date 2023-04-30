###### Your ID ######
# ID1: 318875366
# ID2: 316387927
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    meanX = np.mean(X)
    maxX = np.max(X)
    minX = np.min(X)

    meanY = np.mean(y)
    maxY = np.max(y)
    minY = np.min(y)

    X = np.array([(i - meanX) / (maxX - minX) for i in X])
    y = np.array([(i - meanY) / (maxY - minY) for i in y])

    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    

    X = np.array([np.append(1, i) for i in X])        
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    m = len(X)
    error = np.matmul(X, theta) - y
    J = np.sum(error ** 2) / (2 * m)
    return J
    

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = len(X)
    for k in range(num_iters):
        h = np.matmul(X, theta)
        error = h - y
        gradient = np.matmul(X.T, error) / m
        theta -= alpha * gradient
        J_history.append(compute_cost(X, y, theta))

        # if (k % 100 == 0):
        #     print("Iteration: ", k, " Cost: ", J_history[-1], " Theta: ", theta)
    
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    XT = np.transpose(X)
    X_inverse = np.linalg.inv(np.matmul(XT, X))
    pinv_theta = np.dot(np.matmul(X_inverse, XT), y)

    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = len(X)
    for k in range(num_iters):
        h = np.matmul(X, theta)
        error = h - y
        gradient = np.matmul(X.T, error) / m
        theta -= alpha * gradient
        J_history.append(compute_cost(X, y, theta))

        # if (k % 1000 == 0 and k > 1):
        #     print("Alpha: ", alpha, "Iteration: ", k,
        #            " Cost: ", J_history[-1], " Theta: ", theta, "Delta Cost: ", J_history[-2] - J_history[-1])
    
        if (k > 1 and (J_history[-2] - J_history[-1]) < 1e-8):
            break

    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    for alpha in alphas:
        # assign theta to a random value
        theta = np.random.rand(X_train.shape[1])
        theta, J_history = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)
        cost = compute_cost(X_val, y_val, theta)
        alpha_dict[alpha] = cost
    
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    # add 5 features to the selected_features list
    for _ in range(5):
        best_cost = 100000000
        best_feature = 0
        # iterate over all features and select the best one
        for j in range(X_train.shape[1]):
            if j not in selected_features:
                # add the feature to the selected_features list and train the model
                selected_features.append(j)
                theta = np.random.rand(len(selected_features))
                theta, J_history = efficient_gradient_descent(X_train[:, selected_features], y_train, theta, best_alpha, iterations)
                cost = compute_cost(X_val[:, selected_features], y_val, theta)
                # find the best feature which gives the lowest cost
                if cost < best_cost:
                    best_cost = cost
                    best_feature = j
                selected_features.pop()
        selected_features.append(best_feature)
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    for i in range(len(df.columns)):
        for j in range(i, len(df.columns)):
            col1, col2 = df.columns[i], df.columns[j]
            if i == j:
                df_poly[col1 + '^2'] = df[col1] * df[col2]
            else:
                df_poly[col1 + '*' + col2] = df[col1] * df[col2]

    return df_poly
