'''
Based off the Model Representation Lab from the Coursera course titled:
"Supervised Machine Learning: Regression and Classification".
(Lab 02)

Goal: displaying a graph in matplotlib
'''

import numpy as np
import matplotlib.pyplot as plt


def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

def main():
    ###
    # We will be creating a model that predicts housing price based on 1 feature.
    # In this case, we will use the square footage of the house.
    # x = size in 1000 sq ft
    # y = price in $1000
    ###
    # example small data set:
    x_train = np.array([1.0, 2.0])
    y_train = np.array([300.0, 500.0])
    print(f"x_train = {x_train}")
    print(f"y_train = {y_train}")

    # m is the number of training examples
    m = x_train.shape[0] # you can also use the len() function here
    print(f"x_train.shape:", m)
    print(f"Number of training examples is: {m}")

    # in the lab, we used w and b as our line function parameters.
    # these are the parameters for our model! a more complex model would
    # have more values. 
    # w = slope, b = y-intercept
    # we set our w and b values (pretend these results are from training on a data set):
    w = 200
    b = 100

    tmp_f_wb = compute_model_output(x_train, w, b)

    # After we have a model, we can predict the price of a house given that model.
    x_i = 1.2
    cost_1200_sqft = w * x_i + b

    print(f"Prediction for x=1.2: ${cost_1200_sqft:.0f} thousand dollars")
    
    # Plot our model prediction
    plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
    # Plot the data points
    plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
    # Set the title
    plt.title("Housing Prices")
    # Set the y-axis label
    plt.ylabel('Price (in 1000s of dollars)')
    # Set the x-axis label
    plt.xlabel('Size (1000 sqft)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
