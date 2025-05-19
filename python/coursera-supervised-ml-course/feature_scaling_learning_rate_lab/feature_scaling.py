'''
Based off the Feature Scaling and Learning Rate (Multi-variable) Lab from the Coursera course titled:
"Supervised Machine Learning: Regression and Classification".
(Lab 03, Module 2)

Goals:
- Utilize the multiple variables routines developed in the previous lab
- run Gradient Descent on a data set with multiple features
- explore the impact of the learning rate alpha on gradient descent
- improve performance of gradient descent by feature scaling using z-score normalization
'''

import numpy as np
import matplotlib.pyplot as plt
from lab_utils import run_gradient_descent, plot_cost_i_w, norm_plot, plt_equal_scale
np.set_printoptions(precision=2)

'''
We will use the motivating example of housing price prediction.
The training data set contains many examples with 4 features
(size, bedrooms, floors and age) shown in the table below.
Note, in this lab, the Size feature is in sqft.

We would like to build a linear regression model using these values
so we can then predict the price for other houses - say,
a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old.
'''
# loading in the data from the provided text file
data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
X_train = data[:,:4]
y_train = data[:,4]
X_features = ['size(sqft)','bedrooms','floors','age']

# Plotting each feature vs. the target, price, provides some indication
# of which features have the strongest influence on price.
# In this data set, increasing size also increases price. 
# Bedrooms and floors don't seem to have a strong impact on price.
# Newer houses have higher prices than older houses.
"""
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()
"""

# ===== Learning Rate =====

'''
now, lets run gradient descent on this data set. refer to the lab_utils
to recall how gradient descent is run. 
we'll try a few different values of alpha (or different learning rates)
on the data set to see what works
'''
# set alpha to 9.9e-7
print("\nWhen learning rate = 9.9e-7:")
_, _, hist = run_gradient_descent(X_train, y_train, iterations=10, alpha = 9.9e-7)
plot_cost_i_w(X_train,y_train,hist)
# It appears the learning rate is too high. The solution does not converge. 
# Cost is increasing rather than decreasing. Let's plot the result:

#set alpha to 9e-7
print("\nWhen learning rate = 9e-7:")
_,_,hist = run_gradient_descent(X_train, y_train, iterations=10, alpha = 9e-7)
plot_cost_i_w(X_train,y_train,hist)
# Cost is decreasing 
# You see that cost is decreasing throughout the run showing that alpha is not too large.
# 𝑤0 is still oscillating around the minimum, but the cost is decreasing with every iteration
# rather than increasing. 
# Note above that dj_dw[0] changes sign with each iteration as w[0] jumps over the optimal value.
# This alpha value will converge. You can vary the number of iterations to see how it behaves.

#set alpha to 1e-7
print("\nWhen learning rate = 1e-7:")
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 1e-7)
plot_cost_i_w(X_train,y_train,hist)
# You can see that cost is decreasing as it should.
# 𝑤0 is approaching the minimum without oscillations. dj_w0 is negative throughout the run.
# This solution will also converge.


# ===== Feature Scaling =====
'''
With this dataset, you can see with more iterations that the parameters reach the minimum
at very different rates. Specifically, w0 reaches its minimum much earlier

Why are the parameters (w's) updated unevenly?
- learning rate alpha is shared by all parameter updates
- the common error term is multiplied by the features for the w's and not the b parameter
- the features vary significantly in magnitude making some features update much faster than others. 
  In this case, w0 is multiplied by 'size(sqft)', which is generally > 1000, while 
  w1 is multiplied by 'number of bedrooms', which is generally 2-4.
The solution is Feature Scaling. The course covered 3 methods, but we'll only look at
z-score normalization
'''

# After z-score normalization, all features will have a mean of 0 and a standard deviation of 1.

# Implementation Note: When normalizing the features, it is important to store the values used
# for normalization - the mean value and the standard deviation used for the computations.
# After learning the parameters from the model, we often want to predict the prices of houses
# we have not seen before. Given a new x value (living room area and number of bedrooms), we must
# first normalize x using the mean and standard deviation that we had previously computed from 
# the training set.
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma
    return (X_norm, mu, sigma)


# Let's look at the steps involved in Z-score normalization. The plots below shows the transformation step by step.
mu     = np.mean(X_train,axis=0)   
sigma  = np.std(X_train,axis=0) 
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma      

fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:,0], X_train[:,3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_mean[:,0], X_mean[:,3])
ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_norm[:,0], X_norm[:,3])
ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()

'''
The plots show the relationship between two of the training set parameters, "age" and "size(sqft)". 
These are plotted with equal scale.

Left:   Unnormalized: The range of values or the variance of the 'size(sqft)' feature is much larger than that of age
Middle: The first step removes the mean or average value from each feature. This leaves features that are centered 
        around zero. It's difficult to see the difference for the 'age' feature, but 'size(sqft)' is clearly around zero.
Right:  The second step divides by the standard deviation. This leaves both features centered at zero with a similar scale.
'''

# Let's normalize the data and compare it to the original data:
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print("\n\n")
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")
# The peak to peak range of each column is reduced from a factor of thousands to a factor of 2-3 by normalization.

fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.show()

fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count"); 
fig.suptitle("distribution of features after normalization")
plt.show()

# Notice, above, the range of the normalized data (x-axis) is centered around zero and roughly +/- 2.
# Most importantly, the range is similar for each feature.
# Let's re-run our gradient descent algorithm with normalized data. 
# Note the vastly larger value of alpha. This will speed up gradient descent:
w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )

# The scaled features get very accurate results much, much faster!
# Notice the gradient of each parameter is tiny by the end of this fairly short run.
# A learning rate of 0.1 is a good start for regression with normalized features.
# Let's plot our predictions versus the target values. Note, the prediction is made
# using the normalized feature while the plot is shown using the original feature values. 

# predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features    
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp, label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

# The results look good. A few points to note:
# - with multiple features, we can no longer have a single plot showing results versus features.
# - when generating the plot, the normalized features were used. Any predictions using the 
#   parameters learned from a normalized training set must also be normalized.

# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print("\n", x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f"\npredicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")

'''
Another way to view feature scaling is in terms of the cost contours.
When feature scales do not match, the plot of cost versus parameters in a contour plot is asymmetric.

In the plot below, the scale of the parameters is matched. The left plot is the cost contour plot of w[0],
the square feet versus w[1], the number of bedrooms before normalizing the features. The plot is so asymmetric,
the curves completing the contours are not visible. In contrast, when the features are normalized,
the cost contour is much more symmetric. The result is that updates to parameters during gradient descent can
make equal progress for each parameter.
'''
plt_equal_scale(X_train, X_norm, y_train)
