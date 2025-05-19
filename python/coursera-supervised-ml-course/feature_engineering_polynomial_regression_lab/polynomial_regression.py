'''
Based off the Feature Engineering and Polynomial Regression Lab from the Coursera course titled:
"Supervised Machine Learning: Regression and Classification".
(Lab 04, Module 2)

Goals:
- explore feature engineering and polynomial regression which allows you to use the 
  machinery of linear regression to fit very complicated, even very non-linear functions.
'''
import numpy as np
import matplotlib.pyplot as plt
from utils import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

'''
Polynomial Features:
Let's try using what we know so far to fit a non-linear curve. We'll start with a simple quadratic: y=1+x^2
We'll use np.c_[..] which is a NumPy routine to concatenate along the column boundary.
'''
# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w,model_b = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
plt.plot(x,X@model_w + model_b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()


'''
Well, as expected, not a great fit. What is needed is something like a polynomial feature. 
To accomplish this, you can modify the input data to engineer the needed features.
If you swap the original data with a version that squares the value, you get a polynomial equation
'''
# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# Engineer features 
X = x**2  # <-- added engineered feature
X = X.reshape(-1, 1)  # X should be a 2-D Matrix
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# This new polynomial equation yields a nearly perfect fit!

'''
Selecting Features
Above, we knew that an x^2 term was required. It may not always be obvious which features are required.
One could add a variety of potential features to try and find the most useful.
For example, what if we had instead tried an equation with x, x^2, and x^3 terms?
'''
# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   # <-- added engineered feature
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

'''
Note the values of w: [0.08 0.54 0.03] and b is 0.0106.
 
Gradient descent has emphasized the data that is the best fit to the x^2 data by increasing the corresponding weight
term relative to the others. If you were to run for a very long time, it would continue to reduce the impact of the other terms.

Gradient descent is picking the 'correct' features for us by emphasizing its associated parameter

Let's review this idea:
- less weight value implies less important/correct feature, and in extreme, when the weight becomes zero or very 
  close to zero, the associated feature is not useful in fitting the model to the data.
- above, after fitting, the weight associated with the x^2 feature is much larger than the weights for x
  or x^3 as it is the most useful in fitting the data.
'''

'''
An Alternate View
Above, polynomial features were chosen based on how well they matched the target data.
Another way to think about this is to note that we are still using linear regression once we have created new features.
Given that, the best features will be linear relative to the target. This is best understood with an example.
'''
# create target data
x = np.arange(0, 20, 1)
y = x**2
# engineer features .
X = np.c_[x, x**2, x**3]   # <-- added engineered feature
X_features = ['x','x^2','x^3']

fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
plt.show()
'''
Above, it is clear that the x^2 feature mapped against the target value y is linear.
Linear regression can then easily generate a model using that feature.
'''

'''
Scaling features
As described in the last lab, if the data set has features with significantly different scales,
one should apply feature scaling to speed gradient descent. In the example above, there is x,
x^2, and x^3, which will naturally have very different scales. 
Let's apply Z-score normalization to our example.
'''
# create target data
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization 
X = zscore_normalize_features(X)     
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")

x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X) 

# with normalized features, we can use a more aggressive learning rate
# feature scaling will make gradient descent converge much faster
model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


'''
Complex Functions
With feature engineering, even quite complex functions can be modeled:
'''
x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X) 

model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha = 1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# the result is quite accurate!





