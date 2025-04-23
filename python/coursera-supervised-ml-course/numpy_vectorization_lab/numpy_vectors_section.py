'''
Based off the Python Numpy Vectorization Lab from the Coursera course titled:
"Supervised Machine Learning: Regression and Classification".
(Lab 01, Module 2)
First Half

Goal: learn how to utilize numpy vectors and vectorization over iterative code.
'''

import numpy as np
import time

# naive, linear dot product via looping:
def naive_dot(a, b): 
    """
    Compute the dot product of two vectors

    Args:
    a (ndarray (n,)):  input vector 
    b (ndarray (n,)):  input vector with same dimension as a
        
    Returns:
    x (scalar): 
    """
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

def vector_creation():
    """
    Data creation routines in numpy will generally have a first parameter which is the shape of the object.
    This can either be a single value for a 1-D result or a tuple (n,m,...) specifying the shape of the result.
    Below are examples of creating vectors using these routines.
    """
    print("\n======== Running vector_creation ========\n")
    # NumPy routines which allocate memory and fill arrays with value
    a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
    a = np.zeros((4,));             print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
    a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

    """ Some data creation routines do not take a shape tuple: """
    # NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
    a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
    a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

    """ values can be specified manually as well """
    # NumPy routines which allocate memory and fill with user specified values
    a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
    a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
   
    """
    These have all created a one-dimensional vector a with four elements. 
    a.shape returns the dimensions. Here we see a.shape = (4,) indicating a 1-d array with 4 elements.
    """

    return a

def vector_operations():
    """ Some useful vector operations: """
    print("\n======== Running vector_operations ========\n")
    # === vector indexing operations on 1-D vectors ===
    a = np.arange(10)
    print("Example vector for Indexing Operations, a = np.arange(10):", a)

    # access an element
    print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")
    # access the last element, negative indexes count from the end
    print(f"a[-1] = {a[-1]}")

    # indices must be within the range of the vector or they will produce an error
    try:
        c = a[10]
    except Exception as e:
        print("The error message you'll see on out of bounds access is:")
        print(e)

    # === vector slicing operations (start:stop:step) ===
    a = np.arange(10)
    print("\nExample vector for Slicing Operations, a = np.arange(10):", a)
    # access 5 consecutive elements (start:stop:step)
    c = a[2:7:1];     print("a[2:7:1] = ", c)
    # access 3 elements separated by two 
    c = a[2:7:2];     print("a[2:7:2] = ", c)
    # access all elements index 3 and above
    c = a[3:];        print("a[3:]    = ", c)
    # access all elements below index 3
    c = a[:3];        print("a[:3]    = ", c)
    # access all elements
    c = a[:];         print("a[:]     = ", c)

    # === single vector operations ===
    a = np.array([1,2,3,4])
    print("\nExample vector for Single Vector Operations, a = np.arange(10):", a)
    # negate elements of a
    b = -a 
    print(f"b = -a        : {b}")
    # sum all elements of a, returns a scalar
    b = np.sum(a) 
    print(f"b = np.sum(a) : {b}")
    b = np.mean(a)
    print(f"b = np.mean(a): {b}")
    b = a**2
    print(f"b = a**2      : {b}")

    # === vector on vector element-wise operations ===
    a = np.array([ 1, 2, 3, 4])
    b = np.array([-1,-2, 3, 4])
    print("\nExample vector for Vector Vector Operations, a = np.array([ 1, 2, 3, 4]):", a)
    print("Example vector for Vector Vector Operations, b = np.array([-1,-2, 3, 4]):", b)

    # element-wise addition
    print(f"Binary operators work element wise (a + b): {a + b}")
    
    # try a mismatched vector operation
    c = np.array([1, 2])
    try:
        _ = a + c
    except Exception as e:
        print("The error message you'll see for mismatched vectors is:")
        print(e)

    # === scalar on vector operations ===
    a = np.array([1, 2, 3, 4])
    print("\nExample vector for Scalar Vector Operations, a = np.array([ 1, 2, 3, 4]):", a)

    # multiply a by a scalar
    b = 5 * a 
    print(f"b = 5 * a : {b}")


def dot_product():
    print("\n======== Running dot_product ========\n")
    # test 1-D
    
    a = np.array([1, 2, 3, 4])
    b = np.array([-1, 4, 3, 2])

    print("Example array a:", a)
    print("Example array b:", b)
    print(f"Naive, iterative approach via naive_dot(a, b) = {naive_dot(a, b)}")

    # test 1-D
    c = np.dot(a, b)
    print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
    c = np.dot(b, a)
    print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")


def dot_product_time_test(n: int):
    print("\n======== Running dot_product_time_test ========\n")
    np.random.seed(1)
    a = np.random.rand(n)
    b = np.random.rand(n)

    print("Testing with np arrays of size", n)
    
    tic = time.time()  # capture start time
    c = np.dot(a, b)
    toc = time.time()  # capture end time

    print(f"np.dot(a, b) =  {c:.4f}")
    print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

    tic = time.time()  # capture start time
    c = naive_dot(a,b)
    toc = time.time()  # capture end time

    print(f"naive_dot(a, b) =  {c:.4f}")
    print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

    del(a);del(b)  #remove arrays from memory


if __name__ == "__main__":
    vector_creation()
    vector_operations()
    dot_product()
    dot_product_time_test(n = 10000000)


