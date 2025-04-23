'''
Based off the Python Numpy Vectorization Lab from the Coursera course titled:
"Supervised Machine Learning: Regression and Classification".
(Lab 01, Module 2)
Second Half

Goal: learn how to utilize numpy matrices (2D arrays).
'''

import numpy as np

"""
Matrices, are two dimensional arrays. The elements of a matrix are all of the same type.
In notation, matrices are denoted with capitol, bold letter such as 𝐗.
In this and other labs, m is often the number of rows and n the number of columns.
The elements of a matrix can be referenced with a two dimensional index. In math settings,
numbers in the index typically run from 1 to n. In computer science and these labs, indexing
will run from 0 to n-1.

NumPy's basic data structure is an indexable, n-dimensional array containing elements of the same type (dtype). 
Matrices have a two-dimensional (2-D) index [m,n].
"""

print("\nExamples of Matrix Creation:")
print("The same functions that created 1-D vectors will create 2-D or n-D arrays. Here are some examples:")
a = np.zeros((1, 5))                                    
print(f"a = np.zeros((1, 5))                | a shape = {a.shape}, \na = {a}\n")                     

a = np.zeros((2, 1))                                                                   
print(f"a = np.zeros((2, 1))                | a shape = {a.shape}, \na = {a}\n") 

a = np.random.random_sample((1, 1))  
print(f"a = np.random.random_sample((1, 1)) | a shape = {a.shape}, \na = {a}\n")


# NumPy routine which allocates memory and fills with user specified values
a = np.array([[5], [4], [3]])
print("Using user-specified values:")
print(f"a = np.array([[5], [4], [3]])       | a shape = {a.shape}, np.array: \na = {a}\n")

""" Operations on Matrices """

print("\nExamples of Matrix Operations:")

# vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2)   # reshape is a convenient way to create matrices
print(f"vector for indexing examples: \n{a}")

# access an element
print(f"\na[2,0].shape: {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar")

# access a row
print(f"\na[2].shape: {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")
# It is worth drawing attention to the last example. Accessing a matrix by just specifying the row will return a 1-D vector!

"""
A note on "reshape"
The previous example used reshape to shape the array.
a = np.arange(6).reshape(-1, 2)

This line of code first created a 1-D Vector of six elements. 
It then reshaped that vector into a 2-D array using the reshape command. 
This could have been written:

a = np.arange(6).reshape(3, 2)

To arrive at the same 3 row, 2 column array. The -1 argument tells the routine 
to compute the number of rows given the size of the array and the number of columns.
"""

# vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"\nvector for slicing examples: \n{a}\n")

# access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = \n", a[0, 2:7:1], "\n,  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array\n")

# access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], "\n,  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], "\n,  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = \n", a[1,:], "\n,  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = \n", a[1],   "\n,  a[1].shape   =", a[1].shape, "a 1-D array")