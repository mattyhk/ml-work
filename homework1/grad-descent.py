import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures

# Gradient Descent Implementation
# objFunc - the function to be optimized
# gradFunc - a function that calculates the gradient
# initialX - the initial guess
# beta - the step size
# epsilon - the convergence criterion
# MAXITER - the maximum number of iterations this function can repeat
def gradDescent(objFunc, gradFunc, initialX, beta = 0.1, epsilon = 0.0001, MAXITER = 1000):
  i = 0
  old = initialX 
  while i < MAXITER:
    # print i
    new = old - beta * gradFunc(old, objFunc)
    if abs(objFunc(new) - objFunc(old)) <= epsilon:
      print "Algorithm has converged at objective value " + str(objFunc(new)) + " after " + str(i) + " iterations."
      return
    # print "old: " + str(old) + " new: " + str(new)
    old = new
    i += 1

  print "Algorith has not converged"
  return

# Calculating the gradient of a function numerically at point X
def calcGrad(X, objFunc, h = 0.001):
  centralDiff = objFunc(X + 0.5 * h) - objFunc(X - 0.5 * h)
  grad = centralDiff / h
  return grad

# Parabolic bowl function
# 3x^2 + 2y^2 + x + y + 3xy
def paraBowl(X):
  val = 3 * X[0]**2 + 2 * X[1]**2 + X[0] + X[1] + 3 * X[0] * X[1]
  return val

# Taking Gradient of parabolic bowl above
def paraBowlGrad(X, objFunc):
  newX = 6.0 * X[0] + 3.0 * X[1] + 1
  newY = 3.0 * X[0] + 4.0 * X[1] + 1
  return np.array([newX, newY])

# Non convex function
# x^3 + 2y^2 + 3xy
def nonConvexFunc(X):
  val = X[0] ** 3 + 2.0 * X[1] ** 2 + 3 * X[0] * X[1]
  return val

# Taking gradient of non convex function above
def gradNonConvex(X, objFunc):
  newX = 3.0 * X[0]**2 + 3.0 * X[1]
  newY = 4.0 * X[1] + 3 * X[0]
  return np.array([newX, newY])

# Convex function with many min
# sin(x) + cos(y)
def multMinFunc(X):
  val = math.sin(X[0]) + math.cos(X[1])
  return val

# Gradient of functio above
def gradMultMin(X, objFunc):
  newX = math.cos(X[0])
  newY = - math.sin(X[1])
  return np.array([newX, newY])

# Function compues the maximum likelihood weight vector given:
# Array of 1-dimensional data points
# Vector of Y values
# The value of M, the maximum order of a simple polynomial basis
def maxLikelihoodWeight(X, Y, M):
  poly = PolynomialFeatures(degree = M)
  phi = poly.fit_transform(X)
  sumSquares = np.dot(np.transpose(phi), phi)
  sumSquaresInv = np.linalg.inv(sumSquares)
  a = np.dot(sumSquaresInv, np.transpose(phi))
  W = np.dot(a, Y)
  return W

# Calculates sum of squared error
def SSE(X, Y, W, M):
  # N = Y.size
  # phi = PolynomialFeatures(degree = M).fit_transform(X)
  # rss = 0
  # for i in range(N):
  #   rss = rss + np.dot(np.transpose(W), phi[i])[0] ** 2
  # return rss

  phi = PolynomialFeatures(degree = M).fit_transform(X)
  phiWeight = np.dot(phi, W)
  rss = np.dot(np.transpose(phiWeight - Y), phiWeight - Y)
  return rss  

# Calculates gradient of sum of squared error
def gradSSE(X, Y, W, M):
  N = Y.size
  phi = PolynomialFeatures(degree = M).fit_transform(X)
  phiW = np.dot(phi, W)
  grad = 2 * np.dot(np.transpose(phi), phiW - Y)
  return grad





















































