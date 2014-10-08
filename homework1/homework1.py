import pdb
import random
import pylab as pl
from scipy.optimize import fmin_bfgs
import math
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def regressionPlot(X, Y, order, l, regressionFnc):
    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = designMatrix(X, order)

    # compute the weight vector
    w = regressionFnc(phi, Y, l)
    print 'w', w

    # produce a plot of the values of the function 
    regressionPlotWeights(X, Y, order, w)

    return w

def regressionPlotWeights(X, Y, order, w):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')
    
    # produce a plot of the values of the function 
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
    pl.plot(pts, Yp.tolist()[0])
    pl.title("Test Data " + r'$M = 2, \lambda = 0.01$')
    pl.savefig("lad_2.png", bbox_inches='tight')
    pl.show()

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def bishopCurveData():
    # y = sin(2 pi x) + N(0,0.3),
    return getData('curvefitting.txt')

def regressAData():
    return getData('regressA_train.txt')

def regressBData():
    return getData('regressB_train.txt')

def validateData():
    return getData('regress_validate.txt')

# Matrix that transforms X into higher order polynomial
# First column is all 1s
def designMatrix(X, order):
    return PolynomialFeatures(degree = order).fit_transform(X)

def blogTrainData():
    x = np.genfromtxt("dataset/x_train.csv", delimiter=",")
    y = np.genfromtxt("dataset/y_train.csv")
    y = y.reshape(len(y), 1)
    ones = np.ones((x.shape[0], 1))
    phi = np.concatenate((ones, x), axis=1)
    return x, y, phi

def blogValidateData():
    x = np.genfromtxt("dataset/x_val.csv", delimiter=",")
    y = np.genfromtxt("dataset/y_val.csv")
    y = y.reshape(len(y), 1)
    ones = np.ones((x.shape[0], 1))
    phi = np.concatenate((ones, x), axis=1)
    return x, y, phi

def blogTestData():
    x = np.genfromtxt("dataset/x_test.csv", delimiter=",")
    y = np.genfromtxt("dataset/y_test.csv")
    y = y.reshape(len(y), 1)
    ones = np.ones((x.shape[0], 1))
    phi = np.concatenate((ones, x), axis=1)
    return x, y, phi

def plotError(x_train, x_val, y_train, y_val, phi_train, phi_val):
    lambdas = np.arange(1000, 20000, 100)
    err = []
    for l in lambdas:
        w = ridgeRegression(phi_train, y_train, l)
        e = SSE(w, phi_val, y_val)[0]
        err.append(e)
    plt.plot(lambdas, err)
    plt.xlabel("Value of " + r'$\lambda$')
    plt.ylabel("SSE")
    plt.title("SSE of Validation Data")
    plt.savefig("blog.png", bbox_inches='tight')
    plt.show()



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
    obj = []
    while i < MAXITER:
        new = old - beta * gradFunc(old, objFunc)
        obj.append(objFunc(new))
        if abs(objFunc(new) - objFunc(old)) <= epsilon:
            # print "Algorithm has converged at objective value " + str(objFunc(new)) + " after " + str(i) + " iterations."
            return (objFunc(new), new, obj, i)
        # print "old: " + str(old) + " new: " + str(new)
        old = new
        i += 1

    # print "Algorithm has not converged"
    return (objFunc(new), new, obj, i)

# Calculating the gradient of a function numerically at point X
def calcGrad(X, objFunc, h = 0.001):
    N = len(X)
    grad = np.zeros(N)
    for i in range(N):
        tempXPos = X.copy()
        tempXNeg = X.copy()
        tempXPos[i] = tempXPos[i] + 0.5 * h
        tempXNeg[i] = tempXNeg[i] - 0.5 * h
        centralDiff = objFunc(tempXPos) - objFunc(tempXNeg)
        grad[i] = centralDiff / h
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

# Gradient of function above
def gradMultMin(X, objFunc):
    newX = math.cos(X[0])
    newY = - math.sin(X[1])
    return np.array([newX, newY])


# Function compues the maximum likelihood weight vector given:
# Array of 1-dimensional data points
# Vector of Y values
# The value of M, the maximum order of a simple polynomial basis
def maxLikelihoodWeight(phi, Y):
    sumSquares = np.dot(np.transpose(phi), phi)
    sumSquaresInv = np.linalg.inv(sumSquares)
    a = np.dot(sumSquaresInv, np.transpose(phi))
    W = np.dot(a, Y)
    return W

# Calculates sum of squared error
def SSE(W, phi, Y):
    W = W.reshape((phi.shape[1], 1))
    phiWeight = np.dot(phi, W)
    rss = np.dot(np.transpose(phiWeight - Y), phiWeight - Y)
    return rss  

# Calculates gradient of sum of squared error
def gradSSE(W, phi, Y):
    W = W.reshape((phi.shape[1], 1))
    phiW = np.dot(phi, W)
    grad = 2 * np.dot(np.transpose(phi), phiW - Y)
    return grad

def numericGradSSE(W, phi, Y, h = 0.001):
    W = W.reshape((phi.shape[1], 1))
    N = len(W)
    grad = np.zeros(W.shape)
    for i in range(N):
        tempPos = W.copy()
        tempNeg = W.copy()
        tempPos[i][0] = tempPos[i][0] + 0.5 * h
        tempNeg[i][0] = tempNeg[i][0] - 0.5 * h
        centralDiff = SSE(phi, Y, tempPos, M) - SSE(phi, Y, tempNeg, M)
        grad[i][0] = centralDiff / h
    return grad

def gradDescentSSE(objFunc, gradFunc, initialX, phi, y, beta = 0.05, epsilon = 0.00001, MAXITER = 10000):
    i = 0
    initialX = initialX.reshape((phi.shape[1], 1))
    old = initialX 
    obj = []
    while i < MAXITER:
        new = old - beta * gradFunc(old, phi, y)
        obj.append(objFunc(new, phi, y)[0])
        if abs(objFunc(new, phi, y) - objFunc(old, phi, y)) <= epsilon:
            # print "Algorithm has converged at objective value " + str(objFunc(new, phi, y)) + " after " + str(i) + " iterations."
            return (objFunc(new, phi, y), new, obj, i)
        # print "old: " + str(old) + " new: " + str(new)
        old = new
        i += 1

    # print "Algorith has not converged at obj value " + str(objFunc(new, phi, y))
    return (objFunc(new, phi, y), new, obj, i)

# Implementing Ridge Regression

def centerMatrix(phi):
    phi = phi[:, 1:]
    c = phi - np.mean(phi, axis=0)
    return c

def centerY(y):
    return y - np.mean(y, axis=0)

def ridgeRegression(phi, Y, lambdaParam):
    Z = centerMatrix(phi)
    Yc = centerY(Y)
    I = np.identity(Z.shape[1])
    zSquared = np.dot(np.transpose(Z), Z)
    inverse = np.linalg.inv(zSquared + lambdaParam * I)
    W = np.dot(np.dot(inverse, np.transpose(Z)), Yc)
    w0 = np.mean(Y) - np.dot(np.transpose(W), np.mean(phi[:, 1:], axis=0))
    W = np.insert(W, 0, w0, axis=0)
    return W

def LAD(W, phi, Y, lambdaParam):
    W = W.reshape((phi.shape[1]), 1)
    errMatrix = abs(np.dot(phi, W) - Y)
    err = errMatrix.sum() + lambdaParam * np.dot(np.transpose(W), W)
    return err

def numericGradLambda(W, phi, Y, lambdaParam, objFunc, h = 0.0001):
    W = W.reshape((phi.shape[1], 1))
    N = len(W)
    grad = np.zeros(W.shape)
    for i in range(N):
        tempPos = W.copy()
        tempNeg = W.copy()
        tempPos[i][0] = tempPos[i][0] + 0.5 * h
        tempNeg[i][0] = tempNeg[i][0] - 0.5 * h
        centralDiff = objFunc(tempPos, phi, Y, lambdaParam) - objFunc(tempNeg, phi, Y, lambdaParam)
        grad[i][0] = centralDiff / h
    return grad

def gradDescentLambda(objFunc, gradFunc, initialX, phi, y, lambdaParam, beta = 0.01, epsilon = 0.0001, MAXITER = 10000):
    i = 1
    initialX = initialX.reshape((phi.shape[1], 1))
    old = initialX
    obj = []
    while i < MAXITER:
        new = old - beta * gradFunc(old, phi, y, lambdaParam, objFunc)
        # new = old - (1.0 / i) * gradFunc(old, phi, y, lambdaParam, objFunc)
        obj.append(objFunc(new, phi, y, lambdaParam)[0])
        if abs(objFunc(new, phi, y, lambdaParam) - objFunc(old, phi, y, lambdaParam)) <= epsilon:
            # print "Algorithm has converged at objective value " + str(objFunc(new, phi, y, lambdaParam)) + " after " + str(i) + " iterations."
            return (objFunc(new, phi, y, lambdaParam), new, obj, i)
        # print "old: " + str(old) + " new: " + str(new)
        old = new
        i += 1

    # print "Algorith has not converged at obj value " + str(objFunc(new, phi, y, lambdaParam))
    return (objFunc(new, phi, y, lambdaParam), new, obj, i)

def lasso(W, phi, Y, lambdaParam):
    W = W.reshape((phi.shape[1], 1))
    errMatrix = np.dot(phi, W) - Y
    err = np.dot(np.transpose(errMatrix), errMatrix) + lambdaParam * abs(W).sum()
    return err

def contourPlot1():
    dx1, dx2 = 0.05, 0.05
    x1, x2 = np.mgrid[slice(-5 - dx1, 5+dx1, dx1), slice(-5 - dx2, 5 + dx2, dx2)]
    obj = 3 * x1**2 + 2 * x2**2 + x1 + x2 + 3 * x1 * x2
    obj = obj[:-1, :-1]
    levels = MaxNLocator(nbins=15).tick_values(obj.min(), obj.max())
    cmap = plt.get_cmap('Oranges')
    plt.contourf(x1[:-1, :-1] + dx1 / 2.0, x2[:-1, :-1] + dx2 / 2,
                obj, levels=levels, cmap=cmap)
    plt.colorbar()
    plt.title('Contour Plot of Equation 1')
    plt.show()

def pltObjIteration(objFunc, gradFunc, initialX, phi, y, beta=0.05, epsilon=0.0001, MAXITER = 10000):
    conv, X, y, i = gradDescentSSE(objFunc, gradFunc, initialX, phi, y, beta=beta, epsilon=epsilon)
    x = np.arange(i + 1)
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("Objective Value vs Iteration")
    # plt.text(2, y[0], 'Step Size = ' + str(beta) + ', Convergence Threshold = ' + str(epsilon))
    plt.show()

def pltBetaVsIterations(objFunc, gradFunc, initialX, phi, y, epsilon=0.0001):
    betas = np.arange(0.001, 0.03, 0.001)
    iterations = []
    for b in betas:
        conv, X, objs, i = gradDescentSSE(objFunc, gradFunc, initialX, phi, y, beta=b, epsilon = epsilon, MAXITER=10000)
        iterations.append(i)
    plt.plot(betas, iterations)
    plt.xlabel("Step Size Value")
    plt.ylabel("Number of iterations for Convergence")
    plt.title("Number of Iterations Required vs Step Size Value")
    plt.text(0.015, 500, 'Convergence Threshold = ' + str(epsilon))
    plt.savefig('SSE_1.png', bbox_inches='tight')
    plt.show()
    
def pltLambdaVsIterations(objFunc, gradFunc, initialX, phi, y):
    lambdas = np.arange(0.1, 2, 0.1)
    iterations = []
    for l in lambdas:
        conv, X, objs, i = gradDescentLambda(objFunc, gradFunc, initialX, phi, y, l)
        iterations.append(i)
    plt.plot(betas, iterations)
    plt.xlabel("Step Size Value")
    plt.ylabel("Number of iterations for Convergence")
    plt.title("Number of Iterations Required vs Step Size Value")
    plt.text(0.015, 500, 'Convergence Threshold = ' + str(epsilon))
    plt.savefig('SSE_1.png', bbox_inches='tight')
    plt.show()

def absErr(W, phi, Y):
    W = W.reshape((phi.shape[1]), 1)
    errMatrix = abs(np.dot(phi, W) - Y)
    err = errMatrix.sum()
    return err

def findErrors(x_train, x_val, y_train, y_val):
    M = [1,2,3,5,9]
    lambdas = [0.0, 0.01, 0.1, 0.5, 1]
    err = []
    for m in M:
        initial = np.zeros(m + 1)
        phi_train = designMatrix(x_train, m)
        phi_val = designMatrix(x_val, m)
        for l in lambdas:
            w = gradDescentLambda(lasso, numericGradLambda, initial, phi_train, y_train, l)[1]
            e = SSE(w, phi_val, y_val)
            err.append(e)
    return err








































