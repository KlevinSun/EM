import numpy as np
import random
import math
import matplotlib.pyplot as plt

def EM(dataSource, k, maxIter):         # dataSource is the IRIS data, K is the number of means, and the maxIter is the maximum iteration times
    m, n = np.shape(dataSource)
    weight = np.zeros(k)                # weight of k Gaussian
    mean = np.zeros((k, n))             # mean of k Gaussian
    covMat = np.zeros(k)                # covariance of k Gaussian

    probability = np.zeros((m, k))      # P(y = j | xi)
    paramMeans = np.zeros((k, n))       # params for return
    paramCovs = np.zeros(k)
    paramWeights = np.zeros(k)
    loglikes = np.zeros(maxIter)
    for j in range(k):
        weight[j] = 1/k                 # initiate weight of Gaussians
        covMat[j] = 1                   # initiate covariance of Guassians
        temp = random.randint(0, m-1)   # random means from sample data
        mean[j] = dataSource[temp]

       # temp = [1.0, 1.0, 1.0, 1.0]      # Give all temp to all means
       # mean[j] = temp

    #mean[k-1] = dataSource[random.randint(0, m-1)]     # give random sample data to last mean
    for i in range(maxIter):
        Estep(probability, dataSource, mean, covMat, weight, n)     # Estep
        Mstep(probability, dataSource, weight, covMat, mean, n)     # Mstep
        sample = loglikelihood(probability, dataSource, mean, covMat, weight, n)    # log-likelihood
        loglikes[i] = sample
    for x in range(k):
        paramWeights[x] = weight[x]
        paramMeans[x] = mean[x]
        paramCovs[x] = covMat[x]
    return paramWeights, paramMeans, paramCovs, loglikes


def Estep(probability, dataSource, mean, covMat, weight, n):
    m, k = np.shape(probability)

    for i in range(m):
        Pxy = np.zeros(k)
        sum = 0
        for j in range(k):
            Pxy[j] = sphGausM(dataSource[i], mean[j], covMat[j], n) * weight[j]     # non-normalized probability of ith data belongs to jth Guassian

            #print(sphGausM(dataSource[i], mean[j], covMat[j], n))
            sum += Pxy[j]
        for n in range(k):
            probability[i, n] = Pxy[n]/sum      # normalized probability of ith data belongs to jth Guassian


def sphGausM(dataSource, mean, cov, n):         # spherical Guassian model
    temp = 0
    for i in range(dataSource.size):
        temp += (dataSource[i] - mean[i])**2
    index = -temp/(2*cov)
    bottom = (2*math.pi*cov)**(n/2)
    result = math.e ** index / bottom
    return result


def Mstep(probability, dataSource, weight, covMat, mean, d):
    m, k = np.shape(probability)
    for i in range(k):
        n, ave, variance, w = 0, 0, 0, 0
        for j in range(m):
            n += probability[j, i]
            ave += probability[j, i] * dataSource[j]
        mean[i] = ave / n       # update mean of ith Gaussian
        weight[i] = n / m       # update weight of ith Gaussian
        for x in range(m):
            temp = 0
            for y in range(d):
                temp += (dataSource[x, y] - mean[i, y])**2
            variance += probability[x, i]*temp
        covMat[i] = variance / (d*n)        # update covariance of ith Gaussian


def loglikelihood(probability, dataSource, mean, covMat, weight, n):        # log-likelihood
    m, k = np.shape(probability)
    L = 0
    for i in range(m):
        for j in range(k):
            a = probability[i, j]
            d = sphGausM(dataSource[i], mean[j], covMat[j], n)

            b = np.log(d)
            c = np.log(weight[j])
            L += a*(b + c)
    return L

def plot_loglikes(loglikes, n):             # plot log-likelihoods of all iteration when number of Guassians is n
    dataArr = np.array(loglikes)
    m = np.shape(dataArr)[0]
    axis_x = []
    axis_y = []
    for i in range(m):
        axis_x.append(i)
        axis_y.append(dataArr[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x, axis_y, s=1, c='blue')
    plt.xlabel('iteration times'); plt.ylabel('loglikes');
    plt.savefig("likelihood when k=" + repr(n) + " and iteration=" + repr(loglikes.size))
    plt.show()

def main():
    data = np.loadtxt(open("C:\\test\\irisData.csv", "rb"), delimiter=",", skiprows=0)
    label = np.loadtxt(open("C:\\test\\irisLabels.csv", "rb"), delimiter=",", skiprows=0)

    sortLabel = np.argsort(label)
    data = data[sortLabel[:data.size]]

    weights, means, covs, loglikes = EM(data, 2, 1000)
    params = [weights, means, covs]
    return params, loglikes

if __name__ == "__main__":
    params, loglikes = main()
    plot_loglikes(loglikes, params[0].size)
    print("this is params in sequence of weights, means and covariance when k=" + repr(params[0].size) + ": \n" + repr(params))
    print("\nthis is likelihoods of " + repr(loglikes.size) + " times iterations: \n" + repr(loglikes))