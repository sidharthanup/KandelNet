#!/bin/python2

import numpy as np

W = np.zeros((10,10))

learn_rate = 0.07

inhibition_rate = 0.008

multiplier = 1

n = 2

#def initialize_weights():
#    for i in range(0,10):
#        for j in range(0,10):
#            if i != j:
#                W[i,j] = -1


def distance(i,j):
    return abs(i-j)+1


def train(x, y, num_epochs, multiplier):
    for epochs in range(0, num_epochs):
        for i in range(0,10):
            for j in range(0,10):
              if i != j:
                  #Reward associations
                  if y[j] == 1:
                    W[i,j] = W[i,j] + (((x[i]*y[j])+(y[i]*y[j]))*learn_rate*multiplier)/distance(i,j)
                    W[i,j] = max(0, W[i,j])
                  #Penalize disassociations
                  elif y[j] == 0 and (x[i] == 1 or y[i] == 1):
                    W[i,j] = W[i,j] - ((x[i]+y[i])*inhibition_rate*multiplier)/distance(i,j)
                    W[i,j] = max(0, W[i,j])
              else:
                  #Slightly different Update for i == j because we dont want the neurons to update itself
                  if y[j] == 1:
                      W[i,j]  = W[i,j] + ((x[i]*y[j])*learn_rate*multiplier)/distance(i,j)
                      W[i,j] = max(0, W[i,j])
                  elif y[j] == 0 and x[i] == 1:
                      W[i,j] = W[i,j] - (x[i]*inhibition_rate*multiplier)/distance(i,j)
                      W[i,j] = max(0, W[i,j])
    multiplier = multiplier*1.3
    return multiplier

def test(x, inhibit_vec = {}):
    Z = np.zeros(10)
    y = np.zeros(10)
    m = 0
    #for i in range(0,5):
       # if x[i] == 1:
           # Z[i] = 500
    while (m != n):
        for i in range(0,10):
            temp_sum = 0
            for j in range(0,10):
                temp_sum += W[j,i]*(x[j]+y[j])
            temp_Z = Z[i]
            Z[i] += temp_sum
            #Z[i] = Z[i] - inhibit_vec[i]*Z[i]
            if Z[i] >= 500:
                Z[i] = 500
                y[i] = 1
                if temp_Z < 500:
                    m += 1
            elif Z[i] < 500:
                y[i] = 0
                if temp_Z >= 500:
                    m -= 1
            if m == n:
                if inhibit_vec:
                    key = ""
                    for j in range(0,10):
                        if y[j] == 1:
                            key = key + str(j)
                    if inhibit_vec.get(key) == 1:
                        for c in key:
                            Z[int(c)] = 0
                            y[int(c)] = 0
                            m -= 1
                        continue
                break
    return y

#Update the hash table inhibit_vec
def set_inhibit_vec(x, inhibit_vec):
    key = ""
    for i in range(0,10):
        if x[i] == 1:
            key = key + str(i)
    inhibit_vec[key] = 1
    return inhibit_vec


def bfs(x, num_patterns):
    print("Here are all the stored patterns for the input vector:")
    inhibit_vec = {}
    for i in range(0, num_patterns):
        arr = test(x, inhibit_vec)
        print(arr)
        #inhibit_vec = np.logical_or(arr, inhibit_vec)
        inhibit_vec = set_inhibit_vec(arr, inhibit_vec)
        print inhibit_vec

def dfs(x, num_patterns):
    print("Here are the memories obtained as a result of depth-wise traversal starting from the input vector")
    for i in range(0, num_patterns):
        x = test(x, np.zeros(10))
        print(x)

#initialize_weights()

#Here's a set of mutually orthogonal 10 dimensional vectors with active neurons as 2 for training and testing. For 10d vec and n=2, we can have (10-2)C(2) orthogonal pattern associations stored for a given input vector. But we definitely lose the orthogonality when there is associations between those patterns. So for n = 2 and 10d vec this is the only possible mutually orthogonal set of vectors i.e 5 of them. Mutual orthogonality is crucial for perfect recall and perfect traversal.
x = np.array([1,1,0,0,0,0,0,0,0,0])
y = np.array([0,0,1,1,0,0,0,0,0,0])
a = np.array([0,0,0,0,1,1,0,0,0,0])
z = np.array([0,0,0,0,0,0,1,1,0,0])
b = np.array([0,0,0,0,0,0,0,0,1,1])
multiplier = train(x,y,100,multiplier)
multiplier = train(x,z,100,multiplier)
multiplier = train(x,a,100,multiplier)
train(a,b,100, multiplier)
bfs(x, 3)
print(W)
