# -*- coding: utf-8 -*-
import six
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import sys, time, math

#plt.style.use('ggplot')

# draw a image of handwriting number
def draw_digit_ae(data, n, row, col, _type):
    size = 28
    plt.subplot(row, col, n)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,size)
    plt.ylim(0,size)
    plt.pcolor(Z)
    plt.title("type=%s"%(_type), size=8)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

# draw digit images
def draw_digit_w1(data, n, i, length):
    size = 28
    plt.subplot(math.ceil(length/15), 15, n)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,size)
    plt.ylim(0,size)
    plt.pcolor(Z)
    plt.title("%d"%i, size=9)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")



mnist = ""

with open('mnist.pkl', 'rb') as mnist_pickle:
    mnist = six.moves.cPickle.load(mnist_pickle)

# 確率的勾配降下法で学習させる際の１回分のバッチサイズ
batchsize = 100
n_epoch = 20 #学習の繰り返し回数
n_units = 784

# Prepare dataset
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

model = FunctionSet(l1=F.Linear(784, n_units),
                            l2=F.Linear(n_units, n_units),
                            l3=F.Linear(n_units, 10))


# Neural net architecture
def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y = model.l3(h2)
    return F.softmax_cross_entropy(y, t)


# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)


l1_W = []
l2_W = []

l1_b = []
l2_b = []

train_loss = []
test_loss = []
test_mean_loss = []

prev_loss = -1
loss_std = 0

loss_rate = []

# Learning loop
for epoch in xrange(1, n_epoch+1):
    print 'epoch', epoch
    start_time = time.clock()

    # training
    perm = np.random.permutation(N)
    sum_loss = 0
    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]
        
        optimizer.zero_grads()
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        train_loss.append(loss.data)
        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize

    print '\ttrain mean loss={} '.format(sum_loss / N)
    
    # evaluation
    sum_loss     = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]
        loss = forward(x_batch, y_batch, train=False)

        test_loss.append(loss.data)
        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize

    loss_val = sum_loss / N_test
    
    print '\ttest  mean loss={}'.format(loss_val)
    if epoch == 1:
        loss_std = loss_val
        loss_rate.append(100)
    else:
        print '\tratio :%.3f'%(loss_val/loss_std * 100)
        loss_rate.append(loss_val/loss_std * 100)
        
    if prev_loss >= 0:
        diff = loss_val - prev_loss
        ratio = diff/prev_loss * 100
        print '\timpr rate:%.3f'%(-ratio)
    
    prev_loss = sum_loss / N_test
    test_mean_loss.append(loss_val)
    
    l1_W.append(model.l1.W)
    l2_W.append(model.l2.W)
    end_time = time.clock()
    print "\ttime = %.3f" %(end_time-start_time)

# Draw mean loss graph
"""
plt.style.use('ggplot')
plt.figure(figsize=(10,7))
plt.plot(test_mean_loss, lw=1)
plt.title("")
plt.ylabel("mean loss")
plt.show()
plt.xlabel("epoch")
"""

plt.figure(figsize=(15,70))
cnt = 1
#for i in range(len(l1_W[9])):
for i in range(780):
    draw_digit_w1(l1_W[9][i], cnt, i, len(l1_W[9][i]))
    cnt += 1
    
plt.show()


"""
入力層と出力層の可視化
plt.style.use('fivethirtyeight')

plt.figure(figsize=(15,25))

num = 100
cnt = 0
ans_list  = []
pred_list = []
for idx in np.random.permutation(N_test)[:num]:
    xxx = x_test[idx].astype(np.float32)
    h1 = F.dropout(F.relu(model.l1(Variable(xxx.reshape(1,784)))),  train=False)
    y  = model.l2(h1)
    cnt+=1
    ans_list.append(x_test[idx])
    pred_list.append(y)

cnt = 0
for i in range(int(num/10)):
    for j in range (10):
        img_no = i*10+j
        pos = (2*i)*10+j
        draw_digit_ae(ans_list[img_no],  pos+1, 20, 10, "ans")
        
    for j in range (10):
        img_no = i*10+j
        pos = (2*i+1)*10+j
        draw_digit_ae(pred_list[i*10+j].data, pos+1, 20, 10, "pred")
    
plt.show()
"""
