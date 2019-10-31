#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   optimizer.py    
@Contact :   271856330@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/16 10:53   jzh      1.0         None
'''

import tensorflow as tf
from Momentum import Momentum
from Adagrad import Adagrad
from Rmsprop import Rmsprop
from Adam import Adam

### momentum
# epochs = 10
#
# sess = tf.InteractiveSession()
#
# x = tf.Variable(tf.ones(1))
# y = x**2
# cost = x**2
# optimizer = tf.train.MomentumOptimizer(1e-3,0.9).minimize(cost)
#
# init = tf.global_variables_initializer()
# init.run()
#
# for i in range(epochs):
#     optimizer.run()
#     print("epoch is %.4f: x is %.4f, y is %.4f" % (i,x.eval(),y.eval()))
#
# sess.close()
# print("\n\n\nown optimizer run ")
#
#
#
# sess = tf.InteractiveSession()
#
# x = tf.Variable(tf.ones(1))
# y = x**2
# cost = x**2
# own_optimizer = Momentum(1e-3).minimize(cost)
#
# init = tf.global_variables_initializer()
# init.run()
#
# for i in range(epochs):
#     own_optimizer.run()
#     print("epoch is %.4f: x is %.4f, y is %.4f" % (i,x.eval(),y.eval()))

### adagrad

# epochs = 10
#
# sess = tf.InteractiveSession()
#
# x = tf.Variable(tf.ones(1))
# y = x**2
# cost = x**2
# optimizer = tf.train.AdagradOptimizer(1e-3,0.1).minimize(cost)
#
# init = tf.global_variables_initializer()
# init.run()
#
# for i in range(epochs):
#     optimizer.run()
#     print("epoch is %.4f: x is %.4f, y is %.4f" % (i,x.eval(),y.eval()))
#
# sess.close()
# print("\n\n\nown optimizer run ")
#
#
#
# sess = tf.InteractiveSession()
#
# x = tf.Variable(tf.ones(1))
# y = x**2
# cost = x**2
# own_optimizer = Adagrad(1e-3,0.1).minimize(cost)
#
# init = tf.global_variables_initializer()
# init.run()
#
# for i in range(epochs):
#     own_optimizer.run()
#     print("epoch is %.4f: x is %.4f, y is %.4f" % (i,x.eval(),y.eval()))


### rmsprop

# epochs = 10
#
# sess = tf.InteractiveSession()
#
# x = tf.Variable(tf.ones(1))
# y = x**2
# cost = x**2
# optimizer = tf.train.RMSPropOptimizer(1e-3,0.9,0).minimize(cost)
#
# init = tf.global_variables_initializer()
# init.run()
#
# for i in range(epochs):
#     optimizer.run()
#     print("epoch is %.4f: x is %.4f, y is %.4f" % (i,x.eval(),y.eval()))
#
# sess.close()
# print("\n\n\nown optimizer run ")
#
#
#
# sess = tf.InteractiveSession()
#
# x = tf.Variable(tf.ones(1))
# y = x**2
# cost = x**2
# own_optimizer = Rmsprop(1e-3,1,0.9).minimize(cost)
#
# init = tf.global_variables_initializer()
# init.run()
#
# for i in range(epochs):
#     own_optimizer.run()
#     print("epoch is %.4f: x is %.4f, y is %.4f" % (i,x.eval(),y.eval()))


### Adam
epochs = 10

sess = tf.InteractiveSession()

x = tf.Variable(tf.ones(1))
y = x**2
cost = x**2
optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)

init = tf.global_variables_initializer()
init.run()

for i in range(epochs):
    optimizer.run()
    print("epoch is %.4f: x is %.4f, y is %.4f" % (i,x.eval(),y.eval()))

sess.close()
print("\n\n\nown optimizer run ")



sess = tf.InteractiveSession()

x = tf.Variable(tf.ones(1))
y = x**2
cost = x**2
own_optimizer = Adam(1e-3).minimize(cost)

init = tf.global_variables_initializer()
init.run()

for i in range(epochs):
    own_optimizer.run()
    print("epoch is %.4f: x is %.4f, y is %.4f" % (i,x.eval(),y.eval()))
