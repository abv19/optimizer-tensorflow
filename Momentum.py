#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Momentum.py    
@Contact :   271856330@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/16 14:23   jzh      1.0         None
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export


class Momentum(optimizer.Optimizer):
    def __init__(self,learning_rate = 0.001,momentum = 0.9,use_locking=False, name="Momentum"):
        super(Momentum, self).__init__(use_locking, name)
        self.lr = learning_rate
        self.momentum = momentum

        self.lr_t = None
        self.momentum_t = None

    def _create_slots(self,var_list):
        for v in var_list:
            self._zeros_slot(v,'momentum',self._name)

    def _prepare(self):
        lr = self.lr
        if callable(lr):
            lr = lr()
        self.lr_t = ops.convert_to_tensor(lr)

        momentum = self.momentum
        if callable(momentum):
            momentum = momentum()
        self.momentum_t = ops.convert_to_tensor(momentum)

    def _apply_dense(self, grad, var):
        momentum = self.get_slot(var,'momentum')
        lr = math_ops.cast(self.lr_t,var.dtype.base_dtype)
        gamma = math_ops.cast(self.momentum_t,var.dtype.base_dtype)
        v = gamma * momentum + lr * grad
        update_m = momentum.assign(v)
        update_var = state_ops.assign_sub(var,v)
        return control_flow_ops.group(*[update_var, update_m])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")