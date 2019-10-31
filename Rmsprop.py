#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Rmsprop.py    
@Contact :   271856330@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/17 11:01   jzh      1.0         None
'''

from tensorflow.python.ops import math_ops,init_ops,array_ops,gen_array_ops,state_ops,control_flow_ops
from tensorflow.python.training import optimizer
from tensorflow.python.framework import ops

class Rmsprop(optimizer.Optimizer):
    def __init__(self,learning_rate = 1e-3,init_accumulator = 1.0,alpha = 0.9,use_locking=False,name = "Rmsprop"):
        super(Rmsprop,self).__init__(use_locking,name)

        self.lr = learning_rate
        self.alpha = alpha
        self.init_accumulator = init_accumulator

        self.lr_t = None
        self.alpha_t = None

    def _prepare(self):
        lr = self._call_if_callable(self.lr)
        self.lr_t = ops.convert_to_tensor(lr,name="learning_rate")
        self.alpha_t = ops.convert_to_tensor(self.alpha, name = "alpha")

    def _create_slots(self, var_list):
        for v in var_list:
            dtype = v.dtype.base_dtype
            if v.get_shape().is_fully_defined():
                init = init_ops.constant_initializer(self.init_accumulator,
                                                     dtype=dtype)
            else:
                init = self._init_constant_op(v, dtype)
            self._get_or_make_slot_with_initializer(v, init, v.get_shape(), dtype,
                                                    "accumulator", self._name)


    def _init_constant_op(self, v, dtype):
        def init():
            # Use a Tensor instead of initializer if variable does not have
            # static shape.
            init_constant = gen_array_ops.fill(array_ops.shape(v),
                                               self.init_accumulator)
            return math_ops.cast(init_constant, dtype)

        return init

    def _apply_dense(self, grad, var):
        acc = self.get_slot(var,"accumulator")
        dtype = var.dtype.base_dtype
        lr = math_ops.cast(self.lr_t,dtype)
        alpha = math_ops.cast(self.alpha,dtype)
        accumulator = (1- alpha) * grad ** 2 + alpha * acc #  (grad ** 2 - acc) * (1 - alpha)
        update_acc = acc.assign(accumulator)
        upadate_var = state_ops.assign_sub(var,lr * 1 / math_ops.sqrt(accumulator + 1e-10 ) * grad)
        return control_flow_ops.group(*[update_acc, upadate_var])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")