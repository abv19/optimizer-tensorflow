#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Adagrad.py    
@Contact :   271856330@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/17 10:23   jzh      1.0         None
'''

from tensorflow.python.training import optimizer
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
import math


class Adagrad(optimizer.Optimizer):
    def __init__(self,learning_rate = 1e-3,init_accumulator = 0.1,use_locking=False,name="Adagrade"):
        super(Adagrad,self).__init__(use_locking,name)

        self.lr = learning_rate
        self.init_accumulator = init_accumulator

        self.lr_t = None
        self.init_accumulator_t = None

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
                                               self._initial_accumulator_value)
            return math_ops.cast(init_constant, dtype)

        return init

    def _prepare(self):
        learning_rate = self._call_if_callable(self.lr)
        self.lr_t = ops.convert_to_tensor(
            learning_rate, name="learning_rate")

    def _apply_dense(self, grad, var):
        acc = self.get_slot(var, "accumulator")
        lr = math_ops.cast(self.lr_t, var.dtype.base_dtype)
        accumulator = grad**2 + acc
        acc_update = acc.assign(accumulator)
        var_update = state_ops.assign_sub(var,1 / math_ops.sqrt(accumulator) * grad * lr)
        return control_flow_ops.group(*[acc_update, var_update])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")