#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Adam.py    
@Contact :   271856330@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/31 9:40   jzh      1.0         None
'''

from tensorflow.python.training import optimizer
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops

class Adam(optimizer.Optimizer):
    def __init__(self,lr=1e-3,beta1=0.9,beta2=0.999,epsilon = 1e-8,use_locking=False,name = "Adam"):
        super(Adam,self).__init__(use_locking,name)
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(
            initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _get_beta_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return (self._get_non_slot_variable("beta1_power", graph=graph),
                    self._get_non_slot_variable("beta2_power", graph=graph))

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        epsilon = self._call_if_callable(self._epsilon)

        self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta1_power, beta2_power = self._get_beta_accumulators()

        lr = math_ops.cast(self._lr_t,var.dtype.base_dtype)
        beta1 = math_ops.cast(self._beta1_t,var.dtype.base_dtype)
        beta2 = math_ops.cast(self._beta2_t,var.dtype.base_dtype)
        epsilon = math_ops.cast(self._epsilon_t,var.dtype.base_dtype)

        m1 = beta1 * m + (1-beta1) * grad
        v1 = beta2 * v + (1-beta2) * grad ** 2

        update_m = m.assign(m1)
        update_v = v.assign(v1)

        update_var = state_ops.assign_sub(var, lr * math_ops.sqrt(1 - beta2_power) / (1-beta1_power) * m1 / (math_ops.sqrt(v1) + epsilon))
        return control_flow_ops.group(*[update_var, update_m,update_v])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            beta1_power, beta2_power = self._get_beta_accumulators()
            with ops.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(
                    beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(
                    beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(
            *update_ops + [update_beta1, update_beta2], name=name_scope)

