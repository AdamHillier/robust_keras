import numpy as np

import keras.backend as K
from keras.utils import Sequence
from keras.engine.training_utils import batch_shuffle, check_num_samples, make_batches
from keras.utils.generic_utils import slice_arrays

import math

class AdversarialExampleGenerator(Sequence):
    def __init__(self, model, x_set, y_set, batch_size,
                 epsilon, k, a, random_start=True, incremental=False,
                 class_weight=None, sample_weight=None,
                 shuffle=True):

        self.model = model
        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.epoch = 0
        self.incremental = incremental

        self.x, self.y, self.sample_weights = \
                    model._standardize_user_data(x_set, y_set,
                                                 sample_weight=sample_weight,
                                                 class_weight=class_weight,
                                                 batch_size=batch_size)

        self.num_samples = check_num_samples(self.x + self.y,
                                             batch_size=batch_size,
                                             steps_name="steps")

        self.length = math.ceil(self.num_samples / batch_size)
        
        self.index_array = np.arange(self.num_samples)
        
        # Set up adversary
        self.attack = LinfPGDAttack(self.model,
                                    epsilon,
                                    k,
                                    a,
                                    random_start,
                                    True)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert 0 <= index < self.length

        if index == 0:
            self.epoch += 1

            if self.shuffle == "batch":
                self.index_array = batch_shuffle(self.index_array, self.batch_size)
            elif self.shuffle:
                np.random.shuffle(self.index_array)

            batches = make_batches(self.num_samples, self.batch_size)
            self.batch_enumerator = enumerate(batches)
        
        batch_index, (batch_start, batch_end) = next(self.batch_enumerator)
        batch_ids = self.index_array[batch_start:batch_end]

        try:
            x_batch = slice_arrays(self.x, batch_ids)[0]
            y_batch = slice_arrays(self.y, batch_ids)[0]
            sample_weights_batch = slice_arrays(self.sample_weights, batch_ids)[0]
        except TypeError:
            raise TypeError('TypeError while preparing batch. '
                            'If using HDF5 input data, '
                            'pass shuffle="batch".')

        if self.incremental != False and self.incremental[0] <= self.epoch:
            x_batch_adv = self.attack.perturb(x_batch, y_batch, max((self.epoch - self.incremental[0]) / (self.incremental[1] - self.incremental[0]), 1))
        elif self.incremental == False or self.incremental[1] <= self.epoch:
            x_batch_adv = self.attack.perturb(x_batch, y_batch, 1)
        else:
            x_batch_adv = x_batch

        return x_batch_adv, y_batch, sample_weights_batch

class LinfPGDAttack:
    def __init__(self, model, epsilon, k, a, random_start, incremental = False, starting_epsilon = 0.001):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.incremental = incremental
        self.starting_epsilon = starting_epsilon

        loss = model.xent_loss

        self.grad = model.optimizer.get_gradients(loss, model.inputs[0])
        
        self.input_tensors = [model.inputs[0],
                              model.sample_weights[0],
                              model.targets[0],
                              K.learning_phase()]

        self.get_gradients = K.function(inputs=self.input_tensors, outputs=self.grad)

    def perturb(self, x_nat, y, train_frac = 1):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        """In the case of incremental PGD
           If train_frac < 0.5, interpolate between self.starting_epsilon and self.epsilon
           If train_frac > 0.5, use self.epsilon"""
        if self.incremental and train_frac < 0.5:
            epsilon = self.epsilon * 2 * train_frac + self.starting_epsilon * (1 - 2 * train_frac)
        else:
            epsilon = self.epsilon

        if self.rand:
            x = x_nat + np.random.uniform(-epsilon, epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        for i in range(self.k):
            grad = self.get_gradients([x, np.ones(len(x)), y, 0])

            x += self.a * np.sign(grad[0])

            np.clip(x, x_nat - epsilon, x_nat + epsilon, out=x) 
            np.clip(x, 0, 1, out=x)

        return x
