from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from jax.example_libraries import stax
from jax.example_libraries import optimizers
from functools import partial
from tqdm import tqdm 
from torch.utils.data import DataLoader
from decimal import Decimal

from utils import normalize_dp

class NNLearner(object):
    """NNLearner is a class which allows the creation of objects
    containing a neural network model and an associated trainer
    which facillitates the training of the model given training
    data, test performance, etc.
    
    Intended to help abstract the learning part of the scripts whilst keeping
    reproducibility and flexibility high
    """
    def __init__(self, 
        train_dataset, 
        test_dataset, 
        input_shape=4, 
        stax=None,
        lagrangian=None, 
        loss=None, 
        optimizer="adam", 
        optimizer_parameters=lambda t: jnp.select([t < 1000], [1e-3, 3e-4]), 
        nn_output_modifier=None,
        h=None,
        dof=None, 
        weight_loss=None, 
        weight_cond=None,
        weight_degeneracy=None, 
        base_point_tripple=None,
        ll_normalise_func=None):

        super(NNLearner, self).__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lagrangian = lagrangian
        self.loss = loss
        self.mode = loss
        self.nn_output_modifier = nn_output_modifier
        self.h = h
        self.dof = dof
        self.weight_loss = weight_loss
        self.weight_cond = weight_cond
        self.base_point_tripple = base_point_tripple
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters
        self.stax = stax
        self.ll_normalise_func = ll_normalise_func
        self.weight_degeneracy = weight_degeneracy

        self.input_shape = input_shape
        self.params = None # Will be the current parameters of the network

        # Setup network and optimizers
        self._initial_network_setup()
        self._initial_optimizer_setup()

        self.mse_losses = []
        self.non_triv_values = []
        self.degeneracy_values = []

        self.test_mse_losses = []
        self.test_non_triv_values = []
        self.test_degeneracy_values = []

    def _initial_network_setup(self):
        """Specify functional model (architecture) of the neural network
        followed by initialising the initial parameters of this network
        """
        if self.stax is None:  
            init_fun, self.nn_forward_fn = stax.serial(
                stax.Dense(128),
                stax.Softplus,
                stax.Dense(128),
                stax.Softplus,
                stax.Dense(1),
            )
            rng = jax.random.PRNGKey(0) # for reproducibility
            self.output_shape, self.init_params = init_fun(rng, input_shape=(-1, self.input_shape))
        else:
            rng = jax.random.PRNGKey(0) # for reproducibility
            init_fun, self.nn_forward_fn = self.stax
            self.output_shape, self.init_params = init_fun(rng, input_shape=(-1, self.input_shape))

    def _initial_optimizer_setup(self):
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(self.optimizer_parameters)
        self.opt_state = self.opt_init(self.init_params)

    def learned_lagrangian(self, params):
        # Returns a function representing the same signature as the lagrangian
        # which passes the inputs into the nn model. Behaviour will change based
        # on mode used
        if self.mode == "xdot" or self.mode == "baseline":
            def nn_lagrangian(q, q_t):
                state = jnp.concatenate([q, q_t]) # change to radians only in double_pendulum
                out = self.nn_forward_fn(params, state)
                return jnp.squeeze(out) # there is a trailing dimension to squeeze
        else:
            if self.ll_normalise_func is None:
                def nn_lagrangian(q, q_t):
                    state = jnp.concatenate([q, q_t])
                    out = self.nn_forward_fn(params, state)
                    return jnp.squeeze(out, axis=-1)
            else:
                def nn_lagrangian(q, q_t):
                    state = self.ll_normalise_func(jnp.concatenate([q, q_t]))
                    out =  self.nn_forward_fn(params, state)
                    return jnp.squeeze(out, axis=-1)

        return nn_lagrangian

    @partial(jax.jit, static_argnums=(0,))
    def make_prediction(self, params, batch):
        x, y = batch
        
        if self.mode == "xdot":
            preds = jax.vmap(partial(self.nn_output_modifier, self.learned_lagrangian(params)))(x)
            

        elif self.mode == "tripple_Ld":
            def DEOM_SVI(state):
                q_1k = state[0]
                q_k = state[1]
                q_k1 = state[2]
                q_1k = q_1k[:self.dof]
                q_k = q_k[:self.dof]
                q_k1 = q_k1[:self.dof]

                def Ldk(qk, qk_1):
                    return self.learned_lagrangian(params)(qk, qk_1)
                def D2Ld1k(q_1k, q_k): 
                    return jax.grad(Ldk, argnums=1)(q_1k, q_k)
                def D1Ldk(q_k, q_k1):  
                    return jax.grad(Ldk, argnums=0)(q_k, q_k1)

                result_prep = D2Ld1k(q_1k, q_k) + D1Ldk(q_k, q_k1) 
                return jnp.dot(result_prep, result_prep) 

            preds = jax.vmap(DEOM_SVI)(x)

        else:
            raise ValueError("self.mode is not valid value, value was {}".format(self.mode))

        return preds

    @partial(jax.jit, static_argnums=(0,))
    def compute_degeneracy(self, params, batch):
        x, y = batch
          
        if self.mode == "tripple_Ld":
            def jac_DEOM(state):
                q_k = state[1]
                q_k1 = state[2]
                q_k = q_k[:self.dof]
                q_k1 = q_k1[:self.dof]

                def Ldk(qk, qk_1):
                    return self.learned_lagrangian(params)(qk, qk_1)
                def D1Ldk(q_k, q_k1):  
                    return jax.grad(Ldk, argnums=0)(q_k, q_k1)
                def D2_D1Ldk(q_k, q_k1):
                    return jax.jacrev(D1Ldk, argnums=1)(q_k, q_k1)
                
                det_jacobian = (jax.scipy.linalg.det(D2_D1Ldk(q_k, q_k1))) 
                logistic_minused = -(1/(1+jnp.exp((-0.01*det_jacobian)))-1)
                return logistic_minused 
            
            pred_degeneracy = jax.vmap(jac_DEOM)(x)
        else:
            raise ValueError("input mode is not implemented")

        return pred_degeneracy

    @partial(jax.jit, static_argnums=(0,))
    def loss_func(self, params, batch, time_step=None):
        x, y = batch
        preds = self.make_prediction(params, batch)

        # Take on any custom loss function or default to MSE
        if self.loss == "xdot" or self.loss is None:
            return jnp.mean((preds - y) ** 2)

        elif self.loss == "tripple_Ld":
            pred_degeneracy = self.compute_degeneracy(params, batch)

            weight_loss = self.weight_loss
            weight_cond = self.weight_cond
            base_tripple = self.base_point_tripple
            weight_degeneracy = self.weight_degeneracy
            
            def non_triv_cond(params): #in order to make it vmappable with the other list created from DEOM_SVI
                qk_base = base_tripple[0]
                qk1_base = base_tripple[1]
                p_base = base_tripple[2]

                p0 = -jax.grad(self.learned_lagrangian(params), argnums=0)(qk_base, qk1_base)
                return jnp.sum((p0 - p_base) ** 2)
            
            mse_component = weight_loss * jnp.mean(preds)
            non_triv_component = (weight_cond * (non_triv_cond(params)))
            degeneracy_component = (weight_degeneracy * jnp.mean(pred_degeneracy))              
            return mse_component + non_triv_component + degeneracy_component 
        else:
            return ValueError("Input loss option does not exist")

    @partial(jax.jit, static_argnums=(0,))
    def loss_func_components(self, params, batch, time_step=None):
        x, y = batch
        preds = self.make_prediction(params, batch)
        pred_degeneracy = self.compute_degeneracy(params, batch)

        # Take on any custom loss function or default to MSE
        if self.loss is None:
            return jnp.mean((preds - y) ** 2)

        elif self.loss == "tripple_Ld":    
            weight_loss = self.weight_loss
            weight_cond = self.weight_cond
            base_tripple = self.base_point_tripple
            weight_degeneracy = self.weight_degeneracy
            
            def non_triv_cond(params): #in order to make it vmappable with the other list created from DEOM_SVI
                qk_base = base_tripple[0]
                qk1_base = base_tripple[1]
                p_base = base_tripple[2]

                p0 = -jax.grad(self.learned_lagrangian(params), argnums=0)(qk_base, qk1_base)
                return jnp.sum((p0 - p_base) ** 2)
            
            mse_component = weight_loss * jnp.mean(preds)
            non_triv_component = (weight_cond * (non_triv_cond(params)))
            degeneracy_component = (weight_degeneracy * jnp.mean(pred_degeneracy))              
            return (mse_component + non_triv_component + degeneracy_component), mse_component, non_triv_component, degeneracy_component  
            #+ (0.001/len(preds))*(optimizers.l2_norm(params))

        else:
            return ValueError("Input loss option does not exist")

    def fit2(self, num_epochs=150000, test_every=100):
        """Gradient descent with all observations per step
        """
        print("## Learning the lagrangian")
        training_data = (self.train_dataset.npx, self.train_dataset.npy)
        test_data = (self.test_dataset.npx, self.test_dataset.npy)
        
        opt_init, opt_update, get_params = optimizers.adam(self.optimizer_parameters)
        opt_state = opt_init(self.init_params)

        @jax.jit
        def update_derivative(i, opt_state, batch):
            params = get_params(opt_state)
            return opt_update(i, jax.grad(self.loss_func)(params, batch), opt_state)

        train_losses = []
        test_losses = []

        for epoch in tqdm(range(num_epochs), "Epochs progress"):
            if epoch % test_every == 0:
                params = get_params(opt_state)
                train_loss, train_mse, train_nontriv, train_degen = self.loss_func_components(params, training_data)
                train_losses.append(train_loss)
                test_loss, test_mse, test_nontriv, test_degen = self.loss_func_components(params, test_data)
                test_losses.append(test_loss)
                self.mse_losses.append(train_mse)
                self.non_triv_values.append(train_nontriv)
                self.degeneracy_values.append(train_degen)

                self.test_mse_losses.append(test_mse)
                self.test_non_triv_values.append(test_nontriv)
                self.test_degeneracy_values.append(test_degen)

            opt_state = update_derivative(epoch, opt_state, training_data)
        
        params = get_params(opt_state)

        self.train_losses = train_losses
        self.test_losses = test_losses
        self.params = params

        return params, train_losses, test_losses

    def fit3(self, num_epochs=150000, test_every=100, batch_size=2048):
        """Stochastic minibatch gradient descent
        """
        print("## Learning the lagrangian")
        training_data = (self.train_dataset.npx, self.train_dataset.npy)
        training_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle=True)
        test_data = (self.test_dataset.npx, self.test_dataset.npy)
        
        opt_init, opt_update, get_params = optimizers.adam(self.optimizer_parameters)
        opt_state = opt_init(self.init_params)

        @jax.jit
        def update_derivative(i, opt_state, batch):
            params = get_params(opt_state)
            return opt_update(i, jax.grad(self.loss_func)(params, batch), opt_state)

        train_losses = []
        test_losses = []

        for epoch in tqdm(range(num_epochs), "Epochs progress"):
            if epoch % test_every == 0:
                params = get_params(opt_state)
                train_loss, train_mse, train_nontriv, train_degen = self.loss_func_components(params, training_data)
                train_losses.append(train_loss)
                test_loss, test_mse, test_nontriv, test_degen = self.loss_func_components(params, test_data)
                test_losses.append(test_loss)
                self.mse_losses.append(train_mse)
                self.non_triv_values.append(train_nontriv)
                self.degeneracy_values.append(train_degen)

                self.test_mse_losses.append(test_mse)
                self.test_non_triv_values.append(test_nontriv)
                self.test_degeneracy_values.append(test_degen)


            for batch in training_loader:
                batchx, batchy = batch
                opt_state = update_derivative(epoch, opt_state, (batchx.numpy(), batchy.numpy()))
        
        params = get_params(opt_state)

        self.train_losses = train_losses
        self.test_losses = test_losses
        self.params = params

        return params, train_losses, test_losses

    def fit_lnn(self, num_epochs=150000, test_every=100):
        """Fitting function for LNN models, as in original paper.
        """
        print("## Learning the lagrangian")
        training_data = (self.train_dataset.npx, self.train_dataset.npy)
        test_data = (self.test_dataset.npx, self.test_dataset.npy)
        
        opt_init, opt_update, get_params = optimizers.adam(self.optimizer_parameters)
        opt_state = opt_init(self.init_params)

        @jax.jit
        def update_derivative(i, opt_state, batch):
            params = get_params(opt_state)
            return opt_update(i, jax.grad(self.loss_func)(params, batch), opt_state)

        train_losses = []
        test_losses = []

        for epoch in tqdm(range(num_epochs), "Epochs progress"):
            if epoch % test_every == 0:
                params = get_params(opt_state)
                train_loss = self.loss_func(params, training_data)
                train_losses.append(train_loss)
                test_loss = self.loss_func(params, test_data)
                test_losses.append(test_loss)

            opt_state = update_derivative(epoch, opt_state, training_data)
        
        params = get_params(opt_state)

        self.train_losses = train_losses
        self.test_losses = test_losses
        self.params = params

        return params, train_losses, test_losses