"""
Module containing a learner class which encapsulates a 
model and the associated training process
"""
from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
#from jax import random
import numpy as np
from jax.example_libraries import stax
from jax.example_libraries import optimizers
from functools import partial 
#from decimal import Decimal
from utils import normalize_dp
from tqdm import tqdm
from torch.utils.data import DataLoader

class NNLearnerSymmetric(object):
    """Learner is a class which allows the creation of objects
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
        output_shape=1, 
        stax=None,
        symmetry_params=None,
        testpoints_symmetry = None,
        lagrangian=None, 
        loss=None, 
        optimizer="adam", 
        optimizer_parameters=lambda t: jnp.select([t>0], [3e-3]), 
        nn_output_modifier=None,
        h=None,
        dof=None, 
        weight_loss = None, 
        weight_cond = None,
        weight_degeneracy = None, 
        base_point_tripple = None,
        ll_normalise_func = None,
        symmetry_weight = 1.0):

        super(NNLearnerSymmetric, self).__init__()
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
        self.symmetry_weight = float(symmetry_weight)

        self.input_shape = input_shape
        self.output_shape = output_shape # its supposed to be the lagrangian output
        self.params = None # Will be the current parameters of the network
        self.symmetry_params = symmetry_params # Will be the symmetry parameters
        self.testpoints_symmetry = testpoints_symmetry # points in phase space where symmetry is tested
        self.weight_degeneracy = weight_degeneracy

        # Setup network and optimizers
        self._initial_network_setup()
        self._initial_optimizer_setup()

        # For plotting
        self.mse_losses = []
        self.non_triv_values = []
        self.degeneracy_values = []
        self.symLosses = []
        self.normSyms = []
        self.nn_losses = []
        self.sym_losses = []

        self.symParam0_list = []
        self.symParam1_list = []

        self.test_mse_losses = []
        self.test_non_triv_values = []
        self.test_degeneracy_values = []
        self.test_symLosses = []
        self.test_normSyms = []
        self.test_nn_losses = []
        self.test_sym_losses = []


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
                stax.Dense(self.output_shape),
            )
            rng = jax.random.PRNGKey(0) # for reproducibility
            self.output_shape, self.init_params = init_fun(rng, input_shape=(-1, self.input_shape))
        else:
            rng = jax.random.PRNGKey(0) # for reproducibility
            init_fun, self.nn_forward_fn = self.stax
            self.output_shape, self.init_params = init_fun(rng, input_shape=(-1, self.input_shape))

    # These should be instantiated inside of the fit2 method
    def _initial_optimizer_setup(self):
        if self.optimizer == "adam":
            parameters = {"nn": self.init_params, "sym": self.symmetry_params}
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(self.optimizer_parameters) # adam w learn rate decay
            self.opt_state = self.opt_init(parameters)
        else:
            parameters = {"nn": self.init_params, "sym": self.symmetry_params}
            self.opt_init, self.opt_update, self.get_params = optimizers.sgd(1e-4) # standard sgd
            self.opt_state = self.opt_init(parameters)

    def learned_lagrangian(self, params):
        # Returns a function representing the same signature as the lagrangian
        # which passes the inputs into the nn model. Behaviour will change based
        # on mode
        if self.ll_normalise_func is None:
            def nn_lagrangian(q, q_t):
                state = jnp.concatenate([q, q_t])
                out =  self.nn_forward_fn(params, state)
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
        
        if self.mode == "tripple_Ld":
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
                # return jnp.dot(result_prep, result_prep)
                return result_prep**2
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
        pred_degeneracy = self.compute_degeneracy(params, batch)

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
            return mse_component + non_triv_component + degeneracy_component

        else:
            return ValueError("Input loss option does not exist")

    @partial(jax.jit, static_argnums=(0,))
    def loss_func_components(self, params, batch, time_step=None):
        x, y = batch
        preds = self.make_prediction(params, batch)
        pred_degeneracy = self.compute_degeneracy(params, batch)

        if self.loss == "tripple_Ld":    
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
            
            mse_component = weight_loss* jnp.mean(preds)
            non_triv_component = (weight_cond * (non_triv_cond(params)))
            degeneracy_component = (weight_degeneracy * jnp.mean(pred_degeneracy))              
            return (mse_component + non_triv_component + degeneracy_component), mse_component, non_triv_component, degeneracy_component  

        else:
            return ValueError("Input loss option does not exist")

    def loss_func_translation_symmetry(self, params, paramsSym):
        """loss function to learn direction of translation symmetry
        params: parameters for self.learned_lagrangian(params)
        paramsSym: parameters definining the Lie algebra element of the symmetry action
        """
        
        weights = jnp.array([1.,1.])	
        
        def Ld(state):
            split = np.split(state,2)
            q1 = split[0]
            q2 = split[1]
            return self.learned_lagrangian(params)(q1,q2)
				
        symCheck = lambda state: jnp.dot(jnp.concatenate([paramsSym,paramsSym]),jax.grad(Ld)(state))        # check translation symmetry at point state
        symCheckv = jax.vmap(symCheck)
        symloss = jnp.sum(symCheckv(self.testpoints_symmetry)) 
        symloss = symloss/min(1,len(self.testpoints_symmetry))
        normSym = jnp.abs(jnp.linalg.norm(paramsSym)-1.)
			
        return weights[0]*symloss + weights[1]*normSym

    def loss_func_translation_symmetry_components(self, params, paramsSym):
        """loss function to learn direction of translation symmetry
        params: parameters for self.learned_lagrangian(params)
        paramsSym: parameters definining the Lie algebra element of the symmetry action
        """
        
        def Ld(state):
            split = np.split(state,2)
            q1 = split[0]
            q2 = split[1]
            return self.learned_lagrangian(params)(q1,q2)
                
        symCheck = lambda state: jnp.dot(jnp.concatenate([paramsSym,paramsSym]),jax.grad(Ld)(state))        # check translation symmetry at point state
        symCheckv = jax.vmap(symCheck)
        symloss = jnp.sum(symCheckv(self.testpoints_symmetry)) 
        symloss = symloss/min(1,len(self.testpoints_symmetry))
        normSym = jnp.abs(jnp.linalg.norm(paramsSym)-1.)
            
        return self.symmetry_weight*(symloss + normSym), self.symmetry_weight*symloss, self.symmetry_weight*normSym


    # @partial(jax.jit, static_argnums=(0,))
    def loss_func_joint(self, params_collection, batch, time_step=None):
        params = params_collection["nn"]
        paramsSym = params_collection["sym"]
        return self.loss_func(params, batch, time_step) + (self.symmetry_weight * self.loss_func_translation_symmetry(params, paramsSym))


    def loss_func_just_nn(self, params_collection, batch, time_step=None):
        params = params_collection["nn"]
        return self.loss_func(params, batch, time_step)

    def loss_func_joint_components(self, params_collection, batch, time_step=None):
        params = params_collection["nn"]
        paramsSym = params_collection["sym"]
        full_sym_loss, symloss, normSym = self.loss_func_translation_symmetry_components(params, paramsSym)
        full_nn_loss, mse, non_triv, degen = self.loss_func_components(params, batch)
        full_loss = full_nn_loss + full_sym_loss
        return full_loss, full_nn_loss, full_sym_loss, mse, degen, symloss, normSym


    def get_symmetry_values(self, params_collection):
        symmetry_params = np.array(params_collection["sym"])
        return symmetry_params[0], symmetry_params[1]


    def fit2(self, num_epochs=150000, test_every=100, symmetry_dim=0):
        """
        More safe optimisation routine for jit
        """
        print("## Learning the Lagrangian")
        training_data = (self.train_dataset.npx, self.train_dataset.npy)
        test_data = (self.test_dataset.npx, self.test_dataset.npy)
        
        # The key difference is that we want to mutate opt_state
        parameters = {"nn": self.init_params, "sym": self.symmetry_params}
        opt_init, opt_update, get_params = optimizers.adam(self.optimizer_parameters)
        opt_state = opt_init(parameters)

        @jax.jit
        def update_derivative(i, opt_state, batch):
            params = get_params(opt_state)
            return opt_update(i, jax.grad(self.loss_func_joint)(params, training_data), opt_state)

        train_losses = []
        test_losses = []

        for epoch in tqdm(range(num_epochs), "Epochs progress"):
            if epoch % test_every == 0:
                params = get_params(opt_state)

                train_loss, train_full_nn_loss, train_full_sym_loss, train_mse, train_degen, train_symloss, train_normSym = self.loss_func_joint_components(params, training_data)
                symParam0, symParam1 = self.get_symmetry_values(params)

                train_losses.append(train_loss)
                self.mse_losses.append(train_mse)
                self.non_triv_values.append(0)
                self.degeneracy_values.append(train_degen)
                self.symLosses.append(train_symloss)
                self.normSyms.append(train_normSym)
                self.nn_losses.append(train_full_nn_loss)
                self.sym_losses.append(train_full_sym_loss)
                self.symParam0_list.append(symParam0)
                self.symParam1_list.append(symParam1)

                test_loss, test_full_nn_loss, test_full_sym_loss, test_mse, test_degen, test_symloss, test_normSym = self.loss_func_joint_components(params, test_data)

                test_losses.append(train_loss)
                self.test_mse_losses.append(train_mse)
                self.test_non_triv_values.append(0)
                self.test_degeneracy_values.append(train_degen)
                self.test_symLosses.append(train_symloss)
                self.test_normSyms.append(train_normSym)
                self.test_nn_losses.append(train_full_nn_loss)
                self.test_sym_losses.append(train_full_sym_loss)


            opt_state = update_derivative(epoch, opt_state, training_data)

        params = get_params(opt_state)
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.params = params

        return params, train_losses, test_losses

    def fit_switch(self, num_epochs=150000, test_every=100, symmetry_dim=0):
        """
        This routine trains the model with and without the symmetry parameters 
        for a specified set of epochs
        """
        epoch_switch_point = int(num_epochs/4)
        epoch_switch_point = 5000

        print("## Learning the Lagrangian")
        training_data = (self.train_dataset.npx, self.train_dataset.npy)
        test_data = (self.test_dataset.npx, self.test_dataset.npy)
        
        # The key difference is that we want to mutate opt_state
        parameters = {"nn": self.init_params, "sym": self.symmetry_params}
        opt_init, opt_update, get_params = optimizers.adam(self.optimizer_parameters)
        opt_state = opt_init(parameters)

        @jax.jit
        def update_derivative_just_nn(i, opt_state, batch):
            params = get_params(opt_state)
            return opt_update(i, jax.grad(self.loss_func_just_nn)(params, training_data), opt_state)

        @jax.jit
        def update_derivative_joint(i, opt_state, batch):
            params = get_params(opt_state)
            return opt_update(i, jax.grad(self.loss_func_joint)(params, training_data), opt_state)

        train_losses = []
        test_losses = []

        for epoch in tqdm(range(num_epochs), "Epochs progress"):
            if epoch % test_every == 0:
                params = get_params(opt_state)

                train_loss, train_full_nn_loss, train_full_sym_loss, train_mse, train_degen, train_symloss, train_normSym = self.loss_func_joint_components(params, training_data)
                symParam0, symParam1 = self.get_symmetry_values(params)

                train_losses.append(train_loss)
                self.mse_losses.append(train_mse)
                self.non_triv_values.append(0)
                self.degeneracy_values.append(train_degen)
                self.symLosses.append(train_symloss)
                self.normSyms.append(train_normSym)
                self.nn_losses.append(train_full_nn_loss)
                self.sym_losses.append(train_full_sym_loss)
                self.symParam0_list.append(symParam0)
                self.symParam1_list.append(symParam1)

                test_loss, test_full_nn_loss, test_full_sym_loss, test_mse, test_degen, test_symloss, test_normSym = self.loss_func_joint_components(params, test_data)

                test_losses.append(train_loss)
                self.test_mse_losses.append(train_mse)
                self.test_non_triv_values.append(0)
                self.test_degeneracy_values.append(train_degen)
                self.test_symLosses.append(train_symloss)
                self.test_normSyms.append(train_normSym)
                self.test_nn_losses.append(train_full_nn_loss)
                self.test_sym_losses.append(train_full_sym_loss)

            if epoch < epoch_switch_point:
                opt_state = update_derivative_just_nn(epoch, opt_state, training_data)
            else:
                opt_state = update_derivative_joint(epoch, opt_state, training_data)

        params = get_params(opt_state)
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.params = params

        return params, train_losses, test_losses

    def fit3(self, num_epochs=150000, test_every=100, symmetry_dim=0, batch_size=256):
        """
        More safe optimisation routine for jit
        """
        print("## Learning the Lagrangian")
        training_data = (self.train_dataset.npx, self.train_dataset.npy)
        training_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle=True)
        test_data = (self.test_dataset.npx, self.test_dataset.npy)
        
        # The key difference is that we want to mutate opt_state
        parameters = {"nn": self.init_params, "sym": self.symmetry_params}
        opt_init, opt_update, get_params = optimizers.adam(self.optimizer_parameters)
        opt_state = opt_init(parameters)

        @jax.jit
        def update_derivative(i, opt_state, batch):
            params = get_params(opt_state)
            return opt_update(i, jax.grad(self.loss_func_joint)(params, training_data), opt_state)

        train_losses = []
        test_losses = []

        for epoch in tqdm(range(num_epochs), "Epochs progress"):
            if epoch % test_every == 0:
                params = get_params(opt_state)
                train_loss, train_mse, train_nontriv, train_degen = self.loss_func_components(params['nn'], training_data)
                train_losses.append(train_loss)
                test_loss, test_mse, test_nontriv, test_degen = self.loss_func_components(params['nn'], test_data)
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


class NNLearnerSymmetricExtended(NNLearnerSymmetric):
    """Extension of NNLearnerSymmetric with the symmetry term for
    discovery of affine symmetries.
    """
    def __init__(self, 
        train_dataset, 
        test_dataset, 
        input_shape=4,
        output_shape=1, 
        stax=None,
        symmetry_params=None,
        testpoints_symmetry = None,
        lagrangian=None, 
        loss=None, 
        optimizer="adam", 
        optimizer_parameters=lambda t: jnp.select([t < 1000, t < 2000, t > 3000], [1e-4, 3e-5, 1e-5]), 
        nn_output_modifier=None,
        h=None,
        dof=None, 
        weight_loss = None, 
        weight_cond = None,
        weight_degeneracy = None, 
        base_point_tripple = None,
        ll_normalise_func = None,
        symmetry_weight = 1.0):

        super(NNLearnerSymmetricExtended, self).__init__(train_dataset, 
            test_dataset, 
            input_shape,
            output_shape, 
            stax,
            symmetry_params,
            testpoints_symmetry, 
            lagrangian, 
            loss,
            optimizer, 
            optimizer_parameters, 
            nn_output_modifier,
            h,
            dof, 
            weight_loss,
            weight_cond,
            weight_degeneracy, 
            base_point_tripple,
            ll_normalise_func,
            symmetry_weight)

    def loss_func_translation_symmetry(self, params, paramsSym):
        """loss function to learn affine linear symmetry
        params: parameters for self.learned_lagrangian(params)
        paramsSym: parameters definining the Lie algebra element of the symmetry action, array with 6 components
        """

        w = paramsSym[:2] # identified translation vector
        M = paramsSym[2:].reshape(2,2)
        # when Ld is learned

        weights = jnp.array([1.,1.])    
            
        Ld = lambda q0,q1: self.learned_lagrangian(params)(q0,q1)
    
        def symCheck(state):
            q0,q1 = jnp.split(state,2)
            return jnp.dot(M @ q0 + w, jax.grad(Ld,0)(q0,q1)) + jnp.dot(M @ q1 + w,jax.grad(Ld,1)(q0,q1)) # check symmetry condition at state
            
        symCheckv = jax.vmap(symCheck)
        symloss = jnp.sum(jnp.square(symCheckv(self.testpoints_symmetry))) 
        # symloss = symloss/len(self.testpoints_symmetry) # normalise by number of testpoint symmetries
        
        normSym = jnp.abs(jnp.linalg.norm(paramsSym)-1.)
        # normSym = jnp.square(jnp.linalg.norm(paramsSym)-1.)
                    
        return (symloss) + (normSym)

    def loss_func_translation_symmetry_components(self, params, paramsSym):
        """loss function to learn direction of translation symmetry
        params: parameters for self.learned_lagrangian(params)
        paramsSym: parameters definining the Lie algebra element of the symmetry action
        """
        if self.mode == "tripple_Lc": # when Lc is learned
            raise ValueError("Not implemented for tripple_LC")
            
        else: # when Ld is learned
            w = paramsSym[:2] # identified translation vector
            M = paramsSym[2:].reshape(2,2)
            weights = jnp.array([1.,1.])    
                
            Ld = lambda q0,q1: self.learned_lagrangian(params)(q0,q1)
                
            def symCheck(state):
                q0,q1 = jnp.split(state,2)
                return jnp.dot(M @ q0 + w,jax.grad(Ld,0)(q0,q1)) + jnp.dot(M @ q1 + w,jax.grad(Ld,1)(q0,q1)) # check symmetry condition at state
        
            symCheckv = jax.vmap(symCheck)
            symloss = jnp.sum(jnp.square(symCheckv(self.testpoints_symmetry)))
            symloss = symloss/min(1,len(self.testpoints_symmetry))
                
            normSym = jnp.abs(jnp.linalg.norm(paramsSym)-1.)
                        
            return self.symmetry_weight*(symloss + normSym), self.symmetry_weight*symloss, self.symmetry_weight*normSym
