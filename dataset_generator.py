"""Module containing classes related to generating datasets with different
properties such as length and number of trajectories, collation_fns for creating different
observation types.

Note on terminology: an observation is the input the model will see and is associated 
with a single target typically represented in tuple form (observation, target) such 
that the entire dataset is the set of (observation, target) pairs available 

eg. in LNN the dataset is all the (x, xdot) pairs in a trajectory
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
from functools import partial 

# PyTorch Dataset Utilities
import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    """A dataset created out of a given trajectory and desired observation-target
    setup

    Args:
        input_trajectory (iterable): commonly a list/ndarray of states denoting input
            trajectory
        mode (str): string describing the way we want our observation-target pairs
            values can be one of 'xdot', 'next', 'momentum', 'triple'
        eom (function): equation of motion, aka f_analytical for LNN dataset
        lagrangian (function): function of the lagrangian used for momentum dataset
    """
    def __init__(self, input_trajectory, mode="xdot", eom=None, lagrangian=None, np_mode=False):
        super(TrajectoryDataset, self).__init__()
        self.input_trajectory = input_trajectory
        self.mode = mode


        # Optional args
        self.eom = eom
        self.lagrangian = lagrangian
        self.np_mode = np_mode

        # Create targets based on different desired modes
        if mode == "xdot":
            # For every state x in the trajectory there is a corresponding
            # xdot (ie, a pair of (q,qdot) -> (qdot, qdotdot) as in the LNN paper)
            if eom is None:
                raise ValueError('Missing valid eom parameter')
            else:
                x = input_trajectory
                y = jax.device_get(jax.vmap(eom)(input_trajectory))
                self.npx = np.array(x)
                self.npy = np.array(y)
                self.x = torch.from_numpy(self.npx)
                self.y = torch.from_numpy(self.npy)
                
        elif mode == "tripple_Lc" or mode == "tripple_Ld" :
            x, y = target_zero(input_trajectory)
            self.npx = np.array(x)
            self.npy = np.array(y)
            self.x = torch.from_numpy(np.array(x))
            self.y = torch.from_numpy(np.array(y))
        else:
            raise ValueError('Parameter for target argument is not "xdot", "next", "momentum_Lc","momentum_Ld", "tripple_Lc","tripple_Lc", "momentum_Lc_pintraj", "momentum_Ld_pintraj" value given was {}'.format(target))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.np_mode:
            return self.npx[idx], self.npy[idx]
        else:
            return self.x[idx], self.y[idx]


class MultiTrajectoryDataset(Dataset):
    """A dataset created out of multiple trajectories

    Args:
        input_trajectories (list): a list of list/ndarray of states denoting separate
            input trajectories
        mode (str): string describing the way we want our observation-target pairs
            values can be one of 'xdot', 'next', 'momentum', 'triple'
        eom (function): equation of motion, aka f_analytical for LNN dataset
        lagrangian (function): function of the lagrangian used for momentum dataset
    """
    def __init__(self, input_trajectories, mode="xdot", eom=None, lagrangian=None, np_mode=False):
        super(MultiTrajectoryDataset, self).__init__()
        self.input_trajectories = input_trajectories
        self.mode = mode

        # Optional args
        self.eom = eom
        self.lagrangian = lagrangian
        self.np_mode = np_mode

        for i in range(len(input_trajectories)):
            input_trajectory = input_trajectories[i]
            
            if i == 0:
                # Create targets based on different desired modes
                if mode == "xdot":
                    # For every state x in the trajectory there is a corresponding
                    # xdot (ie, a pair of (q,qdot) -> (qdot, qdotdot) as in the LNN paper)
                    if eom is None:
                        raise ValueError('Missing valid eom parameter')
                    else:
                        x = input_trajectory
                        y = jax.device_get(jax.vmap(eom)(input_trajectory))
                        self.npx = np.array(x)
                        self.npy = np.array(y)
                        self.x = torch.from_numpy(self.npx)
                        self.y = torch.from_numpy(self.npy)

                elif mode == "tripple_Lc" or mode == "tripple_Ld" :
                    x, y = target_zero(input_trajectory)
                    self.npx = np.array(x)
                    self.npy = np.array(y)
                    self.x = torch.from_numpy(np.array(x))
                    self.y = torch.from_numpy(np.array(y))
                else:
                    raise ValueError('Parameter for target argument is not "xdot", "next", "momentum_Lc","momentum_Ld", "tripple_Lc","tripple_Lc", "momentum_Lc_pintraj", "momentum_Ld_pintraj" value given was {}'.format(target))
            else:
                # Create targets based on different desired modes
                if mode == "xdot":
                    # For every state x in the trajectory there is a corresponding
                    # xdot (ie, a pair of (q,qdot) -> (qdot, qdotdot) as in the LNN paper)
                    if eom is None:
                        raise ValueError('Missing valid eom parameter')
                    else:
                        x = input_trajectory
                        y = jax.device_get(jax.vmap(eom)(input_trajectory))
                        self.npx = np.vstack((self.npx, x))
                        self.npy = np.vstack((self.npy, y))
                        self.x = torch.from_numpy(self.npx)
                        self.y = torch.from_numpy(self.npy)

                elif mode == "tripple_Lc" or mode == "tripple_Ld" :
                    x, y = target_zero(input_trajectory)
                    self.npx = np.vstack((self.npx, np.array(x)))
                    self.npy = np.vstack((self.npy, np.array(y)))
                    self.x = torch.from_numpy(self.npx)
                    self.y = torch.from_numpy(self.npy)
                else:
                    raise ValueError('Parameter for target argument is not "xdot", "next", "momentum_Lc","momentum_Ld", "tripple_Lc","tripple_Lc", "momentum_Lc_pintraj", "momentum_Ld_pintraj" value given was {}'.format(target))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.np_mode:
            return self.npx[idx], self.npy[idx]
        else:
            return self.x[idx], self.y[idx]


def numpy_collate(batch):
    """Collate function for a Pytorch to Numpy DataLoader"""
    batchx = [item[0] for item in batch]
    batchy = [item[1] for item in batch]
    np_batchx = [x.cpu().detach().numpy() for x in batchx]
    np_batchy = [y.cpu().detach().numpy() for y in batchy]
    return [np.array(np_batchx), np.array(np_batchy)]


class NumpyLoader(DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

def target_zero(trajectory):
    # q0, q1 , q2 -> 0
    # aka method z-1
    # Returns list
    xs = []
    ys = []
    for i in range(len(trajectory)-2):
        xs.append([trajectory[i], trajectory[i+1], trajectory[i+2]])
        ys.append(jnp.zeros((round(trajectory.shape[1]/2))))

    return xs, ys
