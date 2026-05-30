import pandas as pd
import numpy as np
import torch as th
import meent
import random
import time
from pathlib import Path

def seed_all(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def get_silicon_index(wavelength):
    si_data = pd.read_csv(Path(__file__).parent.parent / 'Si_refractive_data.csv')
    n_si = np.interp(wavelength, si_data['WL'], si_data['n']) 

    return n_si

def create_solver(fourier_order, pol, wavelength, period_x, period_y):
    backend = 2  # Torch 0 means Jax, 1 means numpy(Fastest, can't be autograd), 2 means torch
    device = 0 # 0 : CPU, 1 : GPU
    pol = pol # 0: TE, 1: TM

    n_I = 1.45  # n_incidence = SiO2
    n_II = 1  # n_transmission = Air
    
    theta = 0 * th.pi / 180  # angle of incidence
    phi_rcwa = 0 * th.pi / 180  # angle of rotation

    thickness = th.tensor([325.])  # thickness of each layer

    period = th.tensor([period_x, period_y])  # length of the unit cell. Here it's 1D.

    type_complex = th.complex128
    
    if fourier_order[-1] == 0:
        #1D solver
        fourier_order.pop(-1)
    
    mee = meent.call_mee(backend=backend, pol=pol, 
                        n_top=n_I, n_bot=n_II, theta=theta, phi=phi_rcwa,
                        fto=fourier_order, wavelength=wavelength, period=period,
                        thickness=thickness, type_complex=type_complex,
                        device=device)

    return mee

def get_de(mee, x_order = 1, y_order = 0):
    start = time.time()
    
    res = mee.conv_solve().res
    de_ri, de_ti = res.de_ri, res.de_ti
    end = time.time()

    print(f"Time elapsed: {end-start}")
    
    x = de_ti.shape[-1]//2   
    y = de_ti.shape[-2]//2
    eff = de_ti[y + y_order, x + x_order]
    return eff

def binary_to_index(structure, wavelength):
    n_air = 1.0
    n_si = get_silicon_index(wavelength)
    
    ucell_mapped = structure * (n_si - n_air) #from [0,1] to [0,(n_si-n_air)]
    ucell_mapped += n_air #from [0, (n_si - n_air)] to [n_air, n_si]

    return ucell_mapped

def linear_scheduler(epoch, total_iters, initial_value, final_value, ):
    #total_iters can be different from the max number of epochs
    delta_val = final_value - initial_value
    
    if epoch <= total_iters: 
        decay_factor =  delta_val * epoch/total_iters
        updated_value = initial_value + decay_factor
    else:
        updated_value = final_value
    
    return updated_value
    