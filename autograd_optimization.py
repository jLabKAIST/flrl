import torch as th
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import argparse
import pickle
import warnings 
import os
import json
from datetime import datetime

warnings.filterwarnings("ignore")
import utils.utils_opt as utils_opt
import utils.utils_lsf as utils_lsf

ROOT_DIR = "autograd_optimization_result/"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--folder_save_path", type=str)
    parser.add_argument("--Nx", type=int, default = 8)
    parser.add_argument("--Ny", type=int, default = 4)
    parser.add_argument("--tfo_x", type=int, default = 10)
    parser.add_argument("--tfo_y", type=int, default = 5)
    parser.add_argument("--grid_x", type=int, default = 256)
    parser.add_argument("--grid_y", type=int, default = 128)
    parser.add_argument("--wavelength", type=int, default = 1050)
    parser.add_argument("--angle", type=int, default = 75)
    parser.add_argument("--pol", type=int, choices=range(0, 3), default = 2) #0 TE, 1 TM, 2 TE+TM
    
    parser.add_argument("--num_epochs", type=int, default = 500)
    parser.add_argument("--num_samples", type=int, default = 50)
    parser.add_argument("--norm", type=str, choices=['power', 'tanh'], default = 'power')
    
    #LR
    parser.add_argument("--lr_scheduler", type=str, default = 'linear')
    parser.add_argument("--init_lr", type=float, default = 0.2)
    parser.add_argument("--final_lr", type=float, default = 0.05)
    
    #Binarization beta
    parser.add_argument("--init_beta", type=float, default = 5.0)
    parser.add_argument("--final_beta", type=float, default = 10)
    parser.add_argument("--beta_stop_iter", type=float, default = None)

    #saving purpose
    parser.add_argument("--record_binary", action='store_true')
    parser.add_argument("--record_grad", action='store_true')
    parser.add_argument("--record_evolution", action='store_true')
    
    return parser.parse_args()

def main(args = get_args()):   
    if os.path.isdir(ROOT_DIR) == False:
        os.mkdir(ROOT_DIR)

    timestamp = datetime.today().strftime('%Y%m%d')
    save_path = ROOT_DIR + args.folder_save_path + '_' + timestamp
    
    if os.path.isdir(f"{save_path}") == False:
        print("here")
        os.mkdir(f"{save_path}")
    
    with open(f'{save_path}/{timestamp}_config.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    #Simulation parameter initialization
    period_x = args.wavelength/np.sin(np.deg2rad(args.angle))
    period_y = args.wavelength/2
    fourier_order = [args.tfo_x, args.tfo_y]
    n_air = 1.0
    n_si = utils_opt.get_silicon_index(args.wavelength)
    n_mid = (n_si + n_air)/2
    
    #Solver initialization
    mee = utils_opt.create_solver(fourier_order=fourier_order, pol = 0,
                    wavelength=args.wavelength, period_x=period_x,
                    period_y=period_y)
    
    if args.final_beta == None:
        args.final_beta = args.init_beta

    if args.beta_stop_iter == None:
        args.beta_stop_iter = args.num_epochs
    
    result = {}
    for sample in range(args.num_samples):
        utils_opt.seed_all(seed = args.seed + sample )
        
        #Coeff initialization
        coeff = utils_lsf.construct_fourier_coeff(args.Nx, args.Ny, cols = 'random_gaussian', symmetry = True)
        r, im = utils_lsf.flatten_fourier_coeff(args.Nx, args.Ny, coeff)
        im = np.pad(im, (0, len(r)-len(im)))
        coeff_vectorized = r + 1j*im
        coeff_vectorized = th.tensor(coeff_vectorized)
        
        if args.norm == 'power':
            norm = th.sqrt(th.pow(th.real(coeff_vectorized), 2).sum() + th.pow(th.imag(coeff_vectorized), 2).sum()).detach()
            coeff_vectorized = coeff_vectorized / norm
            coeff_vectorized.requires_grad = True

        opt = th.optim.Adam([coeff_vectorized], lr=args.init_lr)
        
        if args.lr_scheduler == 'linear':
            if args.final_lr == None:
                #constant lr
                args.final_lr = args.init_lr
            end_factor = args.final_lr / args.init_lr
            scheduler = th.optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor = end_factor, total_iters=args.num_epochs//2)
        
        ims = []
        effs = []
        fig, ax = plt.subplots()
        grads = []
        bin_levels = []
        
        for epoch in range(args.num_epochs):
            #beta scheduler
            #if final_beta == init_beta => constant beta
            beta = utils_opt.linear_scheduler(epoch, args.beta_stop_iter, args.init_beta, args.final_beta)
            
            #this is latent coeff
            coeff2 = utils_lsf.inverse_flatten_fourier_coeff_tensor(args.Nx, args.Ny, coeff_vectorized)            
            
            if args.norm == "power" and epoch > 0:
                # coeff_vectorized = th.nn.functional.normalize(th.tensor(coeff_vectorized), 2, dim=0)                
                norm = th.sqrt(th.pow(th.real(coeff2), 2).sum() + th.pow(th.imag(coeff2), 2).sum()).detach()
                coeff2 = coeff2 / norm
            elif args.norm == "tanh":
                #Normalize the fourier coeffs before constructing the lsf
                coeff2 = th.tanh(th.real(coeff2)) + 1j * th.tanh(th.imag(coeff2)).double()
            
            lsf = utils_lsf.generate_lsf_2d_tensor(args.Nx, args.Ny, period_x, period_y, args.grid_x, args.grid_y, coeff2)

            raw_topology = utils_lsf.lsf_to_topology(np.real(lsf), "sigmoid", beta, 0) #raw
            raw_ucell_mapped = utils_opt.binary_to_index(raw_topology.reshape(1, args.grid_y, args.grid_x), args.wavelength)

            if args.record_evolution:
                with th.no_grad():
                    im = ax.imshow(raw_ucell_mapped[0], animated = True, cmap = 'gray')
                    ims.append([im])              
            
            
            #Simulate the partial binarized structure
            mee.ucell = raw_ucell_mapped

            if args.pol == 2:
                mee.pol = 0
                te_eff = utils_opt.get_de(mee, 1, 0)

                mee.pol = 1
                tm_eff = utils_opt.get_de(mee, 1, 0)

                raw_eff = (te_eff + tm_eff) * 0.5
            else:
                mee.pol = args.pol
                raw_eff = utils_opt.get_de(mee, 1, 0)
                
            loss_eff = -raw_eff

            #Calculate the binarization percentage of our partial binarized structure
            binarization_level = -th.sum(th.abs((raw_ucell_mapped.reshape(-1) - n_mid) / (n_si - n_mid))) / len(raw_ucell_mapped.reshape(-1))
            
            print(f"Partial binarized efficiency: {raw_eff.item()}")
            print(f"Weighted efficiency: {-loss_eff.item()}")
            print(f"Binarization level: {binarization_level}")
            
            #Compute the gradient
            loss_eff.backward()
            
            if args.record_grad:
                with th.no_grad():
                    grads.append(coeff_vectorized.grad)
                    print(coeff_vectorized.grad)
            
            #Take a gradient descent with ADAM optimizer
            opt.step()
            scheduler.step()
            
            #Delete the gradient information
            opt.zero_grad()
            
            if args.pol == 2:
                effs.append([raw_eff.item(), te_eff.item(), tm_eff.item()])
            else:
                effs.append([raw_eff.item()])
            bin_levels.append(binarization_level.item())
            
        
        np.save(f"{save_path}/raw_structure_sample_{sample}", raw_ucell_mapped.detach().numpy())
        np.save(f"{save_path}/coeff_sample_{sample}", coeff)
        np.save(f"{save_path}/optimized_coeff_sample_{sample}", coeff2.detach().numpy())
        
        if args.record_evolution:
            np.save(f"{save_path}/sample_{sample}", np.array(ims))
    
            ani = animation.ArtistAnimation(fig, ims, interval=400, blit=True, repeat_delay=1000)
            ani.save(f"{save_path}/sample_{sample}.mp4")
        
        if args.record_binary:
            with th.no_grad():
                #We also keep track of the full binarized structure
                binarized_ucell = th.where(raw_ucell_mapped >= n_mid, n_si, n_air)
                #Simulate the full binarized structure
                mee.ucell = binarized_ucell
                eff_bin = utils_opt.get_de(mee, 1, 0)
                result[f'sample_{sample}'] = [effs, eff_bin]
        else:
            result[f'sample_{sample}'] = [effs]
        
        if args.record_grad:
            with open(f"{save_path}/grad_sample_{sample}.pickle", "wb") as f:
                pickle.dump(grads, f)
        
        np.save(f"{save_path}/bin_level_sample_{sample}", bin_levels)
                
        with open(f"{save_path}/eff_result_sample_{sample}.pickle", "wb") as f:
            pickle.dump(result, f)
        

if __name__ == "__main__":
    main()