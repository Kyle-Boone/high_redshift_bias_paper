'''
These methods are all very specific and are not designed to generalize. I just want to clean up my Jupyter notebooks.
'''
import h5py
import numpy as np
import pandas as pd
from Fits_Helpers import *
from getdist import MCSamples


def get_noises(directory = '../../Data/Gala_z8_Positions/'):
    phases = 25
    noises = np.zeros(phases, dtype=np.float64)
    L = 2000

    for ph in np.arange(phases):
        pos_file = directory + 'Gala_ph0' + '{:02d}'.format(ph) + '.h5'
        
        with h5py.File(pos_file, 'r') as f_:
            dset = f_['dataset']
            pos = dset[:].T
            noises[ph] = L**3/len(pos)
    return noises


def W(k):
    L = 2000
    N_tsc = 2048
    kN = N_tsc*np.pi/L
    return (np.sin(np.pi*k/(2*kN)) / (np.pi*k/(2*kN)))**3


def get_P(directory = '/mnt/nvme1/kboone/Data/BigFiles/Gala_8/'):
    phases = 25
    Pk = []
    for ph in np.arange(phases):
        power_file = directory + 'Gala_ph0' + '{:02d}'.format(ph) + '/Gala_256_ps.dat'
        
        df = pd.read_csv(power_file, sep=r'\s+')
        
        # Shift the header row to the left
        df.columns = np.roll(df.columns, -1)
        
        # Drop the first column (which now contains the original header)
        df = df.drop(df.columns[-1], axis=1)
        k = df['k_mean_over_bin_[h_Mpc^-1]'].to_numpy()
        Pk.append(df['P_bin_[h^-3_Mpc^3]'].to_numpy())
        
    Pk = np.array(Pk)
    return k, Pk


def get_B(z, directory = '/mnt/nvme1/kboone/Data/BigFiles/Gala_8/'):
    rescale = (8+1)/(z+1)
    
    phases = 25
    L = 2000
    N_grid = 256
    kN = N_grid*np.pi/L
    kF = 2*np.pi/L
    
    # Bispectrum data
    max_ksum = kN
    min_ksum = max_ksum/4
    
    min_k = 0.5*kF
    max_k = 64.5*kF
    Bs = []

    for ph in np.arange(phases):
        bs_file = directory + 'Gala_ph0' + '{:02d}'.format(ph) + '/Gala_256_comb.dat'
        B, kB = data_bispec(bs_file, min_k, max_k, min_ksum, max_ksum)
        Bs.append(B)
    
    Bs = np.array(Bs)

    k1, k2, k3 = kB[:,0], kB[:,1], kB[:,2]

    b1b1b2_file = '/mnt/nvme1/kboone/Sims/Bispectra/b1b1b2_comb.dat'
    b1b1b2 = sims_bispec(b1b1b2_file, kB, sym_factor = 3) * rescale**4
    
    b2b2b2_file = '/mnt/nvme1/kboone/Sims/Bispectra/b2b2b2_comb.dat'
    b2b2b2 = sims_bispec(b2b2b2_file, kB, sym_factor = 1) * rescale**6
    
    b1b2b3_file = '/mnt/nvme1/kboone/Sims/Bispectra/b1b2b3_comb.dat'
    b1b2b3 = sims_bispec(b1b2b3_file, kB, sym_factor = 6) * rescale**6
    
    b1b1K2_file = '/mnt/nvme1/kboone/Sims/Bispectra/b1b1K2_comb.dat'
    b1b1K2 = sims_bispec(b1b1K2_file, kB, sym_factor = 3) * rescale**4
    
    b1b1F2_file = '/mnt/nvme1/kboone/Sims/Bispectra/b1b1F2_comb.dat'
    b1b1F2 = sims_bispec(b1b1F2_file, kB, sym_factor = 3) * rescale**4

    return Bs, k1, k2, k3, b1b1b2, b1b2b3, b2b2b2, b1b1K2, b1b1F2


def get_chains(params_file, init_guess_params, names, bias, phases=25):
    '''
    Overarching method for getting the chains for plotting.

    params_file will be the path to the file with all of the parameter best fit values.
    init_guess_params is the initial guess array. BAKE b2 and b3 NORMALIZATION INTO THIS.
    names is a list of the names and labels for the chain.
    bias is a boolean numpy array. Only the 'True' entries will be included in the chain.
    '''
    fits = np.load(params_file)
    
    fit_params = []
    for ph in np.arange(phases):
        fit_params.append(fits[ph]*init_guess_params)
    fit_params = np.array(fit_params)

    C_params = np.cov(fit_params,rowvar=False)

    mean = np.average(fit_params, axis=0)
    
    x = np.random.multivariate_normal(np.array(mean),C_params,30000)

    b = []
    
    for i in np.arange(len(bias)):
        if bias[i]:
            b.append(np.array(x[:,i]).astype(np.float64))

    ssa = np.array(b).T

    chain = MCSamples(samples=ssa,weights=np.ones(30000), names = names[bias], labels = names[bias])

    return fit_params, chain
    


def get_full_fit_chains(params_file, init_guess_params, phases=25):
    '''
    This is for the full 1 loop power / tree level bispectrum fits.
    '''
    fits = np.load(params_file)
    
    fit_params = []
    for ph in np.arange(phases):
        fit_params.append(fits[ph]*init_guess_params)
    fit_params = np.array(fit_params)

    # Convention for b2 and b_nabla^2
    fit_params[:,5] *= 2
    fit_params[:,8] *= -1

    C_params = np.cov(fit_params,rowvar=False)

    mean = np.average(fit_params, axis=0)
    
    x = np.random.multivariate_normal(np.array(mean),C_params,30000)

    b1_ = np.array(x[:,4]).astype(np.float64)  
    b2_ = np.array(x[:,5]).astype(np.float64)
    bT_ = np.array(x[:,6]).astype(np.float64) 
    bOtd_ = np.array(x[:,7]).astype(np.float64)
    bk2_ = np.array(x[:,8]).astype(np.float64)

    ssa = np.c_[b1_.T,b2_.T,bT_.T,bOtd_.T,bk2_.T]
    chain = MCSamples(samples=ssa,weights=np.ones(30000), names = ['b_1','b_2', 'b_{K^2}', 'b_{O_{td}}', r'$b_{\nabla^2}$'], labels = ['b_1','b_2', 'b_{K^2}', 'b_{O_{td}}', r'$b_{\nabla^2}$'])

    return fit_params, chain
    

def get_alt_fit_chains(params_file, init_guess_params, phases=25):
    '''
    This is for the 2 loop power / 1 loop bispectrum fits.
    '''
    fits = np.load(params_file)
    
    fit_params = []
    for ph in np.arange(phases):
        fit_params.append(fits[ph]*init_guess_params)
    fit_params = np.array(fit_params)

    # Convention for b2 and b3
    fit_params[:,5] *= 2
    fit_params[:,6] *= 6

    C_params = np.cov(fit_params,rowvar=False)

    mean = np.average(fit_params, axis=0)
    
    x = np.random.multivariate_normal(np.array(mean),C_params,30000)

    b1_ = np.array(x[:,4]).astype(np.float64)  
    b2_ = np.array(x[:,5]).astype(np.float64)
    b3_ = np.array(x[:,6]).astype(np.float64) 

    ssa = np.c_[b1_.T,b2_.T,b3_.T]
    chain = MCSamples(samples=ssa,weights=np.ones(30000), names = ['b_1','b_2','b_3'], labels = ['b_1','b_2','b_3'])

    return fit_params, chain


def get_sim_noise_fit_chains(params_file, init_guess_params, phases=25):
    '''
    This is for the 2 loop power / 1 loop bispectrum fits.
    '''
    fits = np.load(params_file)
    
    fit_params = []
    for ph in np.arange(phases):
        fit_params.append(fits[ph]*init_guess_params)
    fit_params = np.array(fit_params)

    # Convention for b2 and b3
    fit_params[:,4] *= 2
    fit_params[:,5] *= 6

    C_params = np.cov(fit_params,rowvar=False)

    mean = np.average(fit_params, axis=0)
    
    x = np.random.multivariate_normal(np.array(mean),C_params,30000)

    b1_ = np.array(x[:,3]).astype(np.float64)  
    b2_ = np.array(x[:,4]).astype(np.float64)
    b3_ = np.array(x[:,5]).astype(np.float64) 

    ssa = np.c_[b1_.T,b2_.T,b3_.T]
    chain = MCSamples(samples=ssa,weights=np.ones(30000), names = ['b_1','b_2','b_3'], labels = ['b_1','b_2','b_3'])

    return fit_params, chain


def get_fit_chains(params_file, init_guess_params, phases=25):
    fits = np.load(params_file)
    
    fit_params = []
    for ph in np.arange(phases):
        fit_params.append(fits[ph]*init_guess_params)
    fit_params = np.array(fit_params)

    # Fix the convention discrepancy.
    fit_params[:,4] *= 2
    fit_params[:,5] *= 6

    C_params = np.cov(fit_params,rowvar=False)

    mean = np.average(fit_params, axis=0)
    
    x = np.random.multivariate_normal(np.array(mean),C_params,30000)
    b1_ = np.array(x[:,3]).astype(np.float64)  
    b2_ = np.array(x[:,4]).astype(np.float64)
    b3_ = np.array(x[:,5]).astype(np.float64) 
    bT_ = np.array(x[:,6]).astype(np.float64) 
    if len(init_guess_params) == 8: # This means F2 was varied as well
        F2_ = np.array(x[:,7]).astype(np.float64)
        ssa = np.c_[b1_.T,b2_.T,b3_.T, bT_.T, F2_.T]
        chain = MCSamples(samples=ssa,weights=np.ones(30000), names = ['b_1','b_2', 'b_3', 'b_{K^2}', 'F_2'], labels = ['b_1','b_2', 'b_3', 'b_{K^2}', 'F_2'])
    else:
        ssa = np.c_[b1_.T,b2_.T,b3_.T, bT_.T]
        chain = MCSamples(samples=ssa,weights=np.ones(30000), names = ['b_1','b_2', 'b_3', 'b_{K^2}'], labels = ['b_1','b_2', 'b_3', 'b_{K^2}'])

    return fit_params, chain


def get_chain(fit_params):
    
    C_params = np.cov(fit_params,rowvar=False)

    mean = np.average(fit_params, axis=0)
    
    x = np.random.multivariate_normal(np.array(mean),C_params,30000)
    b1_ = np.array(x[:,3]).astype(np.float64)  
    b2_ = np.array(x[:,4]).astype(np.float64)
    b3_ = np.array(x[:,5]).astype(np.float64) 
    bT_ = np.array(x[:,6]).astype(np.float64) 
    if len(fit_params[0]) == 8: # This means F2 was varied as well
        F2_ = np.array(x[:,7]).astype(np.float64)
        ssa = np.c_[b1_.T,b2_.T,b3_.T, bT_.T, F2_.T]
        chain = MCSamples(samples=ssa,weights=np.ones(30000), names = ['b_1','b_2', 'b_3', 'b_{K^2}', 'F_2'], labels = ['b_1','b_2', 'b_3', 'b_{K^2}', 'F_2'])
    else:
        ssa = np.c_[b1_.T,b2_.T,b3_.T, bT_.T]
        chain = MCSamples(samples=ssa,weights=np.ones(30000), names = ['b_1','b_2', 'b_3', 'b_{K^2}'], labels = ['b_1','b_2', 'b_3', 'b_{K^2}'])

    return chain


def get_params(params_file, init_guess_params, phases=25):
    fits = np.load(params_file)
    
    fit_params = []
    for ph in np.arange(phases):
        fit_params.append(fits[ph]*init_guess_params)
    fit_params = np.array(fit_params)

    return fit_params