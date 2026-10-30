import sys
sys.path.append('/home/kboone/software/Galaxy_Bias/Bispectrum/Convert_Bispectrum/')

import os
import numpy as np
import pandas as pd
import Visual_Helpers as VH
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d as inter
from scipy.interpolate import CubicSpline as spline

import matplotlib
matplotlib.style.use('boone')

cmap = plt.get_cmap("tab10")


power_dir = '/home/kboone/software/Galaxy_Bias/Bispectrum/Two_Points_Data/'
model_power_file = power_dir + 'CLASS_power_z1' # Location of power spectrum file
k_model_orig, Pk_model_orig = VH.read_power(model_power_file)
max_k_model = np.max(k_model_orig)
Pk_model_orig = Pk_model_orig * (0.2322)**2 # Normalization to z=8

P = spline(k_model_orig, Pk_model_orig, bc_type='natural')

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
        N = df['N_modes'].to_numpy()
        Pk.append(df['P_bin_[h^-3_Mpc^3]'].to_numpy())
        
    Pk = np.array(Pk)
    return k, N, Pk


def get_sim_Pk(directory, abbreviation, symfactor=1):
    all_files = os.listdir(directory)
    files = np.array([file for file in all_files if 
                      file.endswith('.dat') and file.startswith(abbreviation) and '_v' in file])
    Pk = []
    for file in files:
        power_file = directory + file
        
        df = pd.read_csv(power_file, sep=r'\s+')
        
        # Shift the header row to the left
        df.columns = np.roll(df.columns, -1)
        
        # Drop the first column (which now contains the original header)
        df = df.drop(df.columns[-1], axis=1)
        k = df['k_mean_over_bin_[h_Mpc^-1]'].to_numpy()
        Pk.append(df['P_bin_[h^-3_Mpc^3]'].to_numpy())
        
    Pk = np.array(Pk)
    return Pk*symfactor


phases = 25
data_dir = '/mnt/nvme1/kboone/Data/BigFiles/Gala_8/'
z = 8
rescale = (8+1)/(z+1)

L = 2000
V = L**3
N_grid = 256
kF = 2*np.pi/L
kN = N_grid*np.pi/L


k, Nk, Pk = get_P(directory = data_dir)
std_P = np.std(Pk, axis=0, ddof=1)
std_cov = np.outer(std_P, std_P)
cov_P = np.cov(Pk.T, ddof=1)
kMax_P = kN


b1 = 9.8
b2 = 42.5
b3 = 51.7


I = np.average(get_sim_Pk('/mnt/nvme1/kboone/Sims/PowerSpectra/', 'b2b2'), axis=0)


def A_integrand(mu, k, kp):
    return P(np.sqrt(k**2 + kp**2 + 2*k*kp*mu))


def A(k, kp):
    '''
    Angular average of the k and k' (I call kp) sum.
    '''
    return 0.5 * quad(A_integrand, -1, 1, args=(k,kp))[0]


diag_comp = 2*b1**4*P(k)**2/Nk + 4*b1**2*b2**2*P(k)*I/Nk
analytic_cov = np.diag(diag_comp)
for i in np.arange(len(k)):
    for j in np.arange(len(k)):
        K = k[i]; KP = k[j]
        analytic_cov[i,j] += 8*b1**2*b2**2*(P(K)**2+P(KP)**2)/V * A(K,KP)
        analytic_cov[i,j] += 16*b1**2*b2**2*P(K)*P(KP)/V * A(K,KP)
        analytic_cov[i,j] += 12*b1**3*b3*P(K)*P(KP)*(P(K)+P(KP))/V