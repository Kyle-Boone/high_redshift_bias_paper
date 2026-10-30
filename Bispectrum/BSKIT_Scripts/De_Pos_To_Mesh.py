# Degrading version of Pos_To_Mesh.py.
import os
import h5py
import argparse
import numpy as np
import scipy.fft as fft
from nbodykit.lab import ArrayMesh
from abacusnbody.analysis.tsc import tsc_parallel

parser = argparse.ArgumentParser()
parser.add_argument('pos_file',
                    help='File of positional galaxy data')
parser.add_argument('write_file',
                    help='File to write the density grid to temporarily')
parser.add_argument('boxsize', type = float,
                    help='BoxSize in Mpc/h')
parser.add_argument('ngrid', type = int,
                    help='N to store the density at')
parser.add_argument('ntsc', type = int,
                    help='N to perform the TSC at')
args = parser.parse_args()

pos_file = args.pos_file
write_file = args.write_file
Lbox = args.boxsize
Nmesh = args.ngrid
Ntsc = args.ntsc

# ND = 256 # N degrade, I'm going to just zero out all modes beyond this.
# kS = ND*np.pi/Lbox
    
with h5py.File(pos_file, 'r') as f:
    dset = f['dataset']
    pos = dset[:].T
    
d = tsc_parallel(pos.astype(np.float32), Ntsc, Lbox, nthread=32)
d = d.astype(np.float32)
ave = np.average(d)
d = (d / np.average(d)) - 1
d = d - np.average(d)

# dk = fft.fftn(d) * (Lbox / Nmesh)**3
# dx   = Lbox / Nmesh
# k    = fft.fftfreq(Nmesh, d=dx).astype(np.float32) * 2*np.pi           # angular frequencies
# kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
# k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
# dk[k_mag>kS] = 0

# d = np.real(fft.ifftn(dk)) * (Nmesh / Lbox)**3
# d = d - np.average(d)
# print(np.average(k_mag>kS))

with fft.set_workers(32):
    dk = fft.fftn(d) * (Lbox / Ntsc)**3
crop = np.full(Ntsc, False, dtype=bool)
crop[:int(Nmesh/2)] = True
crop[-int(Nmesh/2):] = True
crop1, crop2, crop3 = np.meshgrid(crop, crop, crop, indexing='ij')
full_crop = crop1&crop2&crop3
dk_crop = np.reshape(dk[full_crop], (Nmesh, Nmesh, Nmesh))
with fft.set_workers(32):
    d = np.real(fft.ifftn(dk_crop)) * ((Nmesh/Lbox)**3)
d = d - np.average(d)
    
# Create the MeshSource object
mesh = ArrayMesh(d, BoxSize=Lbox)

idx = write_file.rfind('/')+1
data_dir = write_file[:idx]

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
    
mesh.save(write_file)
