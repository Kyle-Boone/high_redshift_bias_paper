import os
import h5py
import asdf
import numpy as np
from abacusnbody.analysis.tsc import tsc_parallel
from abacusnbody.data.read_abacus import read_asdf


# L = 300 # Size of the box in Mpc/h
# N = 256 # Resolution, set this to lower valus for more Gaussian behavior
# # cutoff = 1e13
# # redDir = '/mnt/marvin2/bigsims/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z0.800/'
# # fileExt = 'z_0_8'
# # perKept = 1
# cutoff = 1e9 # Threshold value for how massive a halo should be to host a galaxy.
# redDir = '/mnt/marvin2/bigsims/HighZ/HighZ_L300_N6144/halos_b0.25/z14.000/' # Redshift info dir.
# fileExt = 'z_14' # Will be used for file name, consider adding info about L and N
# perKept = 0.1 # Fraction of particles kept for calculating DM density, 0.1 = 10%.

L = 2000
# N = 512
cutoff = 6.34 * 1E11 # For no cutoff, good test, use 8e10
# redDir = '/mnt/marvin2/bigsims/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z8.000/'
# fileExt = 'z_8'
perKept = 1
do_DM = False # If changing this uncomment region afer if do_DM, I commented it out for personal clarity.

for ph in np.arange(25):
    redDir = '/mnt/marvin2/bigsims/AbacusSummit/AbacusSummit_base_c000_ph0' + '{:02d}'.format(ph) + '/halos/z5.000/'
    # redDir = '/mnt/marvin2/eisenste/Abacus_z8/AbacusSummit_base_c000_ph0' + '{:02d}'.format(ph) + '/halos/z8.000/'

    # # Write a message to inform how many particles are kept in each file.
    # with open('Data/Frac_Kept.txt', 'a') as per_file:
    #     message = fileExt + ': ' + str(perKept) + '\n'
    #     per_file.write(message)
    
    try:
        # This gets a list of all files with info on the DM haloes.
        haloDir = redDir + 'halo_info/'
        haloFiles = np.array(os.listdir(haloDir))[np.char.startswith(os.listdir(haloDir), 'h')]
    
        # This is getting the boxsize and particle mass from the first file.
        af = asdf.open(haloDir + haloFiles[0])
        boxSize = af['header']['BoxSize']
        partMass = af['header']['ParticleMassHMsun']
    
        # Get position of haloes and the number of particles in them
        N_parts = np.array([])
        pos = np.array([[],[],[]]).T
        for haloFile in haloFiles:
            af = asdf.open(haloDir + haloFile)
            # Getting the position and number of particles
            pos = np.append(pos, af['data']['x_com'][:], axis = 0)
            N_parts = np.append(N_parts, af['data']['N'][:])
    
        # I'm expecting the position to range from -0.5 to 0.5 in a pretty full manner.
        if np.abs((np.max(np.abs(pos))-0.5)/0.5) > 0.01: # Added the second np.abs later, maybe change
            raise Exception('Unexpected positional arguments')
    
        # Cropping to halos within my L, which might be smaller than the boxSize.
        sizeCrop = np.where(np.linalg.norm(pos * boxSize, ord = np.inf, axis = 1) <= L/2)[0]
        N_parts = N_parts[sizeCrop]
        # Getting comoving position instead of a rescaled comoving position.
        pos = boxSize * pos[sizeCrop]
    
        # Put galaxies everywhere where there is a halo above the mass cutoff.
        gala_pos = pos[np.where(N_parts * partMass >= cutoff)[0]] + L/2
    
        # Write position of the galaxies to data.
        with h5py.File('Data/6E11_Gala_z5_Positions/Gala_ph0'+"{:02d}".format(ph)+'.h5', 'w') as f:
            dset = f.create_dataset(
                'dataset',
                data=gala_pos.T.astype(np.float32),
                chunks=(1, len(gala_pos)),
                compression='gzip'
            )
    
        # if do_DM:
        #     # Particle data will range from -boxsize/2 to boxsize/2, crop to middle region.
        #     scaleCrop = L/2
        
        #     # fieldDir has field particles (not in a halo)
        #     fieldDir = redDir + 'field_rv_A/'
        #     # haloDir has halo particles (in a halo)
        #     haloDir = redDir + 'halo_rv_A/'
            
        #     fieldFiles = np.array(os.listdir(fieldDir))[np.char.startswith(os.listdir(fieldDir), 'f')]
        #     haloFiles = np.array(os.listdir(haloDir))[np.char.startswith(os.listdir(haloDir), 'h')]
        
        #     # This will contain all of the particles that make it through cuts
        #     allCrops = []
            
        #     for fieldFile in fieldFiles:
        #         # Read in positional data
        #         pos = np.array(read_asdf(fieldDir + fieldFile, load = ['pos'])['pos'])
        #         # This continues if none of the particles in the file are within my spatial crop
        #         if np.max([np.min(np.abs(pos[:,0])), np.min(np.abs(pos[:,1])), np.min(np.abs(pos[:,2]))]) > scaleCrop:
        #             continue
        
        #         # Perform a random crop to keep ~perKept fraction of the particles
        #         mask = np.random.rand(len(pos)) <= perKept
        #         pos = pos[mask]
        
        #         # This applies the spatial crop
        #         crop = pos[np.where(np.linalg.norm(pos, ord = np.inf, axis = 1) < scaleCrop)[0]]
        #         allCrops.append(crop)
        
        #     # Identical to the above loop but for particles in halos.
        #     for haloFile in haloFiles:
        #         pos = np.array(read_asdf(haloDir + haloFile, load = ['pos'])['pos'])
        #         if np.max([np.min(np.abs(pos[:,0])), np.min(np.abs(pos[:,1])), np.min(np.abs(pos[:,2]))]) > scaleCrop:
        #             continue
            
        #         mask = np.random.rand(len(pos)) <= perKept
        #         pos = pos[mask]
                
        #         crop = pos[np.where(np.linalg.norm(pos, ord = np.inf, axis = 1) < scaleCrop)[0]]
        #         allCrops.append(crop)
        
        #     # Get a total length to make it easy to preallocate an array for all positional data
        #     totalLen = 0
        #     for i in np.arange(len(allCrops)):
        #         totalLen += len(allCrops[i])
        
        #     # This will be a better formatted version of the positional data.
        #     cropsArr = np.empty((totalLen, 3))
        
        #     # This is filling the cropsArr array.
        #     startInd = 0
        #     for i in np.arange(len(allCrops)):
        #         cropsArr[startInd:startInd + len(allCrops[i])] = allCrops[i]
        #         startInd += len(allCrops[i])
        
        #     cropsArr += L/2
        
        #     # This stores the position of all the dark matter particles.
        #     with h5py.File('Data/DM_'+fileExt+'.h5', 'w') as f:
        #         dset = f.create_dataset(
        #             'dataset',
        #             data=cropsArr.T.astype(np.float32),
        #             chunks=(1, len(cropsArr)),
        #             compression='gzip'
        #         )
        
        #     # This computes the matter overdensity and stores it.
        #     # Need to rewrite, tsc_parallel expects different range of values.
        #     d_matter = tsc_parallel(cropsArr.astype(np.float32), N, L, nthread=32)
        #     ave_matter = np.average(d_matter)
        #     d_matter = (d_matter / np.average(d_matter)) - 1
        #     d_matter = d_matter - np.average(d_matter)
            
        #     np.save('Data/Density_DM_'+fileExt+'.npy', d_matter)
    
    except:
        with open("error.txt", "a") as error_file:
            message = fileExt + ': ' + str(perKept) + '\n'
            error_file.write(message)
    
