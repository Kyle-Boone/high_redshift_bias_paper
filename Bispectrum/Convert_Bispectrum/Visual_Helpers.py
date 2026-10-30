import itertools
import numpy as np
import pandas as pd


def __expand_symmetry_unique__(k1, k2, k3, B):
    triplet_dict = {}
    N = len(B)

    for i in range(N):
        triple = (k1[i], k2[i], k3[i])
        permutations = set(itertools.permutations(triple))
        for perm in permutations:
            key = perm
            if key not in triplet_dict:
                triplet_dict[key] = B[i]
            else:
                # If duplicate triplets have different B values, decide how to handle them.
                # For example, you can average them:
                # triplet_dict[key] = (triplet_dict[key] + B[i]) / 2
                # For now, we'll assume B[i] is consistent and ignore duplicates.
                pass

    # Extract the unique triplets and their corresponding B values
    new_k1 = []
    new_k2 = []
    new_k3 = []
    new_B = []
    for key, value in triplet_dict.items():
        new_k1.append(key[0])
        new_k2.append(key[1])
        new_k3.append(key[2])
        new_B.append(value)

    return np.array(new_k1), np.array(new_k2), np.array(new_k3), np.array(new_B)


def __round_to_sequence__(k):
    """
    Rounds each element in array k to the nearest value of the form 0.0125 + 0.025n.

    Parameters:
    k (array-like): Input array of values to be rounded.

    Returns:
    numpy.ndarray: Array with elements rounded to the specified sequence.
    """
    # Ensure k is a NumPy array for vectorized operations
    k = np.asarray(k)
    
    # Calculate n for each element
    n = np.round((k - 0.0125) / 0.025)
    
    # Compute the rounded values
    k_rounded = 0.0125 + 0.025 * n
    
    return k_rounded


def read_power(filename):
    df_p = pd.read_csv(filename, sep=r'\s+')
    
    # Shift the header row to the left
    df_p.columns = np.roll(df_p.columns, -1)
    
    # Drop the first column (which now contains the original header)
    df_p = df_p.drop(df_p.columns[-1], axis=1)
    
    P = df_p['P_bin_[h^-3_Mpc^3]'].to_numpy()
    k = df_p['k_mean_over_bin_[h_Mpc^-1]'].to_numpy()

    return k, P


def read_bi(filename, min_k = 0.025, max_k = 2):
    df = pd.read_csv(filename, sep=r'\s+')

    # Shift the header row to the left
    df.columns = np.roll(df.columns, -1)
    
    # Drop the first column (which now contains the original header)
    df = df.drop(df.columns[-1], axis=1)

    B=np.array(df['B_bin_[h^-6_Mpc^6]'])
    k1=np.array(df['k1_mean_[h_Mpc^-1]'])
    k2=np.array(df['k2_mean_[h_Mpc^-1]'])
    k3=np.array(df['k3_mean_[h_Mpc^-1]'])
    
    ksum = k1+k2+k3
    cropInds = np.where((ksum >= 3*min_k) & (ksum <= 3*max_k) & 
                        (k1 >= min_k) & (k1 <= max_k) & 
                        (k2 >= min_k) & (k2 <= max_k) & 
                        (k3 >= min_k) & (k3 <= max_k))[0]
    
    B = B[cropInds]
    k1 = k1[cropInds]
    k2 = k2[cropInds]
    k3 = k3[cropInds]
    ksum = ksum[cropInds]
    
    k1 = __round_to_sequence__(k1)
    k2 = __round_to_sequence__(k2)
    k3 = __round_to_sequence__(k3)
    ksum = __round_to_sequence__(ksum)
    
    sortInds = np.argsort(ksum)
    
    B = B[sortInds]
    k1 = k1[sortInds]
    k2 = k2[sortInds]
    k3 = k3[sortInds]
    ksum = ksum[sortInds]
    
    ineqCrop = np.where((k1 >= k2) & (k2 >= k3))[0]
    
    B = B[ineqCrop]
    k1 = k1[ineqCrop]
    k2 = k2[ineqCrop]
    k3 = k3[ineqCrop]
    ksum = ksum[ineqCrop]
    
    BCrop = np.where(np.abs(B) < 1e10)[0]
    
    B = B[BCrop]
    k1 = k1[BCrop]
    k2 = k2[BCrop]
    k3 = k3[BCrop]
    ksum = ksum[BCrop]

    return B, k1, k2, k3, ksum
