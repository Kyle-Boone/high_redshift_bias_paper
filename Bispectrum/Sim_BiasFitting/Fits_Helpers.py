import numpy as np
import pandas as pd
from collections import defaultdict, deque


def __row_keys__(a, round_decimals=None):
    """Return a list of immutable byte-keys, one per row of a."""
    if round_decimals is not None and np.issubdtype(a.dtype, np.floating):
        a = np.round(a, round_decimals)
    a = np.ascontiguousarray(a)
    # View each row as a single blob, then convert each blob to bytes (hashable)
    v = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
    return [v[i].tobytes() for i in range(len(v))]


def __subset_indices__(array1, array2, *, strict=True, round_decimals=None):
    """
    Return indices idx so that array2[idx] equals array1 row-for-row.
    If a row in array1 is missing from array2 -> KeyError (strict=True) or -1 (strict=False).
    Handles duplicates in array2 stably (earliest unused index is chosen).
    """
    if array1.shape[1] != array2.shape[1]:
        raise ValueError("array1 and array2 must have the same number of columns")

    k1 = __row_keys__(array1, round_decimals=round_decimals)
    k2 = __row_keys__(array2, round_decimals=round_decimals)

    # Map row-key -> queue of indices where it appears in array2
    buckets: dict[bytes, deque[int]] = defaultdict(deque)
    for j, key in enumerate(k2):
        buckets[key].append(j)

    idx = np.empty(len(k1), dtype=int)
    for i, key in enumerate(k1):
        q = buckets.get(key)
        if q and q:
            idx[i] = q.popleft()
        else:
            if strict:
                raise KeyError(f"Row {i} of array1 not found in array2: {array1[i]!r}")
            idx[i] = -1

    return idx


def data_power(power_file, kBins, numBins):
    df = pd.read_csv(power_file, sep=r'\s+')
    df.columns = np.roll(df.columns, -1)
    df = df.drop(df.columns[-1], axis=1)
    
    k = df['k_mean_over_bin_[h_Mpc^-1]'].to_numpy()
    N_P = df['N_modes'].to_numpy()
    
    bin_idx = np.digitize(k, kBins) - 1 # -1 underflow, numBins=overFlow
    mask_P = (0 <= bin_idx) & (bin_idx < numBins)
    N_P_tot = np.bincount(bin_idx[mask_P], weights=N_P[mask_P], minlength=numBins)

    Pk = df['P_bin_[h^-3_Mpc^3]'].to_numpy()
    Pbinned = np.bincount(bin_idx[mask_P], weights=(Pk*N_P)[mask_P], minlength=numBins)/N_P_tot

    return Pbinned, k, bin_idx, mask_P, N_P, N_P_tot


def sims_power(power_file, kBins, numBins, Wk, sym_factor=1):
    df = pd.read_csv(power_file, sep=r'\s+')
    df.columns = np.roll(df.columns, -1)
    df = df.drop(df.columns[-1], axis=1)
    
    k = df['k_mean_over_bin_[h_Mpc^-1]'].to_numpy()
    N_P = df['N_modes'].to_numpy()
    
    bin_idx = np.digitize(k, kBins) - 1 # -1 underflow, numBins=overFlow
    mask_P = (0 <= bin_idx) & (bin_idx < numBins)
    N_P_tot = np.bincount(bin_idx[mask_P], weights=N_P[mask_P], minlength=numBins)

    Pk = sym_factor * df['P_bin_[h^-3_Mpc^3]'].to_numpy()
    Pbinned = np.bincount(bin_idx[mask_P], weights=(Wk**2*Pk*N_P)[mask_P], minlength=numBins)/N_P_tot

    return Pk, Pbinned


def data_bispec(bs_file, min_k, max_k, min_ksum, max_ksum):
    # This is for not doing any rounding, leads to ~10 times more points to compute
    # This is necessary in comparing to computed data though.
    df = pd.read_csv(bs_file, sep=r'\s+')
    
    # Shift the header row to the left
    df.columns = np.roll(df.columns, -1)
    
    # Drop the first column (which now contains the original header)
    df = df.drop(df.columns[-1], axis=1)
    
    B=np.array(df['B_bin_[h^-6_Mpc^6]'])
    k1=np.array(df['k1_mean_[h_Mpc^-1]'])
    k2=np.array(df['k2_mean_[h_Mpc^-1]'])
    k3=np.array(df['k3_mean_[h_Mpc^-1]'])
    
    ksum = k1+k2+k3
    cropInds = np.where((ksum >= min_ksum) & (ksum <= max_ksum) & 
                        (k1 >= min_k) & (k1 <= max_k) & 
                        (k2 >= min_k) & (k2 <= max_k) & 
                        (k3 >= min_k) & (k3 <= max_k))[0]

    B = B[cropInds]
    k1 = k1[cropInds]
    k2 = k2[cropInds]
    k3 = k3[cropInds]
    ksum = ksum[cropInds]

    sortInds = np.argsort(ksum)
        
    B = B[sortInds]
    k1 = k1[sortInds]
    k2 = k2[sortInds]
    k3 = k3[sortInds]
    
    ineqCrop = np.where((k1 >= k2) & (k2 >= k3))[0]
    
    B = B[ineqCrop]
    k1 = k1[ineqCrop]
    k2 = k2[ineqCrop]
    k3 = k3[ineqCrop]
    
    BCrop = np.where(np.abs(B) < 1e12)[0] # Removes numerical issues
    
    B = B[BCrop]
    k1 = k1[BCrop]
    k2 = k2[BCrop]
    k3 = k3[BCrop]

    # binning_cut = np.where(k1<0.9*(k2+k3))[0]
    # B = B[binning_cut]
    # k1 = k1[binning_cut]
    # k2 = k2[binning_cut]
    # k3 = k3[binning_cut]

    return B, np.array([k1, k2, k3]).T


def sims_bispec(bs_file, kB, sym_factor = 1):
    df = pd.read_csv(bs_file, sep=r'\s+')
        
    # Shift the header row to the left
    df.columns = np.roll(df.columns, -1)
    
    # Drop the first column (which now contains the original header)
    df = df.drop(df.columns[-1], axis=1)
    
    B_all=np.array(df['B_bin_[h^-6_Mpc^6]'])
    k1_all=np.array(df['k1_mean_[h_Mpc^-1]'])
    k2_all=np.array(df['k2_mean_[h_Mpc^-1]'])
    k3_all=np.array(df['k3_mean_[h_Mpc^-1]'])
    
    k_all = np.array([k1_all, k2_all, k3_all]).T
    k_all = np.sort(k_all, axis=1)[:,::-1]
    k_uniq, inv = np.unique(k_all, axis=0, return_inverse=True)
    indices = [np.where(inv == i)[0] for i in range(len(k_uniq))]
    
    B_uniq = np.zeros(len(indices))
    for i in np.arange(len(indices)):
        B_uniq[i] = np.sum(B_all[indices[i]]) / len(indices[i])
    B_uniq *= sym_factor
    
    sym_inds = __subset_indices__(kB, k_uniq)
    return B_uniq[sym_inds]