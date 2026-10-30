import numpy as np
import multiprocessing as mp


def run_version(version):

    power_dir = '/home/kboone/software/Galaxy_Bias/Bispectrum/Two_Points_Data/'
    write_dir = '/mnt/nvme1/kboone/Sims/Matter_Data/'
    EXTRA_SMOOTHING = 1
    CORES = 32

    import gc
    import os
    import scipy.fft as fft
    import numexpr as ne; ne.set_num_threads(CORES)
    import Visual_Helpers as VH
    from scipy.interpolate import interp1d as inter
    from scipy.interpolate import CubicSpline as spline
    from concurrent.futures import ThreadPoolExecutor

    # Skip-existing guard: Otd is the last field written per version, so its
    # presence means this version finished. Lets a resubmit resume cleanly.
    if os.path.exists(write_dir + 'Otd_v' + str(version) + '.npy'):
        print('Skipping version ' + str(version) + ' (already complete).')
        return

    L = 2000
    N = 2048 # int(2048/EXTRA_SMOOTHING)
    N_comp = 256 # int(256/EXTRA_SMOOTHING)
    kN = N*np.pi/L
    kS = kN/2

    # --- slab-threaded elementwise helpers ---------------------------------
    # The big complex-cube products (k_i k_j d1k / k^2 ...) and the real/imag
    # float32 compactions are plain single-threaded NumPy. We parallelize them
    # by splitting the cube into axis-0 slabs across a thread pool (NumPy ufuncs
    # release the GIL). Pure NumPy on slices => dtypes stay complex64/float32
    # (no numexpr complex128 upcast); results are bit-for-bit identical.
    _POOL = ThreadPoolExecutor(max_workers=CORES)
    sc_full = np.float32((N / L)**3)

    def _S(arr, a, b):
        # slice axis 0 only for full-length cubes; sparse-k broadcasts (shape[0]==1) pass through
        return arr[a:b] if arr.shape[0] == N else arr

    def _slab(out, fn):
        edges = np.linspace(0, out.shape[0], CORES + 1).astype(int)
        def work(ab):
            a, b = ab
            out[a:b] = fn(a, b)
        list(_POOL.map(work, zip(edges[:-1], edges[1:])))
        return out

    def prod_c64(build):
        # build(a,b) -> complex64 slab; returns the full complex64 cube
        return _slab(np.empty((N, N, N), np.complex64), build)

    def transform(build, take):
        # build(a,b) -> complex64 FFT-input slab; ifftn (scipy-threaded, in-place);
        # returns float32 cube = Re/Im( ifftn(input) * (N/L)^3 ), slab-threaded.
        buf = prod_c64(build)
        cube = fft.ifftn(buf, overwrite_x=True)
        comp = (lambda a, b: cube[a:b].imag * sc_full) if take == 'imag' else \
               (lambda a, b: cube[a:b].real * sc_full)
        return _slab(np.empty((N, N, N), np.float32), comp)
    # ------------------------------------------------------------------------

    model_power_file = power_dir + 'CLASS_power_z1' # Location of power spectrum file
    k_model_orig, Pk_model_orig = VH.read_power(model_power_file)
    max_k_model = np.max(k_model_orig)
    Pk_model_orig = Pk_model_orig * (0.2322)**2 # Normalization to z=8

    P_inter = spline(k_model_orig, Pk_model_orig, bc_type='natural')

    log_k = np.log(k_model_orig); log_Pk = np.log(Pk_model_orig)
    log_P = inter(log_k, log_Pk, bounds_error=False, fill_value='extrapolate')

    def P(k, kS=kS, P_inter=P_inter, log_P=log_P, max_k=max_k_model):
        k = np.array(k)
        interpolate = k < max_k
        smooth = k > kS
        retvals = np.empty_like(k, dtype=np.float32)

        retvals[interpolate] = P_inter(k[interpolate])

        retvals[~interpolate] = np.exp(log_P(np.log(k[~interpolate])))

        retvals[smooth] = 0

        return retvals

    def generate_density_field(N, L, Pk_func, seed, workers=CORES, only_phase=True):
        """
        Generates a realization of the matter density field δ(x) from a given power spectrum P(k).

        Parameters:
        - N (int): Number of grid points in each dimension.
        - L (float): Physical size of the simulation box (in units consistent with k).
        - Pk_func (callable): Function that returns P(k) given k.

        Returns:
        - delta_x (ndarray): Real-space matter density field δ(x) on a 3D grid.
        """
        np.random.seed(seed)
        with fft.set_workers(workers):
            dx = L / N
            k  = fft.fftfreq(N, d=dx).astype(np.float32) * 2*np.pi           # angular frequencies
            # sparse meshgrid: kx/ky/kz are (N,1,1)/(1,N,1)/(1,1,N) broadcast views, not full cubes
            kx, ky, kz = np.meshgrid(k, k, k, indexing='ij', sparse=True)
            k_mag = np.sqrt(kx**2 + ky**2 + kz**2)  # broadcasts to full (N,N,N); needed per-element

            # A ──────────────────────────────────────────────────────────────
            vol_prefac = np.float32((L)**1.5)              # √(Δk³)
            keep = k_mag <= kS
            Pk = np.zeros((N, N, N), np.float32)
            Pk[keep] = Pk_func(k_mag[keep])
            if only_phase:
                rnd = np.exp(2*np.pi*1j*np.random.uniform(size=(N,N,N)).astype(np.float32)) * np.float32(np.sqrt(2))
            else:
                rnd = (np.random.normal(size=(N,N,N)) + 1j*np.random.normal(size=(N,N,N))).astype(np.complex64)
            scale = ne.evaluate("sqrt(Pk/2)*vol_prefac")
            delta_k = rnd * scale  # correct variance
            delta_k = enforce_hermitian_symmetry_fast(delta_k)
            # ────────────────────────────────────────────────────────────────

            # inverse FFT (includes its own 1/N³)
            delta_x = np.real(fft.ifftn(delta_k)) * (N/L)**3                    
            # ────────────────────────────────────────────────────────────────
            # return the broadcast components as a list, NOT a stacked (3,N,N,N) cube.
            # k[i] still indexes correctly and broadcasts in every k[i]*d1k expression.
            return delta_x, delta_k, [kx, ky, kz]


    def enforce_hermitian_symmetry_fast(a):
        """
        In-place Hermitian symmetry for a full (N,N,N) complex cube `a` in NumPy FFT ordering.
        For each pair (i,j,k) and (−i,−j,−k) mod N, copy only one side (preserves variance).
        Works for even/odd N. Returns `a`.
        """
        N = a.shape[0]
        assert a.shape == (N, N, N)
        h = N // 2

        # partner index along one axis: p[0]=0; p[q]=N-q for q>0
        p = np.arange(N)
        p[1:] = np.arange(N-1, 0, -1)  # [0, N-1, N-2, ..., 1]

        # ---- 1) Pair i with N-i for i=1..h-1 (even N) or 1..h (odd N)
        if N > 2:
            # indices to copy FROM (the "positive" side)
            isrc = np.arange(1, h + (N % 2))      # 1..h-1 (even) or 1..h (odd)
            idst = (N - isrc) % N                  # their partners

            # vectorized assign over j,k using partner mapping on axes 1 and 2
            # a[idst, :, :] = conj( a[isrc, partner(j), partner(k)] )
            a[idst, :, :] = np.conj(a[isrc][:, :, :][:, p][:, :, p])

        # ---- 2) Inside i=0 (and i=h if even), pair j with N-j for j=1..h-1 (even) or 1..h (odd)
        special_is = [0] + ([h] if N % 2 == 0 else [])
        if N > 2:
            jsrc = np.arange(1, h + (N % 2))
            jdst = (N - jsrc) % N
            for i in special_is:
                # a[i, jdst, :] = conj( a[i, jsrc, partner(k)] )
                a[i, jdst, :] = np.conj(a[i, jsrc, :][:, p])

        # ---- 3) On lines where i and j are special, pair k with N-k for k=1..h-1 (even) or 1..h (odd)
        if N > 2:
            ksrc = np.arange(1, h + (N % 2))
            kdst = (N - ksrc) % N
            for i in special_is:
                for j in special_is:
                    # a[i, j, kdst] = conj( a[i, j, ksrc] )
                    a[i, j, kdst] = np.conj(a[i, j, ksrc])

        # ---- 4) Make only the truly self-conjugate points real
        # (i,j,k) where each coord is 0 (odd N) or in {0, h} (even N)
        for i in special_is:
            for j in special_is:
                for k in special_is:
                    a[i, j, k] = a[i, j, k].real + 0j

        return a


    # * d calculation
    d1x, d1k, k = generate_density_field(N,L,P,version,only_phase=True)
    sig2 = np.std(d1x)**2
    k_mag = np.sqrt(k[0]**2 + k[1]**2 + k[2]**2)  # broadcasts list components to full (N,N,N)
    
    inv_kdiv2 = ne.evaluate("1/k_mag**2")
    inv_kdiv2[0,0,0] = 0

    del k_mag
    gc.collect()

    crop = np.full(N, False, dtype=bool)
    crop[:int(N_comp/2)] = True
    crop[-int(N_comp/2):] = True
    crop1, crop2, crop3 = np.meshgrid(crop, crop, crop, indexing='ij')
    full_crop = crop1&crop2&crop3

    del crop1, crop2, crop3
    gc.collect()

    d1k_crop = np.reshape(d1k[full_crop], (N_comp, N_comp, N_comp))
    with fft.set_workers(CORES):
        d1x_crop = np.real(fft.ifftn(d1k_crop)) * ((N_comp/L)**3)
    print('For d, average/std: ' + str(np.average(d1x_crop)/np.std(d1x_crop)))
    d1x_crop = d1x_crop - np.average(d1x_crop)
    np.save(write_dir+'d1_v'+str(version)+'.npy', d1x_crop)

    # * Precomputed reusable transforms (built from d1k, reused by many fields below)
    # Each ifftn(k_i ... d1k ...) is purely imaginary (odd number of k factors) or purely
    # real (even number), so we store only the meaningful float32 component. These feed
    # d^(2), theta^(2), d^(3), K^2_(11), K^2_(12), and O_td.
    with fft.set_workers(CORES):
        # ki_d1[i]     = Im( ifftn(k_i d1k) * (N/L)^3 )        (purely imaginary)
        # ki_d1_inv[i] = Im( ifftn(k_i d1k / k^2) * (N/L)^3 )  (purely imaginary)
        ki_d1     = [transform(lambda a, b, i=i: _S(k[i],a,b)*d1k[a:b], 'imag') for i in np.arange(len(k))]
        ki_d1_inv = [transform(lambda a, b, i=i: _S(k[i],a,b)*d1k[a:b]*inv_kdiv2[a:b], 'imag') for i in np.arange(len(k))]

        # kij_d1_inv[i][j] = Re( ifftn(k_i k_j d1k / k^2) * (N/L)^3 )  (purely real, symmetric in i,j)
        # Only the 6 unique (i <= j) cubes are computed; [j][i] points at the SAME array,
        # so indexing all 9 entries costs no extra memory (3 are aliases).
        kij_d1_inv = [[None] * len(k) for _ in np.arange(len(k))]
        for i in np.arange(len(k)):
            for j in np.arange(i, len(k)):
                Tij = transform(lambda a, b, i=i, j=j: _S(k[i],a,b)*_S(k[j],a,b)*d1k[a:b]*inv_kdiv2[a:b], 'real')
                kij_d1_inv[i][j] = Tij
                kij_d1_inv[j][i] = Tij

    gc.collect()

    # * d^3 calculation
    d3x = ne.evaluate("d1x**3 - 3*sig2*d1x")
    gc.collect()
    with fft.set_workers(CORES):
        d3k = fft.fftn(d3x) * (L/N)**3

        d3k_crop = np.reshape(d3k[full_crop], (N_comp, N_comp, N_comp))

        del d3x, d3k
        gc.collect()

        d3x_crop = np.real(fft.ifftn(d3k_crop)) * ((N_comp/L)**3)
    print('For d^3, average/std: ' + str(np.average(d3x_crop)/np.std(d3x_crop)))
    d3x_crop = d3x_crop - np.average(d3x_crop)
    np.save(write_dir+'d3_v'+str(version)+'.npy', d3x_crop)

    # * k^2 d calculation
    with fft.set_workers(CORES):
        k2dk = prod_c64(lambda a, b: d1k[a:b] * (_S(k[0],a,b)**2 + _S(k[1],a,b)**2 + _S(k[2],a,b)**2))
        
        k2dk_crop = np.reshape(k2dk[full_crop], (N_comp, N_comp, N_comp))

        # d1k is no longer needed past this point (only ki_d1/ki_d1_inv/kij_d1_inv
        # caches and the *_2k/*2k transforms are used downstream).
        del k2dk, d1k
        gc.collect()
        
        k2dx_crop = np.real(fft.ifftn(k2dk_crop)) * ((N_comp/L)**3)
    print('For k^2d, average/std: ' + str(np.average(k2dx_crop)/np.std(k2dx_crop)))
    k2dx_crop = k2dx_crop - np.average(k2dx_crop)
    np.save(write_dir+'k2d_v'+str(version)+'.npy', k2dx_crop)

    # * d^(2) and t^(2) calculation
    with fft.set_workers(CORES):
        c5 = np.float32(5/7); c2 = np.float32(2/7)
        c3 = np.float32(3/7); c4 = np.float32(4/7)

        d_2x = ne.evaluate("c5 * d1x**2")
        t_2x = ne.evaluate("c3 * d1x**2")

        for i in np.arange(len(k)):
            a = ki_d1[i]; b = ki_d1_inv[i]
            ne.evaluate("d_2x - a*b", out=d_2x)
            ne.evaluate("t_2x - a*b", out=t_2x)
            # d_2x -= np.imag(fft.ifftn(k[i] * d1k) * (N / L)**3) * np.imag(fft.ifftn(k[i] * d1k * inv_kdiv2) * (N / L)**3)

        for i in np.arange(len(k)):
            for j in np.arange(len(k)):
                if j < i:
                    continue
                elif j > i: # Use symmetry, only use upper half.
                    T = kij_d1_inv[i][j]
                    ne.evaluate("d_2x + 2*c2*T*T", out=d_2x)
                    ne.evaluate("t_2x + 2*c4*T*T", out=t_2x)
                else:
                    T = kij_d1_inv[i][j]
                    ne.evaluate("d_2x + c2*T*T", out=d_2x)
                    ne.evaluate("t_2x + c4*T*T", out=t_2x)
                # d_2x += 2/7 * np.real((fft.ifftn(k[i] * k[j] * d1k * inv_kdiv2) * (N / L)**3)**2)

        d_2k = fft.fftn(d_2x) * (L / N)**3
        t_2k = fft.fftn(t_2x) * (L / N)**3

        d_2k_crop = np.reshape(d_2k[full_crop], (N_comp, N_comp, N_comp))

        d_2x_crop = np.real(fft.ifftn(d_2k_crop)) * (N_comp / L)**3
    print('For d^(2), average/std: ' + str(np.average(d_2x_crop)/np.std(d_2x_crop)))
    d_2x_crop = d_2x_crop - np.average(d_2x_crop)
    np.save(write_dir+'d_2_v'+str(version)+'.npy', d_2x_crop)

    # * d^(3) calculation
    with fft.set_workers(CORES):
        d_3x = ne.evaluate("7*d1x*(d_2x + t_2x)")

        del t_2x, d_2x
        gc.collect()
        
        for i in np.arange(len(k)):
            ki_d1_i = ki_d1[i]; ki_d1_inv_i = ki_d1_inv[i]
            # slab-threaded build + compaction; stays complex64/float32, 3 cubes held at once
            ki_t2_i     = transform(lambda a, b, i=i: _S(k[i],a,b)*t_2k[a:b], 'imag')
            ki_d2_i     = transform(lambda a, b, i=i: _S(k[i],a,b)*d_2k[a:b], 'imag')
            ki_t2_inv_i = transform(lambda a, b, i=i: _S(k[i],a,b)*t_2k[a:b]*inv_kdiv2[a:b], 'imag')
            ne.evaluate("d_3x - 9*ki_d1_i*ki_t2_inv_i - 2*ki_t2_i*ki_d1_inv_i - 7*ki_d2_i*ki_d1_inv_i", out=d_3x)
        # free the last iteration's pinned cubes so they don't ride through the loop below
        del ki_t2_i, ki_d2_i, ki_t2_inv_i
        gc.collect()
            # d_3x -= 9 * np.imag(fft.ifftn(k[i] * d1k) * (N / L)**3) * np.imag(fft.ifftn(k[i] * t_2k * inv_kdiv2) * (N / L)**3)
            # d_3x -= 2 * np.imag(fft.ifftn(k[i] * t_2k) * (N / L)**3) * np.imag(fft.ifftn(k[i] * d1k * inv_kdiv2) * (N / L)**3)
            # d_3x -= 7 * np.imag(fft.ifftn(k[i] * d_2k) * (N / L)**3) * np.imag(fft.ifftn(k[i] * d1k * inv_kdiv2) * (N / L)**3)
        
        for i in np.arange(len(k)):
            for j in np.arange(len(k)):
                if j < i:
                    continue
                elif j > i: # Take advantage of symmetry.
                    T = kij_d1_inv[i][j]
                    kij_t2_inv_ij = transform(lambda a, b, i=i, j=j: _S(k[i],a,b)*_S(k[j],a,b)*t_2k[a:b]*inv_kdiv2[a:b], 'real')
                    ne.evaluate("d_3x + 2*4*T*kij_t2_inv_ij", out=d_3x)
                else:
                    T = kij_d1_inv[i][j]
                    kij_t2_inv_ij = transform(lambda a, b, i=i, j=j: _S(k[i],a,b)*_S(k[j],a,b)*t_2k[a:b]*inv_kdiv2[a:b], 'real')
                    ne.evaluate("d_3x + 4*T*kij_t2_inv_ij", out=d_3x)
                # d_3x += 4 * np.real(fft.ifftn(k[i] * k[j] * d1k * inv_kdiv2) * (N / L)**3) * np.real(fft.ifftn(k[i] * k[j] * t_2k * inv_kdiv2) * (N / L)**3)
        # free the stale loop temp before fftn(d_3x) allocates another cube (at the global peak)
        del kij_t2_inv_ij
        gc.collect()

        ne.evaluate("d_3x / 18", out=d_3x)
        # d_3x *= 1/18
        
        d_3k = fft.fftn(d_3x) * (L / N)**3

        d_3k_crop = np.reshape(d_3k[full_crop], (N_comp, N_comp, N_comp))

        del d_3x, d_3k, t_2k
        gc.collect()
        
        d_3x_crop = np.real(fft.ifftn(d_3k_crop)) * (N_comp / L)**3
    print('For d^(3), average/std: ' + str(np.average(d_3x_crop)/np.std(d_3x_crop)))
    d_3x_crop = d_3x_crop - np.average(d_3x_crop)
    np.save(write_dir+'d_3_v'+str(version)+'.npy', d_3x_crop)

    # ki_d1 / ki_d1_inv caches are only used by d^(2) and d^(3); the remaining
    # fields (K^2_(12), K^2_(11), O_td) only need kij_d1_inv. Free 6 cubes now.
    del ki_d1, ki_d1_inv
    gc.collect()

    # * K^2_(12) calculation
    with fft.set_workers(CORES):

        K2_2x = np.zeros((N,N,N), np.float32)

        for i in np.arange(len(k)):
            for j in np.arange(len(k)):
                if j < i:
                    continue
                elif j > i: # Multiply by two for the symmetry.
                    kij_d1_inv_ij = kij_d1_inv[i][j]
                    kij_d2_inv_ij = transform(lambda a, b, i=i, j=j: _S(k[i],a,b)*_S(k[j],a,b)*d_2k[a:b]*inv_kdiv2[a:b], 'real')
                    ne.evaluate("K2_2x + 2 * kij_d1_inv_ij * kij_d2_inv_ij", out=K2_2x)
                    # K2_2x += np.real(fft.ifftn(k[i] * k[j] * d1k * inv_kdiv2) * (N / L)**3)*np.real(fft.ifftn(k[i] * k[j] * d_2k * inv_kdiv2) * (N / L)**3)
                else:
                    kij_d1_inv_ij = kij_d1_inv[i][j] # d1k/3 not subtracted yet
                    kij_d2_inv_ij = transform(lambda a, b, i=i, j=j: _S(k[i],a,b)*_S(k[j],a,b)*d_2k[a:b]*inv_kdiv2[a:b] - d_2k[a:b]/3, 'real')
                    ne.evaluate("K2_2x + (kij_d1_inv_ij - d1x/3) * kij_d2_inv_ij", out=K2_2x)
                    # K2_2x += np.real(fft.ifftn(k[i] * k[j] * d1k * inv_kdiv2 - d1k/3) * (N / L)**3)*np.real(fft.ifftn(k[i] * k[j] * d_2k * inv_kdiv2 - d_2k/3) * (N / L)**3)
        del kij_d2_inv_ij
        gc.collect()
        ne.evaluate("2*K2_2x - 136*d1x*sig2/63", out=K2_2x)
        # K2_2x = 2*K2_2x - 136*d1x*sig2/63
        
        K2_2k = fft.fftn(K2_2x) * (L/N)**3
        
        K2_2k_crop = np.reshape(K2_2k[full_crop], (N_comp, N_comp, N_comp))

        del K2_2x, K2_2k, d_2k
        gc.collect()
        
        K2_2x_crop = np.real(fft.ifftn(K2_2k_crop)) * ((N_comp/L)**3)
    print('For K^2, average/std: ' + str(np.average(K2_2x_crop)/np.std(K2_2x_crop)))
    K2_2x_crop = K2_2x_crop - np.average(K2_2x_crop)
    np.save(write_dir+'K2_2_v'+str(version)+'.npy', K2_2x_crop)

    # * d^2_(11) calculation
    d2x = ne.evaluate("d1x**2 - sig2")
    with fft.set_workers(CORES):
        d2k = fft.fftn(d2x) * (L/N)**3

        d2k_crop = np.reshape(d2k[full_crop], (N_comp, N_comp, N_comp))

        del d2x
        gc.collect()

        d2x_crop = np.real(fft.ifftn(d2k_crop)) * ((N_comp/L)**3)
    print('For d^2, average/std: ' + str(np.average(d2x_crop)/np.std(d2x_crop)))
    d2x_crop = d2x_crop - np.average(d2x_crop)
    np.save(write_dir+'d2_v'+str(version)+'.npy', d2x_crop)

    # * K^2_(11) calculation
    with fft.set_workers(CORES):

        K2x = np.zeros((N,N,N), np.float32)

        for i in np.arange(len(k)):
            for j in np.arange(len(k)):
                if j < i:
                    continue
                elif j > i: # Double due to symmetry.
                    kij_d1_inv_ij = kij_d1_inv[i][j]
                    ne.evaluate("K2x + 2*kij_d1_inv_ij*kij_d1_inv_ij", out=K2x)
                    # K2x += np.real((fft.ifftn(k[i] * k[j] * d1k * inv_kdiv2) * (N / L)**3)**2)
                else:
                    kij_d1_inv_ij = kij_d1_inv[i][j] # Haven't subtracted d1x/3 yet.
                    ne.evaluate("K2x + (kij_d1_inv_ij-d1x/3)*(kij_d1_inv_ij-d1x/3)", out=K2x)
                    # K2x += np.real((fft.ifftn(k[i] * k[j] * d1k * inv_kdiv2 - d1k/3) * (N / L)**3)**2)

        K2x = K2x - 2*sig2/3

        K2k = fft.fftn(K2x) * (L/N)**3

        K2k_crop = np.reshape(K2k[full_crop], (N_comp, N_comp, N_comp))

        del K2x
        gc.collect()

        K2x_crop = np.real(fft.ifftn(K2k_crop)) * ((N_comp/L)**3)
    print('For K^2, average/std: ' + str(np.average(K2x_crop)/np.std(K2x_crop)))
    K2x_crop = K2x_crop - np.average(K2x_crop)
    np.save(write_dir+'K2_v'+str(version)+'.npy', K2x_crop)

    # * O_td calculation.
    with fft.set_workers(CORES):
        Dij_target = d2k - 1.5*K2k

        Otd = np.zeros((N,N,N), dtype=np.float32)
        
        for i in np.arange(len(k)):
            for j in np.arange(len(k)):
                if j < i:
                    continue
                elif j > i:
                    kij_d1_inv_ij = kij_d1_inv[i][j]
                    target_inv_ij = transform(lambda a, b, i=i, j=j: _S(k[i],a,b)*_S(k[j],a,b)*Dij_target[a:b]*inv_kdiv2[a:b], 'real')
                    ne.evaluate("Otd + 2 * kij_d1_inv_ij * target_inv_ij", out=Otd)
                    # Otd += np.real(fft.ifftn(k[i] * k[j] * d1k * inv_kdiv2) * (N / L)**3) * np.real(fft.ifftn(k[i] * k[j] * Dij_target * inv_kdiv2) * (N / L)**3)
                else:
                    kij_d1_inv_ij = kij_d1_inv[i][j] # d1k/3 not subtracted yet
                    target_inv_ij = transform(lambda a, b, i=i, j=j: _S(k[i],a,b)*_S(k[j],a,b)*Dij_target[a:b]*inv_kdiv2[a:b] - Dij_target[a:b]/3, 'real')
                    ne.evaluate("Otd + (kij_d1_inv_ij - d1x/3) * target_inv_ij", out=Otd)
                    # Otd += np.real(fft.ifftn(k[i] * k[j] * d1k * inv_kdiv2 - d1k/3) * (N / L)**3) * np.real(fft.ifftn(k[i] * k[j] * Dij_target * inv_kdiv2 - Dij_target/3) * (N / L)**3)
        del target_inv_ij
        gc.collect()

        ne.evaluate("8*Otd/21 - 32*sig2*d1x/63", out=Otd)

        Otdk = fft.fftn(Otd) * (L/N)**3

        del Otd, d2k, K2k
        gc.collect()

        Otdk_crop = np.reshape(Otdk[full_crop], (N_comp, N_comp, N_comp))

        del Otdk
        gc.collect()

        Otd_crop = np.real(fft.ifftn(Otdk_crop)) * ((N_comp/L)**3)
    print('For Otd, average/std: ' + str(np.average(Otd_crop)/np.std(Otd_crop)))
    Otd_crop = Otd_crop - np.average(Otd_crop)
    np.save(write_dir+'Otd_v'+str(version)+'.npy', Otd_crop)
    
    gc.collect()
    
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    for version in np.arange(25) + 1:
        p = mp.Process(target=run_version, args=(version,))
        p.start()
        p.join()
