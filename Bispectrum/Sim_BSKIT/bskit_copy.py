"""Bispectrum measurement module, building on nbodykit

Module for making power spectrum and bispectrum measurements with nbodykit
using FFTs, as described e.g. in Donghui Jeong's PhD thesis. The end goal
is to integrate this functionality into nbodykit itself, but for now,
I've collected the associated routines in this module.

Main documentation and example scripts can be found in the module's
repository on github:
    https://github.com/sjforeman/bskit
Also see that repository for a citation you are requested to include in
any published work that uses this code.

Simon Foreman, Perimeter Institute, 2019
"""

import numpy as np
import logging
from scipy.interpolate import interp1d
import itertools

import nbodykit.lab as nbk
import nbodykit.algorithms.fftpower as nbkfftpower
from nbodykit import setup_logging, style, CurrentMPIComm
from nbodykit.meshtools import SlabIterator
from pmesh.pm import ParticleMesh, RealField, ComplexField
from pmesh.window import Affine



def generate_bin_edge_list(kmin = -1.,
                           kmax = -1.,
                           dk = -1.,
                           num_lowk_bins = 0,
                           dk_high = -1.):
    """Generate list of k bin edges, based on input kmin, kmax,
    dk.
    
    If desired, low-k and high-k bins can have different
    widths - this is specified through num_lowk_bins and dk_high.
    For example, (kmin=0.05, kmax=1., dk=0.05, num_lowk_bins=10, dk_high=0.1)
    will result in the 10 lowest-k bins having width 0.05, and the remaining
    higher-k bins having width 0.1.
    
    Parameters
    ----------
    kmin : float
        lower edge of lowest bin
    kmax : float
        upper edge of highest bin
    dk : float
        bin width
    num_lowk_bins : float, optional
        number of low-k bins with width `dk`
    dk_high : float, optional
        width for higher-k bins
    
    Returns
    -------
    array_like
        shape(Nbins,2) array of lower and upper bin edges
    """
    
    # Check that kmin, kmax, and dk are greater than zero
    if kmin <= 0.: raise ValueError('kmin must be > 0!')
    if kmax <= 0.: raise ValueError('kmax must be > 0!')
    if dk <= 0.: raise ValueError('dk must be > 0!')
    
    # Check for different dk at high k
    if num_lowk_bins > 0 and dk_high < 0.:
        raise ValueError('Must specify dk_high if num_lowk_bins > 0!')
    
    # Make evenly-spaced arrays of k_lower, k_upper, and dk values,
    # with k_lower from kmin to kmax-dk
    kL_arr = np.arange(kmin,kmax-dk,dk)
    dk_arr = np.ones_like(kL_arr)*dk
    kU_arr = kL_arr+dk_arr
    
    # If we want to use a different dk at higher k, make separate
    # lists of the higher-k bin edges, and then appending them to the
    # low-k bin edge lists
    if num_lowk_bins > 0 and num_lowk_bins < len(kL_arr):
        kLhigh_arr = np.arange(kU_arr[num_lowk_bins-1],kmax-dk_high,dk_high)
        dkhigh_arr = np.ones_like(kLhigh_arr)*dk_high
        kUhigh_arr = kLhigh_arr+dkhigh_arr
        
        kL_arr = np.hstack((kL_arr[:num_lowk_bins],kLhigh_arr))
        dk_arr = np.hstack((dk_arr[:num_lowk_bins],dkhigh_arr))
        kU_arr = np.hstack((kU_arr[:num_lowk_bins],kUhigh_arr))
    
    # Join the upper and lower bin edge lists into one list of (kL,kU)
    kLU_arr = np.vstack((kL_arr[:],kU_arr[:])).T
    
    return kLU_arr


def generate_triangle_bin_list(kmin = -1.,
                               kmax = -1.,
                               dk = -1.,
                               mu_min = None,
                               mu_max = None,
                               dmu = None,
                               num_fields = 1,
                               num_lowk_bins = 0,
                               dk_high = -1.,
                               return_indices = False):
    """Generate list of (k1,k2,k3) bin boundaries for general bispectrum triangles.
    
    The output array has shape(N_tri,6) where N_tri is the number of
    triangle bins, and the second index is over
        [k1_min,k1_max,k2_min,k2_max,k3_min,k3_max] ,
    subject to restrictions
        k1_min >= k2_min >= k3_min
    and
        k2_max + k3_max >= k1_min .
    If num_fields > 1, these conditions are altered to ensure that all non-redundant
    triangle bins are included in the list.
        
    In the future, binning can be specified as bins in mu (cosine of the angle between
    the two equal-length sides), but this feature has not been implemented yet.
    
    Parameters
    ----------
    kmin : float
        lower edge of lowest bin
    kmax : float
        upper edge of highest bin
    dk : float
        bin width
    mu_min : float, optional
        lower edge of lowest mu bin (NOT IMPLEMENTED)
    mu_max : float, optional
        upper edge of highest mu bin (NOT IMPLEMENTED) 
    num_fields : int, optional
        number of fields being correlated in bispectrum
    num_lowk_bins : float, optional
        number of low-k bins with width `dk`
    dk_high : float, optional
        width for higher-k bins
    return_indices : bool, optional
        instead of bin edges, return shape(N_tri,3) array of k-bin indices
    
    Returns
    -------
    array_like
        either shape(N_tri,6) array of bin edges or shape(N_tri,3) array of k-bin indices
    """
    
    # Check that num_fields is valid
    if num_fields not in [1,2,3]: raise ValueError('num_fields must be 1, 2, or 3!')
    
    # Check for mu information
    if mu_min is not None or mu_max is not None or dmu is not None:
        raise NotImplementedError('Mu binning not implemented yet!')
        
    # Get list of k bin edges
    kLU_arr = generate_bin_edge_list(kmin,kmax,dk,num_lowk_bins,dk_high)
    
    def filter_triangles(arr):
        # Remove bin combinations for which kA_max+kB_max<kC_min,
        # for (A,B,C) = any permutation of (1,2,3).
        # The input array is of triplets of (k_lower,k_upper)
        return np.array(list(filter(lambda x: x[0][1]+x[1][1]>=x[2][0]
                                                         and x[0][1]+x[2][1]>=x[1][0]
                                                         and x[1][1]+x[2][1]>=x[0][0], arr)))
    
    if num_fields == 1:
        # Make array of all combinations of 3 of k values, with replacement
        tri_arr = np.array(list(itertools.combinations_with_replacement(kLU_arr,3)))
        # Remove invalid triangles
        tri_arr_filt = filter_triangles(tri_arr)
        # Currently, the columns are such that 0<=1<=2. The output will be more intuitive if we
        # swap them around
        temp = tri_arr_filt[:,2].copy()
        tri_arr_filt[:,2] = tri_arr_filt[:,0].copy()
        tri_arr_filt[:,0] = temp
        # Reshape array into list of 6-element items
        tri_arr_filt = tri_arr_filt.reshape(tri_arr_filt.shape[0],6)
        
    elif num_fields == 2:
        # Make array of all combinations of 2 k values, with replacement
        tri_arr_2 = np.array(list(itertools.combinations_with_replacement(kLU_arr,2)))
        tri_arr = [[a[0],a[1],b] for a in tri_arr_2 for b in kLU_arr]
        # Remove invalid triangles
        tri_arr_filt = filter_triangles(tri_arr)
        # Currently, the columns are such that 0<=1. The output will be more intuitive if we
        # swap them around
        temp = tri_arr_filt[:,1].copy()
        tri_arr_filt[:,1] = tri_arr_filt[:,0].copy()
        tri_arr_filt[:,0] = temp
        # Reshape array into list of 6-element items
        tri_arr_filt = tri_arr_filt.reshape(tri_arr_filt.shape[0],6)
        
    elif num_fields == 3:
        # Make array of all permutations of 3 k values with repetition
        tri_arr = list(itertools.product(kLU_arr,repeat=3))
        # Remove invalid triangles
        tri_arr_filt = filter_triangles(tri_arr)
        # Reshape array into list of 6-element items
        tri_arr_filt = tri_arr_filt.reshape(tri_arr_filt.shape[0],6)
        
    # Finally, we sort the list by the first k_lower, then second, then third, so that
    # the first column increases most quickly. This ensures that all triangles with a lower
    # maximum side length appear in the list before the maximum side length is incremented.
    tri_bin_edges = np.sort(tri_arr_filt.view('f8,f8,f8,f8,f8,f8'), 
                           order=['f0','f2','f4'], axis=0).view(np.float)
    
    if not return_indices:
        # Return array of triangle bin edges
        return tri_bin_edges
    else:
        # Return array of k bin indices corresponding to k bins for each side
        return np.array([ [ np.where(kLU_arr[:,0]==tri_bin_edges[t][i])[0][0] 
           for i in [0,2,4] ] for t in range(len(tri_bin_edges)) ])
        

class FFTBispectrum(nbkfftpower.FFTBase):
    """A class to organize bispectrum measurements
    
    This class contains methods and containers for bispectrum measurements from
    a single grid, or in the case of cross bispectra, and given set of 2 or 3
    grids. Its structure is based on nbodykit's FFTPower.
    
    Upon initialization, the class converts and paints the input mesh(es)
    in Fourier space, and pre-computes the list of triangle grids over which
    the measurements will occur. Measurements are made with measure_bispectrum
    (slower algorithm) or measure_bispectrum_faster (measure unnormalized B values
    only, using faster algorithm) and measure_gridinfo_faster (measure triangle
    counts and k_mean values within triangle bins). Final results can be saved
    to file using save_bispectrum, or can also be saved to file at intervals
    during execution (recommended).
    
    The results from the three measurement routines are stored in a dict b,
    with keys 'index' (triangle bin indices), 'k_edge' (k1,k2,k3 bin 
    boundaries for each triangle bin), 'k_mean' (k_mean
    values in bins), 'B' (bispectrum values in bins), and 'N_tri' (triangle counts
    in bins).
    
    Parameters
    ----------
    source : CatalogSource, MeshSource
        the source for the first field; if a CatalogSource is provided, it
        is automatically converted to MeshSource using the default painting
        parameters (via :func:`~nbodykit.base.catalogmesh.CatalogMesh.to_mesh`)
    Nmesh : float, optional
        the number of cells per side in the particle mesh used to paint the source
    BoxSize : float, optional
        the size of the box
    dk : float, optional
        the linear spacing of k bins to use; if not provided, the
        fundamental mode of the box is used
    kmin : float, optional
        the lower edge of the first k bin to use. only optional if k_edges is specified
    kmax : float, optional
        the upper edge of the last k bin to use. only optional if k_edges is specified
    num_lowk_bins : float, optional
        number of low-k bins with width `dk`
    dk_high : float, optional
        width for higher-k bins
    dmu : float, optional
        NOT IMPLEMENTED
    mu_min : float, optional
        NOT IMPLEMENTED
    mu_max : float, optional
        NOT IMPLEMENTED
    pos_units_mpcoverh : float, optional
        conversion factor between position units of input grid and Mpc/h
    k_edges : float, optional
        array with shape (N_tri,6) of triangle bin edges, where second index is over
            [ k1_low k1_high k2_low k2_high k3_low k3_high ]
    second : CatalogSource, MeshSource
        second mesh to cross-correlate with
    third : CatalogSource, MeshSource
        third mesh to cross-correlate with
    triangle_type : {'all', 'equilateral', 'isosceles', 'squeezed'}
        type of triangle to measure bispectrum on
    isos_mult : float, optional
        multiplier defining triangles
    isos_tol : float, optional
        fractioanl tolerance for the isosceles condition to be satisfied
    squeezed_bin_index : int, optional
        k bin index of k_long bin
    for_grid_info_only : bool, optional
        if true, don't paint input mesh(es) - saves time if only grid info is wanted
    
    Methods
    -------
    set_k_edges(k_edges)
        set k bin edge array to an input array
    measure_bispectrum(imin=None, imax=None, kmeas_min=None, kmeas_max=None,
                       out_file=None, verbose=0)
        measure bispectrum (along with triangle counts and mean k values) 
        on all triangles bins with indices between imin and imax,
        intermittently writing results to out_file
    measure_bispectrum_faster(imin=None, imax=None, kmeas_min=None, kmeas_max=None,
                       out_file=None, verbose=0)
        measure unnormalized bispectrum  
        on all triangle bins with indices between imin and imax,
        intermittently writing results to out_file
    measure_gridinfo_faster(imin=None, imax=None, kmeas_min=None, kmeas_max=None,
                       out_file=None, verbose=0)
        measure  triangle counts and mean k values) 
        on all triangle bins with indices between imin and imax,
        intermittently writing results to out_file
    save_bispectrum(out_file)
        write bispectrum results to out_file
    """
    logger = logging.getLogger('FFTBispectrum')

    def __init__(self,
                 source,
                 Nmesh = None,
                 BoxSize = None, 
                 dk = None,
                 kmin = None,
                 kmax = None,
                 num_lowk_bins = 0,
                 dk_high = -1.,
                 dmu = None,
                 mu_min = None,
                 mu_max = None,
                 pos_units_mpcoverh = 1.,
                 k_edges = None,
                 second = None,
                 third = None,
                 triangle_type='all',
                 isos_mult = 0,
                 isos_tol = 0.1,
                 squeezed_bin_index = 0,
                 for_grid_info_only = False
                 ):
        
        # Via FFTBase initializer, cast source to MeshSource and set some attributes
        nbkfftpower.FFTBase.__init__(self, source, None, Nmesh, BoxSize)
        self.mesh = self.first
        self.second = None
        self.third = None
        
        # If second or third source arguments exist, cast them to meshes.
        # Also determine number of fields here
        self.num_fields = 1
        if second is not None:
            self.second = nbkfftpower._cast_source(second,Nmesh=Nmesh,BoxSize=BoxSize)
            self.num_fields += 1
        if third is not None:
            if second is None:
                raise ValueError('Need second source defined if third source is defined!')
            self.third = nbkfftpower._cast_source(third,Nmesh=Nmesh,BoxSize=BoxSize)
            self.num_fields += 1

        # Set dk to k_f if not specified
        if dk is None:
            dk = 2 * np.pi / self.attrs['BoxSize'].min()
           
        # Check for input kmin/kmax or k_edges
        if (kmin is None or kmax is None) and k_edges is None:
            raise ValueError('Must specify either {kmin,kmax} values or k_edges array!')

        # Save metadata
        self.attrs['dk'] = dk
        self.attrs['kmin'] = kmin
        self.attrs['kmax'] = kmax
        self.attrs['dmu'] = dmu
        self.attrs['mu_min'] = mu_min
        self.attrs['mu_max'] = mu_max
        self.attrs['pos_units_mpcoverh'] = pos_units_mpcoverh
        self.attrs['triangle_type'] = triangle_type
        self.attrs['num_lowk_bins'] = num_lowk_bins
        self.attrs['dk_high'] = dk_high
        self.attrs['isos_mult'] = isos_mult
        self.attrs['isos_tol'] = isos_tol
        self.attrs['squeezed_bin_index'] = squeezed_bin_index
        
        # Generate list of triangle bin edges, or take directly from input array
        if k_edges is not None:
            self.k_edges = k_edges
        else:
            if triangle_type == 'all':
                self.k_edges = generate_triangle_bin_list(kmin=kmin,kmax=kmax,dk=dk,
                                                          dmu=dmu,mu_min=mu_min,mu_max=mu_max,
                                                          num_fields=self.num_fields,
                                                   num_lowk_bins=num_lowk_bins,dk_high=dk_high)
                self.k_indices = generate_triangle_bin_list(kmin=kmin,kmax=kmax,dk=dk,
                                                          dmu=dmu,mu_min=mu_min,mu_max=mu_max,
                                                          num_fields=self.num_fields,
                                                   num_lowk_bins=num_lowk_bins,dk_high=dk_high,
                                                   return_indices=True)
                
        # Initialize variable for dicts of measured quantities
        self.b = None
        
        # Paint mesh(es) explicitly, to prevent explicit re-painting with each
        # bispectrum measurement
        if not for_grid_info_only:
            self._paint_meshes()
        else:
            self.attrs['painted'] = False
            
        


    def _paint_meshes(self):
        """Paint mesh(es), to prevent explicit re-painting with each bispectrum measurement
        """
        
        cfield = self.mesh.paint(mode='complex')
        self.mesh = nbk.FieldMesh(cfield)
        if self.num_fields > 1:
            cfield = self.second.paint(mode='complex')
            self.second = nbk.FieldMesh(cfield)
        if self.num_fields > 2:
            cfield = self.third.paint(mode='complex')
            self.third = nbk.FieldMesh(cfield)
            
        self.attrs['painted'] = True
        
        

    
    
    
    def measure_bispectrum_faster(self,
                           imin = None,
                           imax = None,
                           kmeas_min = None,
                           kmeas_max = None,
                           out_file = None,
                           verbose = 0):
        """Measure unnormalized bispectrum over range of triangle bins.
        
        The range of triangle bins to use is defined by their minimum and maximum indices
        in the stored `k_edges` array. In the future, the user will be able to specify min
        and max k values, but that feature has not been implemented yet.
        
        This routine only measures the unnormalized bispectrum values in each bin, and
        only stores the bin indices, bin bounds and unnormalized B values to the `b` dict
        afterwards. To obtain k_means and triangle counts (necessary for properly
        normalizing the B values from this routine), one should have run the
        measure_gridinfo_faster routine first and stored its output.
        
        Parameters
        ----------
        imin, imax : int
            minimum and maximum triangle bin indices to measure bispectrum between
        kmeas_min, kmeas_max : float, optional
            NOT IMPLEMENTED
        out_file : str, optional
            filename of output file to save intermittent results to
        verbose : int, optional
            verbosity level of output: 0 (default) for nothing, >0 for more than nothing
        meas_type : {'full', 'grid_info', 'unnorm_b_value'}
            measurement type, as described above
        
        Returns
        -------
        dict
            b dict (see doctring for FFTBispectrum class) containing information for
            bins looped over in this function call
        """

        num_tri_to_write = 100
        
        comm = CurrentMPIComm.get()
        
        # Generate lists of k bin edges and indices
        k_bin_edges = generate_bin_edge_list(self.attrs['kmin'],
                           self.attrs['kmax'],
                           self.attrs['dk'],
                           self.attrs['num_lowk_bins'],
                           self.attrs['dk_high'])
        k_bin_indices = np.arange(len(k_bin_edges))
        
        # Also fetch arrays of triangle bin edges and indices
        triangle_bin_edge_list = self.k_edges[imin:min(imax,len(self.k_edges))]
        triple_index_list = self.k_indices[imin:min(imax,len(self.k_indices))]
        triangle_index_list = range(imin,min(imax,len(self.k_indices)))
        
        # Make empty dict to hold delta fields
        delta_fields = {}
        delta_fields2 = {}
        delta_fields3 = {}
        if verbose > 0 and comm.rank == 0: print('Starting delta_field computations')
            
        # Generate real-space delta fields for each k bin, and add to dict, indexed
        # by k bin index
        for i in k_bin_indices:
            
            def mask(k,v):
                # Compute |\vec{k}| values on the mesh
                kk = sum(ki ** 2. for ki in k)**0.5
                # Mask out values outside desired bin
                return v * ((kk<=k_bin_edges[i][1]) & (kk>=k_bin_edges[i][0]))
        
            if verbose > 0 and comm.rank == 0: print('\tConstructing masked field for bin %d' % i)
                
            # if self.second is not None:
            #     raise NotImplementedError('Only auto bispectrum implemented so far!')
        
            mesh_copy = self.mesh.view()
            mesh_copy._actions.append(('complex', mask, 'wavenumber'))
            delta_fields[i] = mesh_copy.paint(mode='real')

        if self.second is not None:
            for ii in k_bin_indices:
            
                def mask(k,v):
                    # Compute |\vec{k}| values on the mesh
                    kk = sum(ki ** 2. for ki in k)**0.5
                    # Mask out values outside desired bin
                    return v * ((kk<=k_bin_edges[ii][1]) & (kk>=k_bin_edges[ii][0]))
            
                if verbose > 0 and comm.rank == 0: print('\tConstructing masked field for bin %d' % i)
            
                second_copy = self.second.view()
                second_copy._actions.append(('complex', mask, 'wavenumber'))
                delta_fields2[ii] = second_copy.paint(mode='real')

            for iii in k_bin_indices:
            
                def mask(k,v):
                    # Compute |\vec{k}| values on the mesh
                    kk = sum(ki ** 2. for ki in k)**0.5
                    # Mask out values outside desired bin
                    return v * ((kk<=k_bin_edges[iii][1]) & (kk>=k_bin_edges[iii][0]))
            
                if verbose > 0 and comm.rank == 0: print('\tConstructing masked field for bin %d' % i)
            
                third_copy = self.third.view()
                third_copy._actions.append(('complex', mask, 'wavenumber'))
                delta_fields3[iii] = third_copy.paint(mode='real')

        if verbose > 0 and comm.rank == 0: print('\tStarting sum over triangles')
            
        # Now, loop over triangle bins
        B = 0.
        Ntri = 0.
        k1m = 0.
        k2m = 0.
        k3m = 0.
        for t,triple in enumerate(triple_index_list):
            
            # Compute unnormalized B value in triangle bin, using pre-computed delta fields
            # retrieved from dict
            if self.second is not None:
                B_local = np.sum(delta_fields[triple[0]]*delta_fields2[triple[1]]*delta_fields3[triple[2]]) \
                            * self.mesh.attrs['BoxSize'].prod()**2 \
                            / np.prod(self.mesh.attrs['Nmesh']) # / Nbin
            else:
                B_local = np.sum(delta_fields[triple[0]]*delta_fields[triple[1]]*delta_fields[triple[2]]) \
                            * self.mesh.attrs['BoxSize'].prod()**2 \
                            / np.prod(self.mesh.attrs['Nmesh']) # / Nbin
            B_sum = np.sum(comm.gather(B_local))
            B = comm.bcast(B_sum)
        
            # Convert units of B value to h^-1 Mpc^6
            B *= self.attrs['pos_units_mpcoverh']**6.
            
            # Add measurement info to stored arrays of previous measurements
            if self.b is None:
                self.b = { 'index': np.array([triangle_index_list[t]]),
                              'k_edge': np.array([triangle_bin_edge_list[t]]), 
                              'k_mean': np.array([0.]),
                              'B': np.array([B]),
                              'N_tri': np.array([Ntri]) }
            else:
                self.b['index'] = np.append(self.b['index'],[triangle_index_list[t]],axis=0)
                self.b['k_edge'] = np.append(self.b['k_edge'],[triangle_bin_edge_list[t]],axis=0)
                self.b['k_mean'] = np.append(self.b['k_mean'],[0.],axis=0)
                self.b['B'] = np.append(self.b['B'],[B],axis=0)
                self.b['N_tri'] = np.append(self.b['N_tri'],[Ntri],axis=0)
            
            # If out_file is specified, write measurement to disk by appending to file.
            # Perform this writing operation for every num_tri_to_write triangles
            # at a time, to lessen I/O
            if comm.rank == 0:
                if verbose > 0:
                    print('FFT Bk: %d %d %g %g %g %g %g' % (t,triangle_index_list[t],
                                                            self.b['k_mean'][-1][0],
                                                            self.b['k_mean'][-1][1],
                                                            self.b['k_mean'][-1][2],
                                                            self.b['B'][-1],
                                                             self.b['N_tri'][-1]) )
                if out_file is not None:
                    if (t > 0) and ((t+1) % num_tri_to_write == 0):
                        with open(out_file, "a") as f:
                            for tt in range(num_tri_to_write):
                                f.write('%d %e %e %e %e %e %e %e\n' 
                                        % (triangle_index_list[t-(num_tri_to_write-1)+tt],triangle_bin_edge_list[t-99+tt][0],
                                           triangle_bin_edge_list[t-(num_tri_to_write-1)+tt][1],triangle_bin_edge_list[t-99+tt][2],
                                           triangle_bin_edge_list[t-(num_tri_to_write-1)+tt][3],triangle_bin_edge_list[t-99+tt][4],
                                           triangle_bin_edge_list[t-(num_tri_to_write-1)+tt][5],self.b['B'][-num_tri_to_write+tt]))

        # After loop has finished, write any remaining triangles that have been computed
        # but have not yet been written out
        if comm.rank == 0 and out_file is not None and (t > 0) and ((t+1) % num_tri_to_write != 0):
            with open(out_file, "a") as f:
                t_left = (t+1) % num_tri_to_write
                for tt in range(t_left):
                    f.write('%d %e %e %e %e %e %e %e\n' 
                                % (triangle_index_list[t-t_left+1+tt],triangle_bin_edge_list[t-t_left+1+tt][0],
                                   triangle_bin_edge_list[t-t_left+1+tt][1],triangle_bin_edge_list[t-t_left+1+tt][2],
                                   triangle_bin_edge_list[t-t_left+1+tt][3],triangle_bin_edge_list[t-t_left+1+tt][4],
                                   triangle_bin_edge_list[t-t_left+1+tt][5],self.b['B'][-t_left+tt]))


        # Return dict of measurements made in this function call
        b_return = { 'index': self.b['index'][-len(triangle_index_list):],
                      'k_edge': self.b['k_edge'][-len(triangle_index_list):], 
                      'k_mean': self.b['k_mean'][-len(triangle_index_list):],
                      'B': self.b['B'][-len(triangle_index_list):],
                      'N_tri': self.b['N_tri'][-len(triangle_index_list):] }
        return b_return



    def measure_gridinfo_faster(self,
                           imin = None,
                           imax = None,
                           kmeas_min = None,
                           kmeas_max = None,
                           out_file = None,
                           verbose = 0):
        """Compute bispectrum binning info over range of triangle bins.
        
        The range of triangle bins to use is defined by their minimum and maximum indices
        in the stored `k_edges` array. In the future, the user will be able to specify min
        and max k values, but that feature has not been implemented yet.
        
        This routine only measures the triangle counts in bins and k_means in bins, to be
        saved and later combined with the unnormalized bispectrum values from
        measure_bispectrum_faster into a full set of measurements.
        
        Parameters
        ----------
        imin, imax : int
            minimum and maximum triangle bin indices to measure bispectrum between
        kmeas_min, kmeas_max : float, optional
            NOT IMPLEMENTED
        out_file : str, optional
            filename of output file to save intermittent results to
        verbose : int, optional
            verbosity level of output: 0 (default) for nothing, >0 for more than nothing
        meas_type : {'full', 'grid_info', 'unnorm_b_value'}
            measurement type, as described above
        
        Returns
        -------
        dict
            b dict (see doctring for FFTBispectrum class) containing information for
            bins looped over in this function call
        """
        
        num_tri_to_write = 100
        
        comm = CurrentMPIComm.get()
        
        n_mesh = self.mesh.attrs['Nmesh']
        box_size = self.mesh.attrs['BoxSize']
        
        # Generate lists of k bin edges and indices
        k_bin_edges = generate_bin_edge_list(self.attrs['kmin'],
                           self.attrs['kmax'],
                           self.attrs['dk'],
                           self.attrs['num_lowk_bins'],
                           self.attrs['dk_high'])
        k_bin_indices = np.arange(len(k_bin_edges))
        
        # Also fetch arrays of triangle bin edges and indices
        triangle_bin_edge_list = self.k_edges[imin:min(imax,len(self.k_edges))]
        triple_index_list = self.k_indices[imin:min(imax,len(self.k_indices))]
        triangle_index_list = range(imin,min(imax,len(self.k_indices)))
        
        # Make empty dicts to hold number_fields and k_fields
        number_fields = {}
        kk_fields = {}
        if verbose > 0 and comm.rank == 0: 
            print('Starting number_field and/or k_field computations')
            
        # Generate real-space number and k fields for each k bin, and add to dict, indexed
        # by k bin index
        for i in k_bin_indices:
            if verbose > 0 and comm.rank == 0:
                print('\tConstructing number_fields and/or k_fields for bin %d' % i)
            
            number_fields[i] = number_field(box_size,n_mesh,
                                            k_bin_edges[i][0],k_bin_edges[i][1])
            kk_fields[i] = k_field(box_size,n_mesh,
                                    k_bin_edges[i][0],k_bin_edges[i][1],1.)


        if verbose > 0 and comm.rank == 0: print('\tStarting sum over triangles')
            
        # Now, loop over triangle bins
        B = 0.
        Ntri = 0.
        k_mean_local = np.zeros(3)
        k_mean_sum = np.zeros(3)
        k_mean = np.zeros(3)
        for t,triple in enumerate(triple_index_list):
            
            # Compute number of triangles in bin
            Nbin_local = np.sum(number_fields[triple[0]] \
                                *number_fields[triple[1]] \
                                *number_fields[triple[2]]) / np.prod(n_mesh)
            Nbin_sum = np.sum(comm.gather(Nbin_local))
            Nbin = comm.bcast(Nbin_sum)

            # Compute mean values of k1,k2,k3 over triangle bin.
            # This may occasionally land on an empty bin (Nbin=0), so we catch
            # the corresponding error and set k_mean_local[0] = nan, since we won't
            # want to use those bins anyway.
            with np.errstate(divide='raise', invalid='raise'):
                try:
                    k_mean_local[0] = np.sum(kk_fields[triple[0]] \
                                             *number_fields[triple[1]] \
                                             *number_fields[triple[2]]) / np.prod(n_mesh) / Nbin
                except FloatingPointError:
                    k_mean_local[0] = np.nan

                try:
                    k_mean_local[1] = np.sum(number_fields[triple[0]] \
                                             *kk_fields[triple[1]] \
                                             *number_fields[triple[2]]) / np.prod(n_mesh) / Nbin
                except FloatingPointError:
                    k_mean_local[1] = np.nan
                    
                try:
                    k_mean_local[2] = np.sum(number_fields[triple[0]] \
                                             *number_fields[triple[1]] \
                                             *kk_fields[triple[2]]) / np.prod(n_mesh) / Nbin
                except FloatingPointError:
                    k_mean_local[2] = np.nan
                    
            for j in range(3):
                k_mean_sum[j] = np.sum(comm.gather(k_mean_local[j]))
                k_mean[j] = comm.bcast(k_mean_sum[j])
                
            # Convert units of k_means
            k_mean /= self.attrs['pos_units_mpcoverh']
            
            # Add measurement info to stored arrays of previous measurements
            if self.b is None:
                self.b = { 'index': np.array([triangle_index_list[t]]),
                              'k_edge': np.array([triangle_bin_edge_list[t]]), 
                              'k_mean': np.array([k_mean]),
                              'B': np.array([0.]),
                              'N_tri': np.array([Nbin]) }
            else:
                self.b['index'] = np.append(self.b['index'],[triangle_index_list[t]],axis=0)
                self.b['k_edge'] = np.append(self.b['k_edge'],[triangle_bin_edge_list[t]],axis=0)
                self.b['k_mean'] = np.append(self.b['k_mean'],[k_mean],axis=0)
                self.b['B'] = np.append(self.b['B'],[B],axis=0)
                self.b['N_tri'] = np.append(self.b['N_tri'],[Nbin],axis=0)
            
            if comm.rank == 0 and verbose > 1:
                print('Computed Bk for triangle index %d' % t)
            
            # If out_file is specified, write measurement to disk by appending to file.
            # Perform this writing operation for every num_tri_to_write triangles
            # at a time, to lessen I/O
            if comm.rank == 0:
                if verbose > 0 and ((t+1) % 100 == 0):
                    print('FFT Bk: %d %d %g %g %g %g %g' % (t,triangle_index_list[t],
                                                            self.b['k_mean'][-1][0],
                                                            self.b['k_mean'][-1][1],
                                                            self.b['k_mean'][-1][2],
                                                            self.b['B'][-1],
                                                             self.b['N_tri'][-1]) )
                if out_file is not None:
                    if (t > 0) and ((t+1) % num_tri_to_write == 0):
                        with open(out_file, "a") as f:
                            for tt in range(num_tri_to_write):
                                f.write('%d %e %e %e %e %e %e %e %e %e %e\n' 
                                 % (triangle_index_list[t-(num_tri_to_write-1)+tt],
                                    self.b['k_mean'][-num_tri_to_write+tt][0],
                                    self.b['k_mean'][-num_tri_to_write+tt][1],
                                    self.b['k_mean'][-num_tri_to_write+tt][2],
                                    triangle_bin_edge_list[t-(num_tri_to_write-1)+tt][0],
                                triangle_bin_edge_list[t-(num_tri_to_write-1)+tt][1],triangle_bin_edge_list[t-(num_tri_to_write-1)+tt][2],
                                triangle_bin_edge_list[t-(num_tri_to_write-1)+tt][3],triangle_bin_edge_list[t-(num_tri_to_write-1)+tt][4],
                                triangle_bin_edge_list[t-(num_tri_to_write-1)+tt][5],self.b['N_tri'][-num_tri_to_write+tt]))
                        if verbose > 0:
                            print('Wrote %d B values, starting with triangle index %d' % (num_tri_to_write,triangle_index_list[t-(num_tri_to_write-1)+tt]))

        # After loop has finished, write any remaining triangles that have been computed
        # but have not yet been written out
        if comm.rank == 0 and out_file is not None and (t > 0) and ((t+1) % num_tri_to_write != 0):
            with open(out_file, "a") as f:
                t_left = (t+1) % num_tri_to_write
                for tt in range(t_left):
                    f.write('%d %e %e %e %e %e %e %e %e %e %e\n' 
                                % (triangle_index_list[t-t_left+1+tt],
                                    self.b['k_mean'][-t_left+tt][0],
                                    self.b['k_mean'][-t_left+tt][1],
                                    self.b['k_mean'][-t_left+tt][2],
                                   triangle_bin_edge_list[t-t_left+1+tt][0],
                                   triangle_bin_edge_list[t-t_left+1+tt][1],triangle_bin_edge_list[t-t_left+1+tt][2],
                                   triangle_bin_edge_list[t-t_left+1+tt][3],triangle_bin_edge_list[t-t_left+1+tt][4],
                                   triangle_bin_edge_list[t-t_left+1+tt][5],self.b['N_tri'][-t_left+tt]))

        # Return dict of measurements made in this function call
        b_return = { 'index': self.b['index'][-len(triangle_index_list):],
                      'k_edge': self.b['k_edge'][-len(triangle_index_list):], 
                      'k_mean': self.b['k_mean'][-len(triangle_index_list):],
                      'B': self.b['B'][-len(triangle_index_list):],
                      'N_tri': self.b['N_tri'][-len(triangle_index_list):] }
        return b_return