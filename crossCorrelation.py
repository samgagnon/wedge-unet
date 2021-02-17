
import numpy as np
from numpy.fft import fftn, fftshift
from numpy.linalg import norm

class crossCorrelation():

    def __init__(self, fields, k_bins = None, n_bins = None, L = 300*(128/200), do_ft = True):
        '''
        Creates a power spectrum object, with methods that will output either a 
        regular 1d power spectrum or a cylindrical power spectrum. 

        Parameters
        ----------
        fields: list of numpy ndarrays
            fields to cross-correlate

        k_bins: NoneType or 1darray
            if not None, fourier space bin edges for isotropic averaging. 
            the default is 13 logarithmically spaced bins as specified below

        n_bins: NoneType or int
            if want uniformly spaced bins, this will generate bin edges from 
            the minimum k to max k as specified by box specs

        L: int or float
            real space resolution of box in Mpc

        Methods
        -------
        compute_pspec: 
            returns the average k value going into each bin, and the power of 
            that bin

        cylindrical_pspec:
            for a 3D box, generates cylindrical power spectrum (also called 2D
            power spectrum). 
        '''

        #initializing some attributes
        self.fields = fields
        self.k_bins = k_bins
        self.n_bins = n_bins
        
        #----------------------------- box specs ------------------------------#
        self.L = L
        self.ndims = len(fields[0].shape)
        self.n = fields[0].shape[0] #number of pixels along one axis
        self.survey_size = (self.n**self.ndims)#volume of box
        self.origin = self.n//2 #origin by fft conventions


        self.delta_k = 2*np.pi/self.L #kspace resolution of 1 pixel
        self.delta_r = self.L/self.n #real space resolution of 1 pixel

        self.rmax = (self.n - self.origin)*self.delta_k #max radius

        #--------------------- Power spectrum attributes ----------------------#

        self.get_bins() #defining bins variable according to specifications

        if do_ft:
            self.field_fourier = [None]*2
            for i in range(len(self.fields)):
                self.fourier(i)
            P_ij = np.conj(self.field_fourier[0]) * self.field_fourier[1]
            P_ii = np.conj(self.field_fourier[0]) * self.field_fourier[0]
            P_jj = np.conj(self.field_fourier[1]) * self.field_fourier[1]
            self.cross_correlation = P_ij/np.sqrt(P_jj*P_ii)
            # self.phasespec = np.angle(self.cross_correlation)
            # print(self.phasespec)
            self.norm_cross_corr = self.cross_correlation #/ (self.field_fourier[0] * self.field_fourier[1])
            # self.norm_cross_corr = self.phasespec

        self.del_squared = False
        self.ignore_0 = False

        self.grid() #sets up grid of radial distances
        
        self.sort() #sorting radii + field vals (increasing)
        
        #indices determing which elements go into bins
        self.bin_ind = self.get_bin_ind(self.bins, self.r_sorted)
        
        #computing average of bins
        self.field_bins = self.average_bins(self.bin_ind, self.vals_sorted) 
        self.average_k = self.average_bins(self.bin_ind, self.r_sorted)

        self.power = self.field_bins


    #============================== INIT METHODS ==============================#
    
    def fourier(self, index):
        '''computes the fourier transform of the field'''

        fourier_transform = fftshift(fftn(fftshift(self.fields[index])))
        scaled = fourier_transform*(self.delta_r**self.ndims) #scaling factor
        self.field_fourier[index] = scaled

    def get_bins(self):
        '''gets the bin edges'''

        if self.k_bins is not None:
            self.bins = self.k_bins

        elif self.n_bins is not None:
            self.bins = np.linspace(self.delta_k, self.rmax, self.n_bins)

        else: #default bins
            # self.bins = np.logspace(np.log10(0.021), np.log10(self.rmax), num=100)
            self.bins = np.linspace(2*self.delta_k, self.rmax, num=100)

    #======================= METHODS THAT OUTPUT BOXES ========================#

    def compute_pspec(self, del_squared = True, ignore_0 = False, return_k = True,
    normalize = True):
        ''' 
        computes the power spectrum. 

        Parameters
        ----------
        del_squared: Bool
            set to True if want the delta squared quantity rather than raw power

        ignore_0: Bool
            set to True if want to exclude the zero k vector in the averaging

        Returns
        -------
        average_k: 1darray
            the average value of k for the bins

        power: 1darray
            the power (or delta squared) of each bin
        '''
        self.ignore_0 = ignore_0
        self.del_squared = del_squared
        self.normalize = normalize 

        self.p_spec()

        if return_k:
            return self.average_k, self.power
        else:
            return self.power

    def compute_cylindrical_pspec(self, ignore_0 = False, k_perp_bins = None,
    k_par_bins = None, delta_squared = False, return_bins = False):

        ''' 
        computes cylindrical power spectrum
        
        Parameters
        ----------
        ignore_0: Bool
            set to True if wish to ignore the 0 vector
            
        k_perp_bins: NoneType or ndarray
            the bin edges in the perpendicular direction. Default is 13 uniformly
            spaced bins. 
            
        k_par_bins: NoneType or ndarray
            the bin edges in the parallel direction. Default is 13 uniformly 
            spaced bins.
            
        delta_squared: Bool
            set to True if wish to compute delta squared quantity instead
            
        return_bins: Bool
            set to True if want the bin edges for both k perp and k parallel
        
        Returns:
        --------
        cyl_power: 2darray
            the cylindrical power spectrum, where the vertical axis is k parallel
            and the horizontal axis is k perpendicular
        
        k_par_bins: 1darray
            bin edges for the k parallel direction
    
        k_perp_bins: 1darray
            bin edges for the k perpendicular direction
        '''

        #initializing attributes
        self.delsq = delta_squared
        self.ignore_0_cyl = ignore_0 
        self.k_par_bins = k_par_bins
        self.k_perp_bins = k_perp_bins

        self.get_cyl_bins() #getting bins according to specifications

        self.cyl_pspec() #computing cylindrical power spectrum

        if return_bins:
            return self.k_par_bins, self.k_perp_bins, self.cyl_power
        else:
            return self.cyl_power

    #=============== METHODS RELATED TO VANILLA POWER SPECTRUM ================#

    def p_spec(self):
        '''Main method of power spectrum compuation. Organizes functions.'''

        self.grid() #sets up grid of radial distances
        
        self.sort() #sorting radii + field vals (increasing)
        
        #indices determing which elements go into bins
        self.bin_ind = self.get_bin_ind(self.bins, self.r_sorted)
        
        #computing average of bins
        self.field_bins = self.average_bins(self.bin_ind, self.vals_sorted) 
        self.average_k = self.average_bins(self.bin_ind, self.r_sorted)
        
        if self.normalize:
            self.power = self.field_bins/self.survey_size
        else:
            self.power = self.field_bins
        
        if self.del_squared:
            self.power /= 2*np.pi**2

    def grid(self):
        '''
        Generates a fourier space grid with spacing set by box specs, and finds 
        radial distance of each pixel from origin. Useful attribute created:

        radii: numpy ndarray 
            grid that contains radial distance of each pixel from origin, 
            in kspace units
        '''

        indices = (np.indices(self.fields[0].shape) - self.origin)*self.delta_k
        self.radii = norm(indices, axis = 0)
        
        if self.del_squared:
                self.norm_cross_corr *= self.radii**3
                
    def sort(self):
        ''' 
        Sorts radii, and the field value corresponding to each radius in 
        ascending order. sort_ind is here so as to not lose track of which 
        radius corresponds to which field value. Attributes created:
        
        r_sorted: 1darray
            the distances of each pixel sorted in ascending order
            
        vals_sorted: 1darray
            the pixel values sorted, where vals_sorted[i] is the value in the 
            pixel corresponding to r_sorted[i].
        '''

        sort_ind = np.argsort(self.radii.flat)
        self.r_sorted = self.radii.flat[sort_ind]
        self.vals_sorted = self.norm_cross_corr.flat[sort_ind] 

        if self.ignore_0: #excluding zero vector
            self.r_sorted = self.r_sorted[1:]
            self.vals_sorted = self.vals_sorted[1:]
           
#=============== METHODS RELATED TO CYLINDRICAL POWER SPECTRUM ================#
    
    def get_cyl_bins(self):
        '''Getting the bins for k perp and k parallel '''

        if self.k_par_bins is not None:
            self.k_par_bins = self.k_par_bins
        
        else: #default 
            self.k_par_bins = np.linspace(self.delta_k, self.rmax, 12)

        if self.k_perp_bins is not None:
            self.k_perp_bins = self.k_perp_bins
        
        else: #default
            self.k_perp_bins = np.linspace(self.delta_k, self.rmax, 12)

    def cyl_pspec(self):
        '''Main method of the cylindrical power spectrum''' 

        self.compute_kperp_pspecs() #for every fixed k_par slice, compute pspec
        self.sort_kpar() #sort according to k_par
        self.bin_kpar() #bin k_perp power spectra according to k_par bins

        self.cyl_power = self.k_par_averaged/self.survey_size
        
    def compute_kperp_pspecs(self):
        '''
        abs_squared[i] is a 2D slice at fixed k parallel. For every slice, 
        a 2D power spectrum is computed and stored. Attribute created: 

        k_perp_power: ndarray
            k_perp_power[i] contains the 2D pspec of abs_squared[i]
        '''

        k_perp_power = []
        for k_perp_slice in np.rollaxis(self.norm_cross_corr, 0):
            
            spec = crossCorrelation(k_perp_slice, k_bins = self.k_perp_bins, 
            do_ft= False) #power spectrum object
            
            power = spec.compute_pspec(del_squared= self.delsq, 
            ignore_0= self.ignore_0_cyl, return_k= False, normalize= False)
            
            k_perp_power.append(power)
        
        self.k_perp_power = np.array(k_perp_power)

    def sort_kpar(self):
        '''
        Sorting the 2D k perp pspecs according to k_par. Generates the attributes: 

        k_par_radii: 1darray
            the k parallel vectors as defined by grid spacing

        k_par_sorted: 1darray
            self explanatory. 

        k_perp_sorted: 1darray
            very cryptic name, what ever can it mean?
        '''
        
        #values of k_parallel set by grid spacing
        self.k_par = np.arange(-self.n//2,self.n//2)*self.delta_k
        self.k_par_radii = np.abs(self.k_par)

        #sorting
        self.sort_ind = np.argsort(self.k_par_radii)
        self.k_par_sorted = self.k_par_radii[self.sort_ind]
        self.k_perp_sorted = self.k_perp_power[self.sort_ind]

        if self.ignore_0_cyl:
            self.k_par_sorted = self.k_par_sorted[1:]
            self.k_perp_sorted = self.k_perp_sorted[1:]

    def bin_kpar(self):
        ''' 
        Given the bin edges for k parallel, bins the 2D power spectra accordingly.
        '''
        self.k_par_bin_ind = self.get_bin_ind(self.k_par_bins, self.k_par_sorted)
        
        self.k_par_averaged = self.average_bins(self.k_par_bin_ind, self.k_perp_sorted,
        cylindrical= True)

#============================ BINNING FUNCTIONS ===============================#

    def get_bin_ind(self, bins, values):
        '''
        Given bins in kspace and array of values in k space, determines the 
        last index of the array going into each bin.
        
        Parameters: 
        -----------
        bins: 1darray 
            the bin edges in k space units
            
        values: 1darray
            the values, in kspace units, which we wish to bin. Usually radius
            vectors.
        
        Returns:
        --------
        bin_indices: 1darray
            bin_indices[i] contains the index of the last element of values going
            into the ith bin. 

            i.e., the first bin will contain values[:bin_indices[1]+1], the 
            second bin will contain values[bin_indices[1]+1:bin_indices[2]+1], 
            and etc.
        '''
        bin_indices = [0]
        for bin_val in bins:
            val = np.argmax( values > bin_val)
            if val == 0: #ie bin_val > r_max
                val = len(values)
            
            bin_indices.append(val-1)
        return np.array(bin_indices)

    def average_bins(self, bin_indices, values, cylindrical = False):
        ''' 
        puts things in bins, averages the bins. Does it all in one shot 
        with cumsum which is fast, but hard to read. Nice explanation coming one 
        day maybe when I have time?

        Parameters:
        -----------
        bin_indices: 1darray
            indices determining which elements go into bins, generated in 
            get_bin_ind function above

        values: ndarray
            values to be binned

        cylindrical: Bool
            sadly can't do the same exact procedure for cylindrical pspec and 
            regular pspec. You know when to set this variable to True. 
        
        Returns: 
        --------
        averaged_bins: ndarray
            values binned and averaged!
        '''

        if cylindrical: #very cryptic but elegant (?) code
            cumulative_sum = np.cumsum(values, axis = 0)
            bin_sums = cumulative_sum[bin_indices[1:],:]
            bin_sums[1:] -= bin_sums[:len(bin_sums)-1]

            bin_dims = bin_indices[1:] - bin_indices[:len(bin_indices)-1]  
            
            #actually im pretty sure everything is the same but this line...
            #fix this eventually, not crucial
            averaged_bins = bin_sums/bin_dims.reshape(len(bin_dims), 1) 
       
        else:
            cumulative_sum = np.cumsum(values)
            bin_sums = cumulative_sum[bin_indices[1:]]
            bin_sums[1:] -= bin_sums[:len(bin_sums)-1]

            bin_dims = bin_indices[1:] - bin_indices[:len(bin_indices)-1]
            averaged_bins = bin_sums/bin_dims

        return averaged_bins    

if __name__ == '__main__':
    from utils import *
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from skimage.transform import resize
    from powerSpectrum import PowerSpectrum

    def define_files(noise):
        if noise == "HERA":
            pred785 = load_binary_data("../../unet/run_catalogue/992020.1_barHERA/val-predict-delta_T_v3_z007.85_nf0.473681_useTs0_200_300Mpc", np.float64).reshape((128,32,32))
            pred800 = load_binary_data("../../unet/run_catalogue/992020.1_barHERA/val-predict-delta_T_v3_z008.00_nf0.521530_useTs0_200_300Mpc", np.float64).reshape((128,32,32))
            pred810 = load_binary_data("../../unet/run_catalogue/992020.1_barHERA/val-predict-delta_T_v3_z008.10_nf0.544216_useTs0_200_300Mpc", np.float64).reshape((128,32,32))
            pred820 = load_binary_data("../../unet/run_catalogue/992020.1_barHERA/val-predict-delta_T_v3_z008.20_nf0.593328_useTs0_200_300Mpc", np.float64).reshape((128,32,32))
            pred830 = load_binary_data("../../unet/run_catalogue/992020.1_barHERA/val-predict-delta_T_v3_z008.30_nf0.614770_useTs0_200_300Mpc", np.float64).reshape((128,32,32))
            pred850 = load_binary_data("../../unet/run_catalogue/992020.1_barHERA/val-predict-delta_T_v3_z008.50_nf0.655648_useTs0_200_300Mpc", np.float64).reshape((128,32,32))
            
            pred785 = resize(pred785, (128,128,128))
            pred800 = resize(pred800, (128,128,128))
            pred810 = resize(pred810, (128,128,128))
            pred820 = resize(pred820, (128,128,128))
            pred830 = resize(pred830, (128,128,128))
            pred850 = resize(pred850, (128,128,128))

            mask785 = load_binary_data("HERAconv/HERAconv-delta_T_v3_z007.85_nf0.473681_useTs0_200_300Mpc", np.float32).reshape((82,127,127))[:,:29,:29]
            mask800 = load_binary_data("HERAconv/HERAconv-delta_T_v3_z008.00_nf0.521530_useTs0_200_300Mpc", np.float32).reshape((82,127,127))[:,:29,:29]
            mask810 = load_binary_data("HERAconv/HERAconv-delta_T_v3_z008.10_nf0.544216_useTs0_200_300Mpc", np.float32).reshape((82,127,127))[:,:29,:29]
            mask820 = load_binary_data("HERAconv/HERAconv-delta_T_v3_z008.20_nf0.593328_useTs0_200_300Mpc", np.float32).reshape((82,127,127))[:,:29,:29]
            mask830 = load_binary_data("HERAconv/HERAconv-delta_T_v3_z008.30_nf0.614770_useTs0_200_300Mpc", np.float32).reshape((82,127,127))[:,:29,:29]
            mask850 = load_binary_data("HERAconv/HERAconv-delta_T_v3_z008.50_nf0.655648_useTs0_200_300Mpc", np.float32).reshape((82,127,127))[:,:29,:29]

            mask785 = resize(mask785, (128,128,128))
            mask800 = resize(mask800, (128,128,128))
            mask810 = resize(mask810, (128,128,128))
            mask820 = resize(mask820, (128,128,128))
            mask830 = resize(mask830, (128,128,128))
            mask850 = resize(mask850, (128,128,128))

            mask785 = mask785 - mask785.min()
            mask785 = mask785 / mask785.max()
            mask800 = mask800 - mask800.min()
            mask800 = mask800 / mask800.max()
            mask810 = mask810 - mask810.min()
            mask810 = mask810 / mask810.max()
            mask820 = mask820 - mask820.min()
            mask820 = mask820 / mask820.max()
            mask830 = mask830 - mask830.min()
            mask830 = mask830 / mask830.max()
            mask850 = mask850 - mask850.min()
            mask850 = mask850 / mask850.max()

            # plt.imshow(mask785[0], cmap='hot')
            # plt.show()

            print(mask785.max(), mask785.min(), mask785.mean())

            # plt.imshow(pred785[0], cmap='hot')
            # plt.show()

            print(pred785.max(), pred785.min(), pred785.mean())

        elif noise == "Null":
            pred785 = load_binary_data("../../unet/run_catalogue/8312020_3d_null/val-predict-delta_T_v3_z007.85_nf0.473681_useTs0_200_300Mpc", np.float32).reshape((128,128,128))
            pred800 = load_binary_data("../../unet/run_catalogue/8312020_3d_null/val-predict-delta_T_v3_z008.00_nf0.521530_useTs0_200_300Mpc", np.float32).reshape((128,128,128))
            pred810 = load_binary_data("../../unet/run_catalogue/8312020_3d_null/val-predict-delta_T_v3_z008.10_nf0.544216_useTs0_200_300Mpc", np.float32).reshape((128,128,128))
            pred820 = load_binary_data("../../unet/run_catalogue/8312020_3d_null/val-predict-delta_T_v3_z008.20_nf0.593328_useTs0_200_300Mpc", np.float32).reshape((128,128,128))
            pred830 = load_binary_data("../../unet/run_catalogue/8312020_3d_null/val-predict-delta_T_v3_z008.30_nf0.614770_useTs0_200_300Mpc", np.float32).reshape((128,128,128))
            pred850 = load_binary_data("../../unet/run_catalogue/8312020_3d_null/val-predict-delta_T_v3_z008.50_nf0.655648_useTs0_200_300Mpc", np.float32).reshape((128,128,128))

            mask785 = load_binary_data("scramble_masks/scramble-uniform-delta_T_v3_z007.85_nf0.473681_useTs0_200_300Mpc").reshape((200,200,200))
            mask800 = load_binary_data("scramble_masks/scramble-uniform-delta_T_v3_z008.00_nf0.521530_useTs0_200_300Mpc").reshape((200,200,200))
            mask810 = load_binary_data("scramble_masks/scramble-uniform-delta_T_v3_z008.10_nf0.544216_useTs0_200_300Mpc").reshape((200,200,200))
            mask820 = load_binary_data("scramble_masks/scramble-uniform-delta_T_v3_z008.20_nf0.593328_useTs0_200_300Mpc").reshape((200,200,200))
            mask830 = load_binary_data("scramble_masks/scramble-uniform-delta_T_v3_z008.30_nf0.614770_useTs0_200_300Mpc").reshape((200,200,200))
            mask850 = load_binary_data("scramble_masks/scramble-uniform-delta_T_v3_z008.50_nf0.655648_useTs0_200_300Mpc").reshape((200,200,200))

            mask785 = resize(mask785, (128,128,128))
            mask800 = resize(mask800, (128,128,128))
            mask810 = resize(mask810, (128,128,128))
            mask820 = resize(mask820, (128,128,128))
            mask830 = resize(mask830, (128,128,128))
            mask850 = resize(mask850, (128,128,128))

        elif noise == "Null2":
            pred785 = load_binary_data("../../unet/run_catalogue/12302020_null2/val-predict-delta_T_v3_z007.85_nf0.473681_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred800 = load_binary_data("../../unet/run_catalogue/12302020_null2/val-predict-delta_T_v3_z008.00_nf0.521530_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred810 = load_binary_data("../../unet/run_catalogue/12302020_null2/val-predict-delta_T_v3_z008.10_nf0.544216_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred820 = load_binary_data("../../unet/run_catalogue/12302020_null2/val-predict-delta_T_v3_z008.20_nf0.593328_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred830 = load_binary_data("../../unet/run_catalogue/12302020_null2/val-predict-delta_T_v3_z008.30_nf0.614770_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred850 = load_binary_data("../../unet/run_catalogue/12302020_null2/val-predict-delta_T_v3_z008.50_nf0.655648_useTs0_200_300Mpc", np.float64).reshape((128,128,128))

            mask785 = load_binary_data("shuffle/shuffle-delta_T_v3_z007.85_nf0.473681_useTs0_200_300Mpc", np.float64).reshape((200,200,200))
            mask800 = load_binary_data("shuffle/shuffle-delta_T_v3_z008.00_nf0.521530_useTs0_200_300Mpc", np.float64).reshape((200,200,200))
            mask810 = load_binary_data("shuffle/shuffle-delta_T_v3_z008.10_nf0.544216_useTs0_200_300Mpc", np.float64).reshape((200,200,200))
            mask820 = load_binary_data("shuffle/shuffle-delta_T_v3_z008.20_nf0.593328_useTs0_200_300Mpc", np.float64).reshape((200,200,200))
            mask830 = load_binary_data("shuffle/shuffle-delta_T_v3_z008.30_nf0.614770_useTs0_200_300Mpc", np.float64).reshape((200,200,200))
            mask850 = load_binary_data("shuffle/shuffle-delta_T_v3_z008.50_nf0.655648_useTs0_200_300Mpc", np.float64).reshape((200,200,200))

            mask785 = resize(mask785, (128,128,128))
            mask800 = resize(mask800, (128,128,128))
            mask810 = resize(mask810, (128,128,128))
            mask820 = resize(mask820, (128,128,128))
            mask830 = resize(mask830, (128,128,128))
            mask850 = resize(mask850, (128,128,128))

        elif noise == "SKA":
            # pred785 = load_binary_data("../../unet/run_catalogue/2102021_SKA_val/val-predict-delta_T_v3_z007.85_nf0.473681_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            # pred785 = load_binary_data("../../unet/run_catalogue/2112021_SKA_val/val-predict-delta_T_v3_z008.00_nf0.473681_useTs0_200_300Mpc", np.float64).reshape((128,128,128))            
            # Best is below
            # pred785 = load_binary_data("../../unet/run_catalogue/2152021_oldnet_val/val-predict-delta_T_v3_z007.85_nf0.473681_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            # pred785 = load_binary_data("../../unet/run_catalogue/2152021_oldnet_val/val-predict-manual7.85", np.float64).reshape((128,128,128))
            pred785 = load_binary_data("../../unet/run_catalogue/2142021_nobad_manualval/val-predict-bad8.00", np.float64).reshape((128,128,128))
            # pred785 = load_binary_data("../../unet/run_catalogue/2142021_oldnet_manualval/val-predict-delta_T_v3_z008.00_nf0.511529_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            # pred785 = load_binary_data("../../unet/run_catalogue/2142021_oldnet_manualval/val-predict-manual7.85", np.float64).reshape((128,128,128))
            # pred785 = load_binary_data("../../unet/run_catalogue/2152021_nobad_val/val-predict-delta_T_v3_z008.00_nf0.473681_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            # pred785 = load_binary_data("../../unet/run_catalogue/2142021_nobad_manualval/val-predict-bad8.00", np.float64).reshape((128,128,128))
            pred800 = load_binary_data("../../unet/run_catalogue/2102021_SKA_val/val-predict-delta_T_v3_z008.00_nf0.521530_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred810 = load_binary_data("../../unet/run_catalogue/1232020_isensee_SKA/val-predict-delta_T_v3_z008.10_nf0.544216_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred820 = load_binary_data("../../unet/run_catalogue/1232020_isensee_SKA/val-predict-delta_T_v3_z008.20_nf0.593328_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred830 = load_binary_data("../../unet/run_catalogue/1232020_isensee_SKA/val-predict-delta_T_v3_z008.30_nf0.614770_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred850 = load_binary_data("../../unet/run_catalogue/1232020_isensee_SKA/val-predict-delta_T_v3_z008.50_nf0.655648_useTs0_200_300Mpc", np.float64).reshape((128,128,128))

            # mask785 = load_binary_data("SKA_conv/chop-SKAconv-delta_T_v3_z007.85_nf0.473681_useTs0_200_300Mpc", np.float32).reshape((128,128,128))
            mask800 = load_binary_data("SKA_conv/chop-SKAconv-delta_T_v3_z008.00_nf0.521530_useTs0_200_300Mpc", np.float32).reshape((128,128,128))
            # mask785 = load_binary_data("SKA_conv/chop-SKAconv-delta_T_v3_z008.00_nf0.511529_useTs0_200_300Mpc", np.float32).reshape((128,128,128))
            mask785 = load_binary_data("SKA_masks/chop-SKAinterp-delta_T_v3_z008.00_nf0.511529_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            mask810 = load_binary_data("SKA_conv/chop-SKAconv-delta_T_v3_z008.10_nf0.544216_useTs0_200_300Mpc", np.float32).reshape((128,128,128))
            mask820 = load_binary_data("SKA_conv/chop-SKAconv-delta_T_v3_z008.20_nf0.593328_useTs0_200_300Mpc", np.float32).reshape((128,128,128))
            mask830 = load_binary_data("SKA_conv/chop-SKAconv-delta_T_v3_z008.30_nf0.614770_useTs0_200_300Mpc", np.float32).reshape((128,128,128))
            mask850 = load_binary_data("SKA_conv/chop-SKAconv-delta_T_v3_z008.50_nf0.655648_useTs0_200_300Mpc", np.float32).reshape((128,128,128))

        else:

            pred785 = load_binary_data("../../unet/run_catalogue/9102020_isensee_bar/val-predict-delta_T_v3_z007.85_nf0.473681_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred800 = load_binary_data("../../unet/run_catalogue/9102020_isensee_bar/val-predict-delta_T_v3_z008.00_nf0.521530_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred810 = load_binary_data("../../unet/run_catalogue/9102020_isensee_bar/val-predict-delta_T_v3_z008.10_nf0.544216_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred820 = load_binary_data("../../unet/run_catalogue/9102020_isensee_bar/val-predict-delta_T_v3_z008.20_nf0.593328_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred830 = load_binary_data("../../unet/run_catalogue/9102020_isensee_bar/val-predict-delta_T_v3_z008.30_nf0.614770_useTs0_200_300Mpc", np.float64).reshape((128,128,128))
            pred850 = load_binary_data("../../unet/run_catalogue/9102020_isensee_bar/val-predict-delta_T_v3_z008.50_nf0.655648_useTs0_200_300Mpc", np.float64).reshape((128,128,128))

            mask785 = load_binary_data("../../unet/masks3d/delta_T_v3_z007.85_nf0.473681_useTs0_200_300Mpc").reshape((200,200,200))
            mask800 = load_binary_data("../../unet/masks3d/delta_T_v3_z008.00_nf0.521530_useTs0_200_300Mpc").reshape((200,200,200))
            mask810 = load_binary_data("../../unet/masks3d/delta_T_v3_z008.10_nf0.544216_useTs0_200_300Mpc").reshape((200,200,200))
            mask820 = load_binary_data("../../unet/masks3d/delta_T_v3_z008.20_nf0.593328_useTs0_200_300Mpc").reshape((200,200,200))
            mask830 = load_binary_data("../../unet/masks3d/delta_T_v3_z008.30_nf0.614770_useTs0_200_300Mpc").reshape((200,200,200))
            mask850 = load_binary_data("../../unet/masks3d/delta_T_v3_z008.50_nf0.655648_useTs0_200_300Mpc").reshape((200,200,200))

            mask785 = resize(mask785, (128,128,128))
            mask800 = resize(mask800, (128,128,128))
            mask810 = resize(mask810, (128,128,128))
            mask820 = resize(mask820, (128,128,128))
            mask830 = resize(mask830, (128,128,128))
            mask850 = resize(mask850, (128,128,128))

        crosCor785 = crossCorrelation([mask785<0.5, pred785])
        crosCor800 = crossCorrelation([mask800<0.5, pred800])
        crosCor810 = crossCorrelation([mask810<0.5, pred810])
        crosCor820 = crossCorrelation([mask820<0.5, pred820])
        crosCor830 = crossCorrelation([mask830<0.5, pred830])
        crosCor850 = crossCorrelation([mask850<0.5, pred850])
        norm785 = crosCor785.power
        norm800 = crosCor800.power
        norm810 = crosCor810.power
        norm820 = crosCor820.power
        norm830 = crosCor830.power
        norm850 = crosCor850.power
        norm_list = [norm785, norm800, norm810, norm820, norm830, norm850]
        k = crosCor850.average_k
        return k, norm_list

    # plt.rc('text', usetex=True )
    mpl.rc('font', family='Times New Roman')
    plt.rcParams['mathtext.fontset'] = 'stix'

    files_used = ["Noiseless", "HERA", "SKA"]

    fig, ax = plt.subplots(len(files_used))
    fig.set_figheight(13)
    fig.set_figwidth(7)
    # plt.tick_params(axis='both', which='major', labelsize=15)
    for i in range(len(files_used)):
        k, norm_list = define_files(files_used[i])
        ax[i].grid(True)
        ax[i].tick_params(axis='both', which='major', labelsize=15)
        ax[i].plot(k[1:], norm_list[0][1:], label='$z$ = 7.85')
        ax[i].plot(k[1:], norm_list[1][1:], label='$z$ = 8.00')
        ax[i].plot(k[1:], norm_list[2][1:], label='$z$ = 8.10')
        ax[i].plot(k[1:], norm_list[3][1:], label='$z$ = 8.20')
        ax[i].plot(k[1:], norm_list[4][1:], label='$z$ = 8.30')
        ax[i].plot(k[1:], norm_list[5][1:], label='$z$ = 8.50')
    props = dict(boxstyle='square', facecolor='white', alpha=1.0)
    ax[0].text(0.75, 0.90, 'Noiseless', transform=ax[0].transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
    ax[1].text(0.75, 0.90, 'HERA', transform=ax[1].transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
    ax[2].text(0.75, 0.90, 'SKA', transform=ax[2].transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
    ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=20)
    ax[1].set_ylabel('Normalized Cross-Power', fontname='Times New Roman', fontsize=25, labelpad=10)
    ax[len(files_used)-1].set_xlabel('$k$ [Mpc$^{-1}$]', fontname='Times New Roman', fontsize=25)
    plt.savefig('tall-cross-correlation.pdf', format='pdf')
    plt.show()