"""
All the functions used to compute the wave spectrum are available in this script!
"""


from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import struct
from os import path   
from scipy.stats import binned_statistic_2d


def load_camera_mesh( meshfile ):
    with open(meshfile, "rb") as mf:
        npts = struct.unpack( "I", mf.read( 4 ) )[0]
        limits = np.array( struct.unpack( "dddddd", mf.read( 6*8 ) ) )
        Rinv = np.reshape( np.array(struct.unpack("ddddddddd", mf.read(9*8) )), (3,3) )
        Tinv = np.reshape( np.array(struct.unpack("ddd", mf.read(3*8) )), (3,1) ) 
                
        data = np.reshape( np.array( bytearray(mf.read( npts*3*2 )), dtype=np.uint8 ).view(dtype=np.uint16), (3,npts), order="F" )
        
        mesh_cam = data.astype( np.float32 )
        mesh_cam = mesh_cam / np.expand_dims( limits[0:3], axis=1) + np.expand_dims( limits[3:6], axis=1 );
        mesh_cam = Rinv@mesh_cam + Tinv;
    
        return mesh_cam
    
    
def run_interpolation(X,Y,Z):
    '''
    creates an interpolator
    '''
    Xflat = list(X.flatten())
    Yflat = list(Y.flatten())
    Zflat = list(Z.flatten())
    interpolator=interpolate.LinearNDInterpolator(np.array([Xflat,Yflat]).T,Zflat)
    return interpolator


def ffgrid(x,y,z, xvec, yvec, dx,dy):
    
    """
    The input:
        The function recieves ungridded x, y, z as 1-D arrays, 
        generates a grid of resolution specified by the user as 
        xvec & yvec. Note, xvec & yvec should be center bins.

    The ouput: 
        The gridded Z on the X,Y space 
    """

    X,Y = np.meshgrid(xvec, yvec)

    #establish edge bins
    x_edge = xvec - (dx/2.)
    x_edge = np.append(x_edge, max(x_edge)+ dx)

    y_edge = yvec - (dy/2.)
    y_edge = np.append(y_edge, max(y_edge)+ dy)

    #call binning function
    ret = binned_statistic_2d(x, y, z, 'mean', bins=[x_edge, y_edge], 
        expand_binnumbers=False)

    Z = ret.statistic
    Z = Z.T #need to transpose to match the meshgrid orientation with is (J,I)
    return X, Y, Z 


def grid_interpolate_wassXYZ(wass_frame, num, all_planes, 
                           baseline =2.5, distance = 70, xcentre = 0, 
                           ycentre = -30, N = 1024, checks=0):
    '''
    The function loads the raw xyz, removes the mean plane, bin grid and interpolate to 20cm spatial resolation.
    it returns the gridded XYZ only. 
    
    NB: y distance is -ve from camera
    '''

    meshname = path.join(wass_frame,"mesh_cam.xyzC")
    #print("Loading ", meshname )
    P1Cam =  np.vstack( (np.loadtxt( path.join(wass_frame,"P1cam.txt"))  ,[0, 0, 0, 1] ) )
 
    xyz = load_camera_mesh(meshname)   
    assert len(all_planes)==4, "Plane must be a 4-element vector"
    a=all_planes[0]
    b=all_planes[1]
    c=all_planes[2]
    d=all_planes[3];
    q = (1-c)/(a*a + b*b)
    Rpl=np.array([[1-a*a*q, -a*b*q, -a], [-a*b*q, 1-b*b*q, -b], [a, b, c] ] )
    Tpl=np.expand_dims( np.array([0,0,d]), axis=1)
    
    #remove the plane
    assert xyz.shape[0]==3, "Mesh must be a 3xN numpy array"    
    xyzp = (Rpl@xyz + Tpl)*baseline
    
    #extract the wass xyz
    x = xyzp[0,:]
    y = xyzp[1,:]
    zp = xyzp[2,:]

    xmax = xcentre + distance/2
    xmin = xcentre - distance/2
    ymax = ycentre + distance/2
    ymin = ycentre - distance/2

    #xvec , yvec are centre bins
    xvector = np.linspace(xmin,xmax,N)
    yvector = np.linspace(ymin,ymax,N)
    dx = abs(np.mean(np.diff(xvector)))
    dy = abs(np.mean(np.diff(yvector)))

    #run the ffgridding function
    xx, yy, zz = ffgrid(x, y, zp, xvector, yvector, dx, dy)
    x_gridded = np.squeeze(xx.flatten())
    y_gridded = np.squeeze(yy.flatten())
    z_gridded = np.squeeze(zz.flatten())
    ind_nonans = np.where(~np.isnan(z_gridded)) #indices that are free from nans

    #run our interp fc
    interpolator_z = run_interpolation(x_gridded[ind_nonans], y_gridded[ind_nonans], z_gridded[ind_nonans])
    ind_nan = np.where(np.isnan(z_gridded))
    Zi = z_gridded
    Zi[ind_nan] = interpolator_z(x_gridded[ind_nan], y_gridded[ind_nan])
    Zi = Zi.reshape(np.shape(xx))

    if checks:
        fig = plt.figure( figsize=(14,6))
        plt.pcolormesh(xx,yy, Zi, cmap = 'RdBu_r', vmin = -1.2, vmax = 1.2, shading='gouraud')
        #figfile = path.join(outdir,"interpolated_data.png" )
        plt.gca().invert_yaxis()
        plt.title('Frame ' + str(num), fontsize = 20)
        plt.xlabel('X (m)', fontsize = 20)
        plt.ylabel('Y (m)', fontsize = 20)
        c =plt.colorbar()
        plt.gca().set_aspect('equal')
        c.set_label("Elevations (m)", fontsize=20)
        plt.gca().tick_params(axis='x', labelsize=16)
        plt.gca().tick_params(axis='y', labelsize=16)
        c.ax.tick_params(labelsize=13)

    return xx, yy, Zi
    


def drop_filled_values(nc_file, var_name):
    """
    replaces all the -999999 (alternative for nans) values with 0.
    """    
    thresh = -10 # anthing less than -10m
    import netCDF4
    with netCDF4.Dataset(nc_file, 'r+') as ds:
        Z = ds[var_name][:]
        Z[Z < thresh] = 0
        ds[var_name][:] = Z
        return Z


def detrend2_xy(X, Y, z):
    '''
    detrends the z variable by using pseudoinverse
    X,Y,Z are of the same shape
    '''
    foo = np.isnan(z) == False
    M2 = np.column_stack([np.ones(np.sum(foo)), X[foo], Y[foo], X[foo]*Y[foo]])
    a2 = np.linalg.inv(M2.T @ M2) @ M2.T @ z[foo]
    zt = a2[0] + a2[1]*X + a2[2]*Y + a2[3]*X*Y
    z2 = z - zt
    return z2


def square_crop_fromCentre(Z, desired_size=96):
    
    """
    Crops the WASS data at the centre based on the indices range provided by the user
    NB: the desired size is not in meters.. 
    """
    nt, _, _ = np.shape(Z)
    # Determine the dimensions of the original cube
    original_shape = np.shape(Z) 

    # Determine the desired shape of the cropped cube
    cropped_shape = (nt, desired_size, desired_size)

    # Calculate the starting indices for cropping
    start_indices = tuple(np.subtract(original_shape[1:], cropped_shape[1:]) // 2)

    # Calculate the ending indices for cropping
    end_indices = tuple(np.add(start_indices, cropped_shape[1:]))

    # Crop the data
    cropped_data = Z[:, start_indices[0]:end_indices[0], start_indices[1]:end_indices[1]]
    print('Cropped data shape:',  cropped_data.shape)
   
    return cropped_data

def hann_taper(cube):

    """
    Tapers the cube data
    """
    rows, cols, time = cube.shape
    row_taper = np.hanning(rows).reshape(rows, 1, 1)
    col_taper = np.hanning(cols).reshape(1, cols, 1)
    time_taper = np.hanning(time).reshape(1, 1, time)
    return cube * row_taper * col_taper * time_taper


def zero_pad_cube(cube, new_shape):
    """
    Pads the cube data with zeros to the desired shape
    """
    old_shape = cube.shape
    old_center = np.array(old_shape[:2]) // 2
    new_center = np.array(new_shape[:2]) // 2

    padding = (new_center - old_center)
    start = padding
    end = new_shape[:2] - padding - old_shape[:2]

    new_cube = np.zeros(new_shape)
    new_cube[start[0]:start[0]+old_shape[0], start[1]:start[1]+old_shape[1], :old_shape[2]] = cube

    return new_cube


def wrap_to_pm_pi(angles):
    """
    wrap the angles between plus or minus pi
    """
    foo = np.mod(angles + np.pi, 2 * np.pi) - np.pi
    angles_wrapped = np.where(foo >= np.pi, foo - 2 * np.pi, foo)
    return angles_wrapped


def reduce_maskSize(mask, reduce_by):

    """
    This shrinks the mask of Z by the provided value
    """
    J,I = np.shape(mask)
    drop = reduce_by 
    my_zeros_mask = np.zeros((np.shape(mask)))

    for j in range(J):
        for i in range(I):
            indi = np.argwhere(mask[j,:]>0)
            indi_dropped = indi[drop:-drop]
            my_zeros_mask[j,indi_dropped] = 1

            indj = np.argwhere(mask[:,i]>0)
            indj_dropped = indj[drop:- drop]
            my_zeros_mask[indj_dropped,i] = 1
    return my_zeros_mask    


def taper_maskZ(new_mask, debug = 1):
    '''
    Takes in the reduced shape of the masked data and 
    outputs a tapered mask
    '''
    
    my_zeros_masky = np.zeros((np.shape(new_mask)))
    J,I = np.shape(new_mask)
    
    def my_hann(N):
        return 0.5 - (0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1)))

    for j in range(J):
        #for the x-direction
        indi = np.argwhere(new_mask[j,:]>0)
        Nx = len(indi) #get the length of the ones per row
        fooxt = my_hann(Nx) #tapered for an instance along x
        extracted_values = new_mask[j,indi]
        updated_values = np.squeeze(extracted_values)*fooxt
        my_zeros_masky[j,np.squeeze(indi)] = updated_values
    
    my_zeros_maskx = np.zeros((np.shape(new_mask)))
    for i in range(I):

        #do the same for the y-direction
        indj = np.argwhere(new_mask[:,i]>0)
        Ny = len(indj) #get the length of the ones per column
        fooyt = my_hann(Ny) #tapered for an instance along y
        extracted_values = new_mask[indj,i]
        updated_values = np.squeeze(extracted_values)*fooyt
        my_zeros_maskx[np.squeeze(indj),i] = updated_values 
            
    my_zeros_mask = my_zeros_masky * my_zeros_maskx
    #my_zeros_mask = np.nan_to_num( my_zeros_mask)
    
    if debug: #set as 1 to see result
        plt.figure()
        plt.pcolor(my_zeros_mask, cmap='gray')
        plt.colorbar()
        plt.title('Tapered maskZ')
        plt.gca().invert_yaxis()

    return my_zeros_mask


def spatial_tapering(mask, data):
    """
    This funtion applys the tapered mask on the data through multiplication
    """
    result = np.zeros(data.shape)
    for i in range(data.shape[0]):
        result[i,:, :] = mask * data[i,:, :]
    return result



def temporal_taper(taperedZ):
    """
    This function tapers the provided cube shape data in time
    """
    nt, _, _ = np.shape(taperedZ)
    def hann_taper(N):
        return 0.5 - (0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    Nt = hann_taper(nt) 

    Zt = np.zeros_like(taperedZ)
    for i in range(nt):
        Zt[i, :, :] = Nt[i] * taperedZ[i, :, :] # Multiply the i-th element of the 1D array to the i-th time step of the 3D cube data
    return Zt


def padZ_withZeros(cube, new_shape):

    """
    This function add zeros to the edges of the tappered WASS data
    """
    old_shape = cube.shape
    old_center = np.array(old_shape[:2]) // 2
    new_center = np.array(new_shape[:2]) // 2

    padding = (new_center - old_center)
    start = padding
    end = new_shape[:2] - padding - old_shape[:2]

    new_cube = np.zeros(new_shape)
    new_cube[start[0]:start[0]+old_shape[0], start[1]:start[1]+old_shape[1], :old_shape[2]] = cube

    return new_cube


def ramp(data, ko , check=0):
    
    """
    Takes in the vector created with a contant interval of what should be on the x-axis (e.g wavenumber)
    returns a ramped form of the vector supplied.
    """

    # evaulation window
    foo = 1 - np.tanh(2*data/ko)**8 #the power 8 doesnt matter
    foo2 = 1-foo
    
    foo2[foo2<= ko]=='nan'
    if check: #set to 1
        plt.figure()
        plt.semilogx(data,foo, 'k')
        plt.semilogx(data,foo2, '--', color ='k') #increases with increease along x
        plt.title('Ramp fuction from LR_10', fontsize =20)
        plt.xlabel('data', fontsize =20)
        plt.ylabel('function', fontsize =20)
        plt.legend(['ramp','ramp2'])
    return foo, foo2

def normalized_directionalSPEC(raw_2dspec,Nk):
    '''
    Takes raw 2d spectral and normalises along the wavenumber dim
    '''
    Dm = np.zeros_like(raw_2dspec)
    for i in range(Nk):
        foo = np.nanmax(raw_2dspec[:,i-1])
        Dm[:,i-1] = raw_2dspec[:,i-1]/foo
    return Dm
    

def compute_2dspectral_smallWindow(Z_detrended, Etot, resolution, dirty=1, fine=0, magnitude = 3, fs = 12,dk = 0.05 ):
    '''
    input the cropped data from the WASS nc data. Make sure that the time is in the first col.
    outputs:K,theta, spec2d
    set fine =1 to get the best plotting
    '''

    dummyZ = list()
    nt,nx,nx =np.shape(Z_detrended)
    for idx in np.arange(nt):
        foo = np.array(Z_detrended[idx,:,:])
        dummyZ.append(foo)
    dummyZ = np.asarray(dummyZ) #converts the list to array

    shiftedZ = np.moveaxis(dummyZ, 0,2) #moves the the time col to the last
    taperedZ = hann_taper(shiftedZ)                                       
    paddedZ = zero_pad_cube(taperedZ , (magnitude*nx, magnitude*nx, nt))
    print('finished tapering')

    #compute the loss during tapering of the data
    fac = np.var(shiftedZ)/np.var(taperedZ)
    print('The loss factor is =' , fac)

    dt = 1/fs
    dx =resolution
    dy = dx
    pi = np.pi

    #make the length of the data in all directions to be
    N = np.shape(paddedZ)
    Nx = N[0] - 1 if N[0] % 2 != 0 else N[0]
    Ny = N[1] - 1 if N[1] % 2 != 0 else N[1]
    Nt = N[2] - 1 if N[2] % 2 != 0 else N[2]

    #get the length of the record
    Lx = Nx * dx
    Ly = Ny * dy

    #carry out fourier transform analysis
    AKa = np.fft.fftshift(np.fft.fftn(paddedZ ))

    Kx = np.fft.fftshift(np.fft.fftfreq(Nx)*2*pi/dx) #multiply by 2pi to make it in rad/m
    Ky = np.fft.fftshift(np.fft.fftfreq(Ny)*2*pi/dy) #multiply by 2pi to make it in rad/m
    W = np.fft.fftshift(np.fft.fftfreq(Nt)*2*pi/dt) #multiply by 2pi to make it in rad/s
    print('Finished running the FFT')

    #compute the spectrum unambigiously
    Eoka = np.abs(AKa/Nx/Ny/Nt)**2

    #take care of the tapering 
    Eoka = Eoka *fac

    dw = np.mean(np.diff(W))
    dky=np.mean(np.diff(Kx))
    dkx=np.mean(np.diff(Ky))
    ko =np.sqrt(np.power(dkx,2) + np.power(dky,2))
    
    foo = np.squeeze(np.where(W<0))
    E1 = np.sum(Eoka[...,foo], axis =2) #integrate over the -ve frequencies

    E2 = 2*E1 #take care of the positive freq
  
    fac =(Etot/np.nansum(E2*dky*dkx))
    print('The loss factor is =' , fac)

    #checks variance
    if not round(Etot, 5) == round(np.nansum(E2*dky*dkx) * fac, 5):
        print('Encoutered an error!')

    #recover the loss due to tapering
    E2 = E2 * fac
    print(f' Fourier space = {np.nansum(E2*dky*dkx) }')
    print(f' Physical space = {np.nanvar(Z_detrended)}')

    #turn kx, ky to 2d
    KKx, KKy = np.meshgrid(Kx, Ky)
    KK= np.sqrt(np.power(KKx,2) + np.power(KKy,2))
    kbin = np.arange(np.min(KK),np.max(KK),dk)
    Nk = len(kbin)

    spec_1d = np.zeros_like(kbin)
    for j in np.arange(0,Nk):
        ind = np.argwhere((KK.flatten()>kbin[j-1] - dk/2.0) &  (KK.flatten()<=kbin[j-1] + dk/2.0))
        spec_1d[j-1] = np.sum(E2.flatten()[ind])*dk

    if dirty:
        fig = plt.figure()
        plt.loglog(kbin[2:],spec_1d[2:], "k", linewidth = '1',label='')
        plt.plot(kbin[2:],7e-3*kbin[2:]**(-3), "red", linewidth = '1',label='')
        plt.xlabel('k (rad/m)')
        plt.ylabel('E (m3/rad)')
        plt.xlim([0.001, 15])

    #establish edge bins
    kx_edge = Kx - (dk/2.)
    kx_edge = np.append(kx_edge, max(kx_edge)+ dk)
    ky_edge = kx_edge

    #call binning function
    ret = binned_statistic_2d(KKx.flatten(), KKy.flatten(), E2.flatten(), 'sum', bins=[kx_edge, ky_edge], 
        expand_binnumbers=False)
    spec_2d = ret.statistic
    spec_2d = spec_2d.T #need to transpose to match the meshgrid orientation which is (J,I)

    theta = wrap_to_pm_pi(np.arctan2(KKy,KKx))
    K2 = np.sqrt(np.power(KKx,2) + np.power(KKy,2))
  

    #--------------------------------------normalize and replace zeros with nans
    dk =ko/2
    kbin = np.arange(0,np.max(K2.flatten()),dk)
    Nk = len(kbin)

    #turn all zeroes to nans
    spec2d = np.zeros_like(K2.flatten())
    for i in range(0,Nk):

        ind = np.argwhere((K2.flatten()>kbin[i-1] - dk/2.0) &  (K2.flatten()<=kbin[i-1] + dk/2.0))
        Lmax = np.max(spec_2d.flatten()[ind])
        spec2d[ind] = spec_2d.flatten()[ind]/Lmax

    spec2d = spec2d.reshape(np.shape(spec_2d))
    print('finished all computations')

    #set all zeros to nans
    if dirty:
        spec2d_nanfree = spec2d
        spec2d = spec2d.reshape(np.shape(spec_2d))
        spec2d = spec2d.astype('float')
        spec2d[spec2d  == 0] = 'nan' 

    #------------------------------------vis----------------
    if dirty: #dirty plot
        plt.figure(figsize=(10,8))
        levels = np.linspace(0, 1, 200+1)
        cs = plt.contourf(K2,theta, spec2d, levels = levels, cmap ='jet')
        plt.xlim([4.2,0])
        plt.xlabel('k (rad/m)', fontsize =20)
        plt.ylabel('Theta (rad)', fontsize =20)
        plt.colorbar()
        
    if fine:
        print('now plotting')
        fontsize=38
        levels = np.linspace(0, 1, 100+1)
        print('Making spectral plot')

        plt.figure(figsize=(14,11))
        cs = plt.contourf(K2,theta, spec2d, levels = levels, cmap ='jet')
        plt.ylim([np.pi, -np.pi])
        plt.xlim([4.2,0])
        ax =plt.gca()
        ax.set_xlabel("$k \ (rad/m)$", fontsize = fontsize, weight='bold')
        ax.set_ylabel(r"${\theta}$ (rad)", fontsize = fontsize)
        y_tick = np.linspace(-np.pi,np.pi,5)
        #y_label = [r"$-\pi$",r"$-\frac{3\pi}{4}$", r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$+\frac{\pi}{4}$", r"$+\frac{\pi}{2}$",r'$+\frac{3\pi}{4}$',r"$+\pi$"]
        y_label = [r"$-\pi$", r"$-\frac{\pi}{2}$",  r"$0$",  r"$+\frac{\pi}{2}$",r"$+\pi$"]
        ax.set_yticks(y_tick)
        ax.set_yticklabels(y_label, fontsize=fontsize, weight='bold')
        plt.axhline(np.deg2rad(45./2), color = 'k', linewidth = '4', linestyle ='--')
        plt.legend(['dominant wave/wind dir'],  fontsize=28)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.064)
        c=plt.colorbar(cs, cax= cax, fraction=0.246)
        c.set_label(r'$D(k,{\theta})$', rotation=90, fontsize=50, weight='bold')
        c.ax.get_yaxis().labelpad = 5
        c.ax.tick_params(labelsize=fontsize)
        c.set_ticks(list(np.arange(0,11,1)))
        ax.tick_params(axis='x', labelsize=fontsize, pad=15)
        plt.show()
        
    return K2,theta, spec2d, spec_2d,ko 
