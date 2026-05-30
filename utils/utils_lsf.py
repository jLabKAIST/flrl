import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision.transforms.v2 import Lambda

def construct_fourier_coeff(Nx, Ny, cols, symmetry = False):
    """
        Build a Fourier coefficient matrix based on the agent's policy sampling.
            
    """
    
    
    def preprocess_imag_general(mat, dim, num_col, num_row):
        center_idx = len(mat)//2

        #make the conjugate: 5 1 2 3 4 --> 5 1 2 -1 -5
        mat[center_idx+1:] = np.flip(mat[:center_idx]) * -1

        if dim == 1:
            mat[center_idx] = 0
            return mat
        elif dim == 2:
            #1D vector to 2D matrix
            mat = np.reshape(mat, (-1, num_col))
            
            #make the conjugate on the right side of the matrix
            mat[:num_row//2, num_col//2+1:] = np.flip(mat[:num_row//2, :num_col//2]*-1, axis = 1)
            
            #make the matrix symmetrical around x-axis (to get an x-axis symmetrical device)
            mat[num_row//2+1:,] = np.flip(mat[:num_row//2,], axis = 0)
            
            #remove the imaginary part in the central y-axis
            mat[:, num_col//2] = 0 
            return mat

    def preprocess_real_general(mat, dim, num_col, num_row):
        center_idx = len(mat)//2

        #for center
        mat[center_idx+1:] = np.flip(mat[:center_idx])
        
        if dim == 1:
            return mat
        elif dim == 2:
            #1D vector to 2D matrix
            mat = np.reshape(mat, (-1, num_col))
            
            #ensure the conjugate
            mat[:num_row//2, num_col//2+1:] = np.flip(mat[:num_row//2, :num_col//2], axis = 1)
            
            #make the device symmetrical
            mat[num_row//2+1:] = np.flip(mat[:num_row//2], axis = 0)
            
            return mat
    
    def preprocess_imag_symmetry(mat):
        center_col = mat.shape[1]//2
        center_row = mat.shape[0]//2

        #make the center column real
        mat[:, center_col:] = 0

        #make the top and bottom to be symmetrical
        mat[center_row + 1:, :] = np.flip(mat[:center_row, :], axis = 0)

        #make the conjugate
        mat = mat + np.flip(mat, axis = 1) * -1

        return mat

    def preprocess_real_symmetry(mat):
        center_col = mat.shape[1]//2
        center_row = mat.shape[0]//2

        #make the right part zero
        mat[:, center_col + 1:] = 0

        #make the top and bottom to be symmetrical
        mat[center_row + 1:, :] = np.flip(mat[:center_row, :], axis = 0)

        mat = mat + np.flip(mat) 

        mat[:, center_col] = mat[:, center_col]/2

        return mat

    def preprocess_imag(mat):
        center_col = mat.shape[1]//2
        center_row = mat.shape[0]//2

        #make the center column real
        mat[:, center_col:] = 0

        #make the conjugate
        mat = mat + np.flip(np.flip(mat, axis = 0), axis = 1) * -1

        return mat
    
    def preprocess_real(mat):
        center_col = mat.shape[1]//2
        center_row = mat.shape[0]//2

        #make the right part zero
        mat[:, center_col + 1:] = 0

        mat = mat + np.flip(mat) 

        mat[:, center_col] = mat[:, center_col]/2

        return mat

    num_col = 2*Nx+1
    num_row = 1
    
    if Ny == 0:
        dim = 1
    elif Ny > 0:
        dim = 2
        num_row = 2*Ny+1
    
    size = num_row * num_col

    if cols == "random_uniform":
        r = np.round(np.random.uniform(low = -1, high = 1, size = (size)), 2)
        im =  np.round(np.random.uniform(low = -1, high = 1, size = (size)), 2)
    elif cols == "random_gaussian":
        r = np.round(np.tanh(np.random.randn(size)), 2)
        im =  np.round(np.tanh(np.random.randn(size)), 2)
    elif cols == 'zeros':
        #To get all air structure
        r = np.zeros(shape = (size))
        im =  np.zeros(shape = (size))

    # if symmetry:
    #     r = preprocess_real_symmetry(r)
    #     im = preprocess_imag_symmetry(im)
    # elif not symmetry:
    #     r = preprocess_real(r)
    #     im = preprocess_imag(im)
    
    r = preprocess_real_general(r, dim = dim, num_col=num_col, num_row=num_row)
    im = preprocess_imag_general(im, dim = dim, num_col=num_col, num_row=num_row)
        
    z = r + im * 1j
    
    return z

def flatten_fourier_coeff(Nx, Ny, coeff):
    """
        Flatten fourier coefficient matrix to a 1-d vector
        
    """
    if Ny == 0:
        submatrix = coeff[:(Nx+1)]
    else:
        #slice the coefficient matrix
        submatrix = coeff[:(Ny+1), :(Nx+1)]
        
    real_vector = np.real(submatrix).transpose().flatten()
    imag_vector = np.imag(submatrix).transpose().flatten()
    
    #cut the zero element of imag_vector
    #note that the length of real vector should be bigger than then imag_vector due to the real number constraint on the principal y-axis
    cutting_size = Ny + 1
    imag_vector = imag_vector[:len(imag_vector) - cutting_size]
    
    # assert len(real_vector) == (Nx + 1) * (Ny + 1)
    # assert len(imag_vector) == (Nx)     * (Ny + 1)
    
    return real_vector, imag_vector

def inverse_flatten_fourier_coeff(Nx, Ny, real_vector, imag_vector):
    """
        Construct back the flat vector to 2d coefficient matrix
    """
    #add zero elements to the end of the array
    imag_vector = np.pad(imag_vector, pad_width = (0, len(real_vector) - len(imag_vector)), constant_values = 0)
    
    z = real_vector + 1j*imag_vector
    
    matrix = np.reshape(z, (Nx+1, Ny+1)).transpose()
    
    #Flip the matrix around y axis (left to right)
    tmp = np.flip(matrix[:, :matrix.shape[1]-1], axis = 1)
    tmp = np.real(tmp) - 1j * np.imag(tmp)
    matrix = np.concatenate((matrix, tmp), axis = 1)
    
    #flip the matrix around x axis (upper to bottom)
    tmp = np.flip(matrix[:matrix.shape[0]-1, :], axis = 0)
    matrix = np.concatenate((matrix, tmp), axis = 0)
    
    if Ny == 0:
        return matrix.squeeze(axis = 0)
    else:
        return matrix

def inverse_flatten_fourier_coeff_tensor(Nx, Ny, vector):
    """
        Construct back the flat vector to 2d coefficient matrix (with tensor oepration)
    """    
    matrix = th.reshape(vector, (Nx+1, Ny+1))
    matrix = th.transpose(matrix, 0, 1)
    
    #Flip the matrix around y axis (left to right)
    tmp = th.flip(matrix[:, :matrix.shape[1]-1], dims = [1])
    tmp = th.real(tmp) - 1j * th.imag(tmp)
    matrix = th.cat((matrix, tmp), axis = 1)
    
    #flip the matrix around x axis (upper to bottom)
    tmp = th.flip(matrix[:matrix.shape[0]-1, :], dims = [0])
    matrix = th.cat((matrix, tmp), axis = 0)

    return matrix

def generate_lsf(Nx, Ny, period_x, period_y, grid_x, grid_y, coeff):
    if Ny == 0:
        return generate_lsf_1d(Nx, Ny, period_x, period_y, grid_x, grid_y, coeff)
    else:
        return generate_lsf_2d_tensor(Nx, Ny, period_x, period_y, grid_x, grid_y, coeff)

def generate_lsf_2d(Nx, Ny, period_x, period_y, grid_x, grid_y, coeff):
    a_x = -period_x/2
    b_x = period_x/2

    a_y = -period_y/2
    b_y = period_y/2
    
    x = np.linspace(start = a_x, stop = b_x, num = grid_x)
    y = np.linspace(start = a_y, stop = b_y, num = grid_y)
    kx = np.arange(start = -Nx, stop = Nx+1, step = 1)
    ky = np.arange(start = -Ny, stop = Ny+1, step = 1)

    theta_x = lambda x: 2*np.pi * (x - a_x) / (b_x-a_x)
    theta_y = lambda y: 2*np.pi * (y - a_y) / (b_y-a_y)
    
    psi = []
    
    for yy in y:
        #evaluate on each x of the metasurface grid point
        for xx in x:
            #evaluate on each y of the metasurface grid point

            #tmp is the summation value holder
            tmp = 0

            for ky_idx in range(len(ky)):
                #iterate for all kx
                val1 = ky[ky_idx] * theta_y(yy)
                
                for kx_idx in range(len(kx)):
                    #iterate for all ky
                    val2 = kx[kx_idx] * theta_x(xx)

                    #calculate the exp term
                    exp = np.exp(np.complex128(1*1j) * (val1 + val2))
                    
                    #sum all over ks
                    tmp += coeff[ky_idx, kx_idx] * exp

            psi.append(tmp)
            
    psi = np.reshape(psi, (len(y), len(x)))
    
    #ensure that the lsf is real valued
    assert np.sum(np.imag(np.round(psi))) == 0
    
    return psi

def generate_lsf_1d(Nx, Ny, period_x, period_y, grid_x, grid_y, coeff):
    a_x = -period_x/2
    b_x = period_x/2
    
    x = np.linspace(start = a_x, stop = b_x, num = grid_x)
    kx = np.arange(start = -Nx, stop = Nx+1, step = 1)

    theta_x = lambda x: 2*np.pi * (x - a_x) / (b_x-a_x)
    
    psi = []
    
    for xx in x:
        #evaluate on each y of the metasurface grid point

        #tmp is the summation value holder
        tmp = 0
            
        for kx_idx in range(len(kx)):
            #iterate for all ky
            val2 = kx[kx_idx] * theta_x(xx)

            #calculate the exp term
            exp = np.exp(np.complex128(1*1j) * (val2))
            
            #sum all over ks
            tmp += coeff[kx_idx] * exp

        psi.append(tmp)
            
    psi = np.reshape(psi, (len(x)))
    
    #ensure that the lsf is real valued
    assert np.sum(np.imag(np.round(psi))) == 0
    
    return psi

def spectral_grid_initialization(Nx, Ny, grid_x, grid_y, period_x, period_y):
    """
        Helper function to generate a grid of exp terms.
        Each element is repeated around itself for convolution operation
    """
    
    #Initialization
    a_x = -period_x/2
    b_x = period_x/2

    a_y = -period_y/2
    b_y = period_y/2

    Lx = b_x - a_x
    Ly = b_y - a_y

    #Define (x, y) grid
    #Repeated from (x1, x2, x3, ...) to (x1, x1, ..., x2, x2, ...) as many as (2*Nx+1) or (2*Ny+1) copies
    #Repetition is done to perform convolution with (2*Nx+1, 2*Ny+1) Fourier coefficients grid
    x = th.linspace(a_x, b_x, grid_x)
    x = x.repeat_interleave((2*Nx+1), dim = 0)
    y = th.linspace(a_y, b_y, grid_y)
    y = y.repeat_interleave((2*Ny+1), dim = 0)

    #Define (kx, ky) spectral grid
    nx = (2*Nx+1)//2
    ny = (2*Ny+1)//2
    kx = th.linspace(-nx, nx, (2*Nx+1))
    kx = kx.repeat(grid_x)
    ky = th.linspace(ny, -ny, (2*Ny+1))
    ky = ky.repeat(grid_y)

    x_grid, y_grid = th.meshgrid(x, y, indexing='xy')
    kx_grid, ky_grid = th.meshgrid(kx, ky, indexing='xy')

    x_coordinate_trfm = Lambda(lambda x: th.exp(1j * 2*np.pi*(x - a_x)/Lx))
    y_coordinate_trfm = Lambda(lambda y: th.exp(1j * 2*np.pi*(y - a_y)/Ly))

    theta_x_grid = x_coordinate_trfm(x_grid)
    theta_y_grid = y_coordinate_trfm(y_grid)

    return theta_x_grid, theta_y_grid, kx_grid, ky_grid


def lsf_to_topology(lsf, function, beta = 1, threshold_point = 0):
    if function == "tanh":
        tanh = lambda x, beta: (th.exp(beta*(x-threshold_point)) - th.exp(-beta*(x-threshold_point)))/(th.exp(beta*(x-threshold_point)) + th.exp(-beta*(x-threshold_point)))
        z = tanh(lsf, beta)
    elif function == "sigmoid":
        sigmoid = lambda x, beta: 1/(1+th.exp(-beta*(x-threshold_point)))
        z = sigmoid(lsf, beta)
    
    return z

def to_pairs(real, imag):
    imag = np.pad(imag, (0, len(real) - len(imag)), mode='constant')
    
    real = np.reshape(real, (-1, 1))
    imag = np.reshape(imag, (-1, 1))
    
    return np.concatenate([real, imag], axis = 1)

def generate_lsf_2d_tensor(Nx, Ny, period_x, period_y, grid_x, grid_y, coeff, device = 'cpu'):
    """
        Level set function construction with tensor operation (convolution)
    """
    theta_x_grid, theta_y_grid, kx_grid, ky_grid = spectral_grid_initialization(Nx, Ny, grid_x, grid_y, period_x, period_y)
    
    exp_grid = (theta_x_grid ** kx_grid) * (theta_y_grid ** ky_grid)

    exp_grid = exp_grid.reshape(1,1,*exp_grid.shape).to(device)
    coeff = coeff.reshape(1,1,*coeff.shape).to(device)
    exp_grid = exp_grid.type(th.cdouble)

    lsf = F.conv2d(exp_grid, coeff, bias = None, stride = (2*Ny+1, 2*Nx+1), padding = 'valid')

    #ensure that the lsf is real valued
    assert th.sum(th.round(th.imag(lsf))).item() == 0

    return np.real(lsf.squeeze(dim = 0))
