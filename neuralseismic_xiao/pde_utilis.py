import torch

import torch.nn.functional as F

def equation_fd(inn_var, out_var, m_train, m0_train, u0_real_train, u0_imag_train, f):
    
    # inn_var: Spatial grid spacing  
    # out_var: The perturbed part of the wavefield, referred to as "scattered wavefield" in the paper
    # m_train: Updated medium parameter model, which is the square of the slowness
    # m0_train: Background medium model
    # u0_real_train, u0_imag_train: Real and imaginary parts of the background wavefield
    # f: Frequency
    # omega: 2 * pi * f
    # d2udx2 and d2udy2: Second-order spatial derivatives of the wavefield in the x and y directions
    
    omega = (2.0 * f * torch.pi)[...,1:-1,1:-1]
    d2udx2 = torch.zeros(out_var.shape).to(out_var.device)
    d2udy2 = torch.zeros(out_var.shape).to(out_var.device)
    d2udx2[:,0:1,1:-1,1:-1] = (out_var[:,0:1,0:-2, 1:-1] + out_var[:,0:1,2:, 1:-1] - 2*out_var[:,0:1,1:-1, 1:-1]) / (inn_var)**2
    d2udx2[:,1:2,1:-1,1:-1] = (out_var[:,1:2,0:-2, 1:-1] + out_var[:,1:2,2:, 1:-1] - 2*out_var[:,1:2,1:-1, 1:-1]) / (inn_var)**2
    d2udy2[:,0:1,1:-1,1:-1] = (out_var[:,0:1,1:-1, 0:-2] + out_var[:,0:1,1:-1, 2:] - 2*out_var[:,0:1,1:-1, 1:-1]) / (inn_var)**2
    d2udy2[:,1:2,1:-1,1:-1] = (out_var[:,1:2,1:-1, 0:-2] + out_var[:,1:2,1:-1, 2:] - 2*out_var[:,1:2,1:-1, 1:-1]) / (inn_var)**2
    res_x = omega**2*(m_train[:,:,1:-1,1:-1]) * out_var[:,0:1,1:-1,1:-1] + d2udx2[:,0:1,1:-1,1:-1] + d2udy2[:,0:1,1:-1,1:-1] + omega**2 * (m_train[:,:,1:-1,1:-1]- m0_train[:,:,1:-1, 1:-1])* u0_real_train[:,:,1:-1,1:-1] #equation 4 in the paper
    res_y = omega**2*(m_train[:,:,1:-1,1:-1]) * out_var[:,1:2,1:-1,1:-1] + d2udx2[:,1:2,1:-1,1:-1] + d2udy2[:,1:2,1:-1,1:-1] + omega**2 * (m_train[:,:,1:-1,1:-1]- m0_train[:,:,1:-1,1:-1])* u0_imag_train[:,:,1:-1,1:-1] 
    # return torch.cat((res_x, res_y), dim=1), d2udx2, d2udy2
    return torch.cat((res_x, res_y), dim=1)

def equation_fd_4th(inn_var, out_var, m_train, m0_train, u0_real_train, u0_imag_train, f):
    # 4th-order finite difference stencil coefficients
    coeffs = [-1/12, 4/3, -5/2, 4/3, -1/12]

    omega = (2.0 * f * torch.pi)[..., 2:-2, 2:-2]
    d2udx2 = torch.zeros(out_var.shape).to(out_var.device)
    d2udy2 = torch.zeros(out_var.shape).to(out_var.device)

    # 4th-order finite difference in the x-direction
    d2udx2[:, 0:1, 2:-2, 2:-2] = (
        coeffs[0] * out_var[:, 0:1, 0:-4, 2:-2] +
        coeffs[1] * out_var[:, 0:1, 1:-3, 2:-2] +
        coeffs[2] * out_var[:, 0:1, 2:-2, 2:-2] +
        coeffs[3] * out_var[:, 0:1, 3:-1, 2:-2] +
        coeffs[4] * out_var[:, 0:1, 4:, 2:-2]
    ) / (inn_var)**2

    d2udx2[:, 1:2, 2:-2, 2:-2] = (
        coeffs[0] * out_var[:, 1:2, 0:-4, 2:-2] +
        coeffs[1] * out_var[:, 1:2, 1:-3, 2:-2] +
        coeffs[2] * out_var[:, 1:2, 2:-2, 2:-2] +
        coeffs[3] * out_var[:, 1:2, 3:-1, 2:-2] +
        coeffs[4] * out_var[:, 1:2, 4:, 2:-2]
    ) / (inn_var)**2

    # 4th-order finite difference in the y-direction
    d2udy2[:, 0:1, 2:-2, 2:-2] = (
        coeffs[0] * out_var[:, 0:1, 2:-2, 0:-4] +
        coeffs[1] * out_var[:, 0:1, 2:-2, 1:-3] +
        coeffs[2] * out_var[:, 0:1, 2:-2, 2:-2] +
        coeffs[3] * out_var[:, 0:1, 2:-2, 3:-1] +
        coeffs[4] * out_var[:, 0:1, 2:-2, 4:]
    ) / (inn_var)**2

    d2udy2[:, 1:2, 2:-2, 2:-2] = (
        coeffs[0] * out_var[:, 1:2, 2:-2, 0:-4] +
        coeffs[1] * out_var[:, 1:2, 2:-2, 1:-3] +
        coeffs[2] * out_var[:, 1:2, 2:-2, 2:-2] +
        coeffs[3] * out_var[:, 1:2, 2:-2, 3:-1] +
        coeffs[4] * out_var[:, 1:2, 2:-2, 4:]
    ) / (inn_var)**2

    # Residual computations
    res_x = omega**2 * (m_train[:, :, 2:-2, 2:-2]) * out_var[:, 0:1, 2:-2, 2:-2] + \
            d2udx2[:, 0:1, 2:-2, 2:-2] + d2udy2[:, 0:1, 2:-2, 2:-2] + \
            omega**2 * (m_train[:, :, 2:-2, 2:-2] - m0_train[:, :, 2:-2, 2:-2]) * u0_real_train[:, :, 2:-2, 2:-2]

    res_y = omega**2 * (m_train[:, :, 2:-2, 2:-2]) * out_var[:, 1:2, 2:-2, 2:-2] + \
            d2udx2[:, 1:2, 2:-2, 2:-2] + d2udy2[:, 1:2, 2:-2, 2:-2] + \
            omega**2 * (m_train[:, :, 2:-2, 2:-2] - m0_train[:, :, 2:-2, 2:-2]) * u0_imag_train[:, :, 2:-2, 2:-2]

    # Return the results
    return torch.cat((res_x, res_y), dim=1)

def equation_fd_loss(inn_var, out_var, m_train, m0_train, u0_real_train, u0_imag_train, f):
    
    # inn_var: Spatial grid spacing  
    # out_var: The perturbed part of the wavefield, referred to as "scattered wavefield" in the paper
    # m_train: Updated medium parameter model, which is the square of the slowness
    # m0_train: Background medium model
    # u0_real_train, u0_imag_train: Real and imaginary parts of the background wavefield
    # f: Frequency
    # omega: 2 * pi * f
    # d2udx2 and d2udy2: Second-order spatial derivatives of the wavefield in the x and y directions
    
    omega = (2.0 * f * torch.pi)[...,1:-1,1:-1]
    d2udx2 = torch.zeros(out_var.shape).to(out_var.device)
    d2udy2 = torch.zeros(out_var.shape).to(out_var.device)
    d2udx2[:,0:1,1:-1,1:-1] = (out_var[:,0:1,0:-2, 1:-1] + out_var[:,0:1,2:, 1:-1] - 2*out_var[:,0:1,1:-1, 1:-1]) / (inn_var)**2
    d2udx2[:,1:2,1:-1,1:-1] = (out_var[:,1:2,0:-2, 1:-1] + out_var[:,1:2,2:, 1:-1] - 2*out_var[:,1:2,1:-1, 1:-1]) / (inn_var)**2
    d2udy2[:,0:1,1:-1,1:-1] = (out_var[:,0:1,1:-1, 0:-2] + out_var[:,0:1,1:-1, 2:] - 2*out_var[:,0:1,1:-1, 1:-1]) / (inn_var)**2
    d2udy2[:,1:2,1:-1,1:-1] = (out_var[:,1:2,1:-1, 0:-2] + out_var[:,1:2,1:-1, 2:] - 2*out_var[:,1:2,1:-1, 1:-1]) / (inn_var)**2
    res_x = omega**2*(m_train[:,:,1:-1,1:-1]) * out_var[:,0:1,1:-1,1:-1] + d2udx2[:,0:1,1:-1,1:-1] + d2udy2[:,0:1,1:-1,1:-1] + omega**2 * (m_train[:,:,1:-1,1:-1]- m0_train[:,:,1:-1, 1:-1])* u0_real_train[:,:,1:-1,1:-1] #equation 4 in the paper
    res_y = omega**2*(m_train[:,:,1:-1,1:-1]) * out_var[:,1:2,1:-1,1:-1] + d2udx2[:,1:2,1:-1,1:-1] + d2udy2[:,1:2,1:-1,1:-1] + omega**2 * (m_train[:,:,1:-1,1:-1]- m0_train[:,:,1:-1,1:-1])* u0_imag_train[:,:,1:-1,1:-1] 
    # return torch.cat((res_x, res_y), dim=1), d2udx2, d2udy2
    aa=torch.cat((res_x, res_y), dim=1)*1e-2
    f = torch.zeros(aa.shape, device=aa.device)
    loss_f = F.mse_loss(aa, f)
    return loss_f


def equation_fd_8th(inn_var, out_var, m_train, m0_train, u0_real_train, u0_imag_train, f):
    # 8th-order finite difference stencil coefficients
    coeffs = [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]

    omega = (2.0 * f * torch.pi)[..., 4:-4, 4:-4]
    d2udx2 = torch.zeros(out_var.shape).to(out_var.device)
    d2udy2 = torch.zeros(out_var.shape).to(out_var.device)

    # 8th-order finite difference in the x-direction
    d2udx2[:, 0:1, 4:-4, 4:-4] = (
        coeffs[0] * out_var[:, 0:1, 0:-8, 4:-4] +
        coeffs[1] * out_var[:, 0:1, 1:-7, 4:-4] +
        coeffs[2] * out_var[:, 0:1, 2:-6, 4:-4] +
        coeffs[3] * out_var[:, 0:1, 3:-5, 4:-4] +
        coeffs[4] * out_var[:, 0:1, 4:-4, 4:-4] +
        coeffs[5] * out_var[:, 0:1, 5:-3, 4:-4] +
        coeffs[6] * out_var[:, 0:1, 6:-2, 4:-4] +
        coeffs[7] * out_var[:, 0:1, 7:-1, 4:-4] +
        coeffs[8] * out_var[:, 0:1, 8:, 4:-4]
    ) / (inn_var)**2

    d2udx2[:, 1:2, 4:-4, 4:-4] = (
        coeffs[0] * out_var[:, 1:2, 0:-8, 4:-4] +
        coeffs[1] * out_var[:, 1:2, 1:-7, 4:-4] +
        coeffs[2] * out_var[:, 1:2, 2:-6, 4:-4] +
        coeffs[3] * out_var[:, 1:2, 3:-5, 4:-4] +
        coeffs[4] * out_var[:, 1:2, 4:-4, 4:-4] +
        coeffs[5] * out_var[:, 1:2, 5:-3, 4:-4] +
        coeffs[6] * out_var[:, 1:2, 6:-2, 4:-4] +
        coeffs[7] * out_var[:, 1:2, 7:-1, 4:-4] +
        coeffs[8] * out_var[:, 1:2, 8:, 4:-4]
    ) / (inn_var)**2

    # 8th-order finite difference in the y-direction
    d2udy2[:, 0:1, 4:-4, 4:-4] = (
        coeffs[0] * out_var[:, 0:1, 4:-4, 0:-8] +
        coeffs[1] * out_var[:, 0:1, 4:-4, 1:-7] +
        coeffs[2] * out_var[:, 0:1, 4:-4, 2:-6] +
        coeffs[3] * out_var[:, 0:1, 4:-4, 3:-5] +
        coeffs[4] * out_var[:, 0:1, 4:-4, 4:-4] +
        coeffs[5] * out_var[:, 0:1, 4:-4, 5:-3] +
        coeffs[6] * out_var[:, 0:1, 4:-4, 6:-2] +
        coeffs[7] * out_var[:, 0:1, 4:-4, 7:-1] +
        coeffs[8] * out_var[:, 0:1, 4:-4, 8:]
    ) / (inn_var)**2

    d2udy2[:, 1:2, 4:-4, 4:-4] = (
        coeffs[0] * out_var[:, 1:2, 4:-4, 0:-8] +
        coeffs[1] * out_var[:, 1:2, 4:-4, 1:-7] +
        coeffs[2] * out_var[:, 1:2, 4:-4, 2:-6] +
        coeffs[3] * out_var[:, 1:2, 4:-4, 3:-5] +
        coeffs[4] * out_var[:, 1:2, 4:-4, 4:-4] +
        coeffs[5] * out_var[:, 1:2, 4:-4, 5:-3] +
        coeffs[6] * out_var[:, 1:2, 4:-4, 6:-2] +
        coeffs[7] * out_var[:, 1:2, 4:-4, 7:-1] +
        coeffs[8] * out_var[:, 1:2, 4:-4, 8:]
    ) / (inn_var)**2

    # Residual computations
    res_x = omega**2 * (m_train[:, :, 4:-4, 4:-4]) * out_var[:, 0:1, 4:-4, 4:-4] + \
            d2udx2[:, 0:1, 4:-4, 4:-4] + d2udy2[:, 0:1, 4:-4, 4:-4] + \
            omega**2 * (m_train[:, :, 4:-4, 4:-4] - m0_train[:, :, 4:-4, 4:-4]) * u0_real_train[:, :, 4:-4, 4:-4]

    res_y = omega**2 * (m_train[:, :, 4:-4, 4:-4]) * out_var[:, 1:2, 4:-4, 4:-4] + \
            d2udx2[:, 1:2, 4:-4, 4:-4] + d2udy2[:, 1:2, 4:-4, 4:-4] + \
            omega**2 * (m_train[:, :, 4:-4, 4:-4] - m0_train[:, :, 4:-4, 4:-4]) * u0_imag_train[:, :, 4:-4, 4:-4]

    # Return the results
    return torch.cat((res_x, res_y), dim=1)


def vel_fd(inn_var, out_var, m0_train, u0_real_train, u0_imag_train, f):
    omega = (2.0 * f * torch.pi)[...,1:-1,1:-1]
    d2udx2 = torch.zeros(out_var.shape).to(out_var.device)
    d2udy2 = torch.zeros(out_var.shape).to(out_var.device)
    d2udx2[:,0:1,1:-1,1:-1] = (out_var[:,0:1,0:-2, 1:-1] + out_var[:,0:1,2:, 1:-1] - 2*out_var[:,0:1,1:-1, 1:-1]) / (inn_var)**2
    d2udx2[:,1:2,1:-1,1:-1] = (out_var[:,1:2,0:-2, 1:-1] + out_var[:,1:2,2:, 1:-1] - 2*out_var[:,1:2,1:-1, 1:-1]) / (inn_var)**2
    d2udy2[:,0:1,1:-1,1:-1] = (out_var[:,0:1,1:-1, 0:-2] + out_var[:,0:1,1:-1, 2:] - 2*out_var[:,0:1,1:-1, 1:-1]) / (inn_var)**2
    d2udy2[:,1:2,1:-1,1:-1] = (out_var[:,1:2,1:-1, 0:-2] + out_var[:,1:2,1:-1, 2:] - 2*out_var[:,1:2,1:-1, 1:-1]) / (inn_var)**2
    print(out_var.shape)
    print((d2udx2[:,0:1,1:-1,1:-1] + d2udy2[:,0:1,1:-1,1:-1] - omega ** 2 * m0_train[...,1:-1,1:-1] * u0_real_train[...,1:-1,1:-1]).shape)
    print((omega**2*u0_real_train[...,1:-1,1:-1]+ omega**2*out_var[:0:1,1:-1,1:-1]))
    m = (d2udx2[:,0:1,1:-1,1:-1] + d2udy2[:,0:1,1:-1,1:-1] - omega ** 2 * m0_train[...,1:-1,1:-1] * u0_real_train[...,1:-1,1:-1])/(omega**2*u0_real_train[...,1:-1,1:-1]+ omega**2*out_var[:,0:1,1:-1,1:-1])
    return m

def laplacian_wavenumber(out_var, in_var=0.025):
    """
    Compute the Laplacian of input data using Fourier wavenumber method.
    Args:
        out_var (tensor): Input tensor of shape (b, c, w, h)
        in_var (float): Grid spacing
    Returns:
        wlap (tensor): Laplacian of the input tensor
    """
    batchsize, channel, nx, ny = out_var.size()
    device = out_var.device

    # Perform 2D FFT
    w_h = torch.fft.fftn(out_var, dim=(-2, -1))  # FFT on the last two dimensions

    # Define wavenumbers (Fourier space grid)
    kx = torch.fft.fftfreq(nx, d=in_var, device=device).reshape(1, 1, nx, 1)  # Shape (1, 1, nx, 1)
    ky = torch.fft.fftfreq(ny, d=in_var, device=device).reshape(1, 1, 1, ny)  # Shape (1, 1, 1, ny)
    kx = kx * 2 * torch.pi  # 转换为弧度
    ky = ky * 2 * torch.pi
    # Construct symmetric wavenumber grid
    lap = kx ** 2 + ky ** 2  # Wavenumber squared grid
    lap[..., 0, 0] = 1.0  # Avoid division by zero at the DC component

    # Apply negative Laplacian in Fourier space
    wlap_h = -lap * w_h
    wlap = torch.fft.ifftn(wlap_h, dim=(-2, -1)).real  # Inverse FFT and take real part

    return wlap


def equation_fft(inn_var, out_var, m_train, m0_train, u0_real_train, u0_imag_train, f):
    omega = (2.0 * f * torch.pi)[..., 1:-1, 1:-1]
    
    # Compute Laplacian in frequency domain
    laplace_out = laplacian_wavenumber(out_var, inn_var)

    # Residual computation for real and imaginary parts
    res_x = (
        omega**2 * m_train[:, :, 1:-1, 1:-1] * out_var[:, 0:1, 1:-1, 1:-1]
        + laplace_out[:, 0:1, 1:-1, 1:-1]
        + omega**2 * (m_train[:, :, 1:-1, 1:-1] - m0_train[:, :, 1:-1, 1:-1]) * u0_real_train[:, :, 1:-1, 1:-1]
    )
    res_y = (
        omega**2 * m_train[:, :, 1:-1, 1:-1] * out_var[:, 1:2, 1:-1, 1:-1]
        + laplace_out[:, 1:2, 1:-1, 1:-1]
        + omega**2 * (m_train[:, :, 1:-1, 1:-1] - m0_train[:, :, 1:-1, 1:-1]) * u0_imag_train[:, :, 1:-1, 1:-1]
    )

    # Combine residuals for x and y components
    return torch.cat((res_x, res_y), dim=1)

def equation_fft_pad(inn_var, out_var, m_train, m0_train, u0_real_train, u0_imag_train, f):
    omega = (2.0 * f * torch.pi)
    
    # Compute Laplacian in frequency domain
    laplace_out = laplacian_wavenumber(out_var, inn_var)

    # Residual computation for real and imaginary parts
    res_x = (
        omega**2 * m_train * out_var[:, 0:1,...]
        + laplace_out[:, 0:1, ...]
        + omega**2 * (m_train - m0_train) * u0_real_train
    )
    res_y = (
        omega**2 * m_train * out_var[:, 1:2,...]
        + laplace_out[:, 1:2, ...]
        + omega**2 * (m_train[:, :, ...] - m0_train[:, :, ...]) * u0_imag_train[:, :, ...]
    )
    #print(res_x.shape)
    # Combine residuals for x and y components
    return torch.cat((res_x, res_y), dim=1)

def equation_fd_conv(inn_var, out_var, m_train, m0_train, u0_real_train, u0_imag_train, f):
    omega = (2.0 * f * torch.pi)
    
    
    
    # Define 4th-order finite difference kernels for second-order derivatives
    kernel = torch.tensor([[[[0, 0, -1/12, 0, 0],
                               [0, 0, 16/12, 0, 0],
                               [-1/12, 16/12, -30*2/12, 16/12, -1/12],
                               [0, 0, 16/12, 0, 0],
                               [0, 0, -1/12, 0, 0]]]], dtype=torch.float32, device=out_var.device) 

    # Debugging: Print shapes of input tensors
    # print("inn_var:", inn_var)
    # print("out_var shape:", out_var.shape)
    # print("m_train shape:", m_train.shape)
    # print("m0_train shape:", m0_train.shape)
    # print("u0_real_train shape:", u0_real_train.shape)
    # print("u0_imag_train shape:", u0_imag_train.shape)
    
    # Separate real and imaginary parts of out_var
    out_var_real = out_var[:, 0:1, :, :]
    out_var_imag = out_var[:, 1:2, :, :]
    
    # Apply convolution to compute second-order derivatives for real part
    du_real_laplace = 1/(inn_var ** 2) * F.conv2d(out_var_real, kernel, padding=2, groups=1)
    
    du_imag_laplace = 1/(inn_var ** 2) * F.conv2d(out_var_imag, kernel, padding=2, groups=1)
     # Debugging: Print shapes of convolution results
    # print("du_real_laplace shape:", du_real_laplace.shape)
    # print("du_imag_laplace shape:", du_imag_laplace.shape)
       
    loss_real = omega*omega*m_train*out_var_real + du_real_laplace + omega*omega*(m_train-m0_train)* u0_real_train

    
    loss_imag = omega*omega*m_train*out_var_imag + du_imag_laplace + omega*omega*(m_train-m0_train)* u0_imag_train
    # Debugging: Print shapes of loss tensors
    # print("loss_real shape:", loss_real.shape)
    # print("loss_imag shape:", loss_imag.shape)



    pde_loss = torch.sqrt((torch.pow(loss_real,2)).mean() + (torch.pow(loss_imag,2)).mean())
    
    return pde_loss.item()
    
