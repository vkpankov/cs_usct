import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interpn


# Angular Spectrum Method Downward into the Medium from the Transmitters
def downward_continuation(x, z, c_x_z, f, tx_x_f, bckgnd_wf_x_z_f, aawin):

    # Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
    ft = lambda sig: np.fft.fftshift(np.fft.fft(aawin * sig, axis=0), axes=0)
    ift = lambda sig: aawin * np.fft.ifft(np.fft.ifftshift(sig, axes=0), axis=0)

    # Spatial Grid
    dx = np.mean(np.diff(x))
    nx = x.size
    x = dx * np.arange(-(nx - 1) / 2, (nx - 1) / 2 + 1)
    dz = np.mean(np.diff(z))

    # FFT Axis for Lateral Spatial Frequency
    kx = np.mod(
        np.fft.fftshift(np.arange(nx) / (dx * nx)) + 1 / (2 * dx), 1 / dx
    ) - 1 / (2 * dx)

    # Convert to Slowness [s/m]
    s_x_z = 1 / c_x_z  # Slowness = 1/(Speed of Sound)
    s_z = np.mean(s_x_z, axis=1)  # Mean Slowness vs Depth (z)
    ds_x_z = s_x_z - np.tile(s_z[:, np.newaxis], (1, x.size))  # Slowness Deviation

    # Generate Wavefield at Each Frequency
    wf_x_z_f = np.zeros((z.size, x.size, f.size)).astype("complex64")
    for f_idx in np.arange(f.size):
        # Continuous Wave Response By Downward Angular Spectrum
        wf_x_z_f[0, :, f_idx] = tx_x_f[:, f_idx]  # Injection Surface (z = 0)
        for z_idx in np.arange(z.size - 1):
            kz_squared = (f[f_idx] * s_z[z_idx]) ** 2 - kx ** 2
            kz_squared[kz_squared < 0] = 0  # Set negative values to 0 to avoid invalid sqrt
            kz = np.sqrt(kz_squared)  # Now apply sqrt to the non-negative values
            H = np.exp(-1j * 2 * np.pi * kz * dz)  # Propagation Filter in Spatial Frequency Domain
            H[kz_squared <= 0] = 0  # Remove Evanescent Components

            # Create Phase-Shift Correction in Spatial Domain
            dH = np.exp(-1j * 2 * np.pi * f[f_idx] * ds_x_z[z_idx, :] * dz)
            # Downward Continuation with Split-Stepping
            wf_x_z_f[z_idx + 1, :, f_idx] = bckgnd_wf_x_z_f[
                z_idx + 1, :, f_idx
            ] + dH * ift(H * ft(wf_x_z_f[z_idx, :, f_idx]))
    return wf_x_z_f


# Angular Spectrum Method Upwards into the Medium from the Receivers
def upward_continuation(x, z, c_x_z, f, tx_x_f, aawin):

    # Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
    ft = lambda sig: np.fft.fftshift(np.fft.fft(aawin * sig, axis=0), axes=0)
    ift = lambda sig: aawin * np.fft.ifft(np.fft.ifftshift(sig, axes=0), axis=0)

    # Spatial Grid
    dx = np.mean(np.diff(x))
    nx = x.size
    x = dx * np.arange(-(nx - 1) / 2, (nx - 1) / 2 + 1)
    dz = np.mean(np.diff(z))

    # FFT Axis for Lateral Spatial Frequency
    kx = np.mod(
        np.fft.fftshift(np.arange(nx) / (dx * nx)) + 1 / (2 * dx), 1 / dx
    ) - 1 / (2 * dx)

    # Convert to Slowness [s/m]
    s_x_z = 1 / c_x_z  # Slowness = 1/(Speed of Sound)
    s_z = np.mean(s_x_z, axis=1)  # Mean Slowness vs Depth (z)
    ds_x_z = s_x_z - np.tile(s_z[:, np.newaxis], (1, x.size))  # Slowness Deviation

    # Generate Wavefield at Each Frequency
    wf_x_z_f = np.zeros((z.size, x.size, f.size)).astype("complex64")
    for f_idx in np.arange(f.size):
        # Continuous Wave Response By Downward Angular Spectrum
        wf_x_z_f[-1, :, f_idx] = tx_x_f[:, f_idx]  # Injection Surface (z = 0)
        for z_idx in np.arange(z.size - 1, 1, -1):
            kz_squared = (f[f_idx] * s_z[z_idx]) ** 2 - kx ** 2
            kz_squared[kz_squared < 0] = 0  # Set negative values to 0 to avoid invalid sqrt
            kz = np.sqrt(kz_squared)  # Now apply sqrt to the non-negative values
            H = np.exp(1j * 2 * np.pi * kz * dz)  # Propagation Filter in Spatial Frequency Domain
            H[kz_squared <= 0] = 0  # Remove Evanescent Components

            # Create Phase-Shift Correction in Spatial Domain
            dH = np.exp(1j * 2 * np.pi * f[f_idx] * ds_x_z[z_idx, :] * dz)
            # Downward Continuation with Split-Stepping
            wf_x_z_f[z_idx - 1, :, f_idx] = ift(H * ft(dH * wf_x_z_f[z_idx, :, f_idx]))
    return wf_x_z_f
