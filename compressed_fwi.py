from Functions import upward_continuation, downward_continuation
import scipy.io as sio
import numpy as np
from scipy.interpolate import interpn
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Ultrasound Tomography (Full WaveForm Inversion) with Compressed Sensing")
parser.add_argument("--compression_ratio", type=float, default=0.25, help="Compression ratio for measurements (0 < ratio < 1)")
parser.add_argument("--lambda_reg", type=float, default=0.001, help="L1 regularization parameter")
parser.add_argument("--c_bkgnd", type=float, default=1525, help="Background speed of sound (m/s)")
parser.add_argument("--input_file", type=str, default="sim_breast.mat", help="Input data file (.mat)")
parser.add_argument("--output_file", type=str, default="slowness_4x.npy", help="Output file for slowness model (.npy)")
parser.add_argument("--Niter", type=int, default=10, help="Number of iterations for optimization")

args = parser.parse_args()

# Load simulated dataset from specified input file
data_in = sio.loadmat(args.input_file)
P_f = data_in["P_f"][0]
f = data_in["f"][0]
t = data_in["t"][0]
xelem = data_in["xelem"][0]
rotAngle = data_in["rotAngle"][0]
recording_x_t = data_in["recording_x_t"]
array_separation = data_in["array_separation"][0, 0]
del data_in

# Set up time-frequency grid for Fourier transform
freq_bins_used = np.arange(0, int((f.size + 1) / 2), 10)  # Frequency bins to use
fused = f[freq_bins_used]  # Frequencies used
P_fused = P_f[freq_bins_used]  # Pulse frequency bins
dt = np.mean(np.diff(t))
ff, tt = np.meshgrid(fused, t)
delays = np.exp(-1j * 2 * np.pi * ff * tt) * dt  # Fourier transform delays

# Transform time-domain data to frequency domain
recording_x_f = np.zeros((xelem.size, fused.size, rotAngle.size), dtype="complex64")
for rot_idx in range(rotAngle.size):
    recording_x_f[:, :, rot_idx] = np.dot(recording_x_t[:, :, rot_idx], delays)
del recording_x_t

# Compress data using random Gaussian matrix (compressed sensing)
M = int(recording_x_f.shape[0] * args.compression_ratio)
assert M < recording_x_f.shape[0], "Compressed size must be less than original"
Phi = np.random.randn(M, recording_x_f.shape[0]) / np.sqrt(M)  # Normalized compression matrix
compressed_recording_x_f = np.zeros((M, fused.size, rotAngle.size), dtype="complex64")
for f_idx in range(fused.size):
    for rot_idx in range(rotAngle.size):
        compressed_recording_x_f[:, f_idx, rot_idx] = Phi @ recording_x_f[:, f_idx, rot_idx]

# Simulation grid setup
Nzi, Nxi = 201, 256  # Grid dimensions
zi = np.linspace(-array_separation / 2, array_separation / 2, Nzi)
dxi = np.mean(np.diff(xelem))
xi = np.arange(-(Nxi - 1) / 2, (Nxi - 1) / 2 + 1) * dxi
Xi, Zi = np.meshgrid(xi, zi)
Ri = np.sqrt(Xi**2 + Zi**2)
XELEM_GRID, XI = np.meshgrid(xelem, xi)
x_src_idx = np.argmin(np.abs(XI - XELEM_GRID), axis=0)

# Anti-aliasing window
ord = 100
xmax = (np.max(np.abs(xi)) + np.max(np.abs(xelem))) / 2
aawin = 1.0 / np.sqrt(1 + (xi / xmax) ** ord)

# Source signal for angular spectrum method
tx_x_f = np.zeros((xi.size, fused.size), dtype="complex64")
tx_x_f[x_src_idx, :] = np.tile(P_fused[np.newaxis, :], [x_src_idx.size, 1])

# Slowness model grid
lateral_span = xelem.size * np.mean(np.diff(xelem))
radial_span = np.sqrt(array_separation**2 + lateral_span**2)
Nx, Nz = 601, 601
x = np.linspace(-radial_span / 2, radial_span / 2, Nx)
z = np.linspace(-radial_span / 2, radial_span / 2, Nz)
X, Z = np.meshgrid(x, z)
R = np.sqrt(X**2 + Z**2)
dx, dz = np.mean(np.diff(x)), np.mean(np.diff(z))
grid_conv_factor = (dx / dxi) * (dz / dxi)

# Initial slowness model
c_bkgnd = args.c_bkgnd
slowness = np.ones(X.shape) / c_bkgnd

# Nonlinear conjugate gradient parameters
search_dir = np.zeros((Nz, Nx))
gradient_img_prev = np.zeros((Nz, Nx))

# Main optimization loop
plt.figure(figsize=(16, 6))
for iter in tqdm(range(args.Niter)):
    gradient_img = np.zeros(slowness.shape)
    scaling = np.zeros((rotAngle.size, fused.size), dtype="complex64")
    ddwf_x_z_f_rot = np.zeros((Nzi, Nxi, fused.size, rotAngle.size), dtype="complex64")

    # Process each rotation angle
    for rot_idx in range(rotAngle.size):
        # Rotate slowness for forward modeling
        Ti = np.arctan2(Zi, Xi) - rotAngle[rot_idx]
        C = interpn(
            (z, x), 1 / slowness, (Ri * np.sin(Ti), Ri * np.cos(Ti)),
            method="linear", bounds_error=False, fill_value=c_bkgnd
        )

        # Forward modeling (downward continuation)
        dwf_x_z_f = downward_continuation(xi, zi, C, fused, tx_x_f, np.zeros((zi.size, xi.size, fused.size)), aawin)
        forwardProject_x_f = dwf_x_z_f[-1, x_src_idx, :]  # Uncompressed modeled data
        compressed_forwardProject_x_f = Phi @ forwardProject_x_f  # Compressed modeled data

        # Scale simulated data to match measurements
        for f_idx in range(fused.size):
            REC = compressed_recording_x_f[:, f_idx, rot_idx]
            REC_SIM = compressed_forwardProject_x_f[:, f_idx]
            scaling[rot_idx, f_idx] = np.inner(np.conj(REC_SIM), REC) / np.inner(np.conj(REC_SIM), REC_SIM + 1e-12)
            forwardProject_x_f[:, f_idx] *= scaling[rot_idx, f_idx]
            dwf_x_z_f[:, :, f_idx] *= scaling[rot_idx, f_idx]

        # Data misfit gradient
        residual_x_f = compressed_forwardProject_x_f - compressed_recording_x_f[:, :, rot_idx]
        res_x_f = np.zeros((xi.size, fused.size), dtype="complex64")
        res_x_f[x_src_idx, :] = Phi.T @ residual_x_f  # Adjoint of compression
        uwf_x_z_f = upward_continuation(xi, zi, C, fused, res_x_f, aawin)
        ddwf_x_z_f = -1j * 2 * np.pi * np.mean(np.diff(zi)) * fused * dwf_x_z_f
        ddwf_x_z_f_rot[:, :, :, rot_idx] = ddwf_x_z_f
        grad_img_data_misfit = np.real(np.sum(ddwf_x_z_f * np.conj(uwf_x_z_f), axis=2))

        # L1 regularization gradient (promotes sparsity in forwardProject_x_f)
        sign_forwardProject_x_f = np.sign(forwardProject_x_f)  # Subgradient of ||F(x)||_1
        res_L1_x_f = np.zeros((xi.size, fused.size), dtype="complex64")
        res_L1_x_f[x_src_idx, :] = sign_forwardProject_x_f
        uwf_x_z_f_L1 = upward_continuation(xi, zi, C, fused, res_L1_x_f, aawin)
        grad_img_L1 = np.real(np.sum(ddwf_x_z_f * np.conj(uwf_x_z_f_L1), axis=2))

        # Combine gradients
        grad_img = grad_img_data_misfit + args.lambda_reg * grad_img_L1

        # Accumulate gradient
        T = np.arctan2(Z, X) + rotAngle[rot_idx]
        gradient_img += interpn(
            (zi, xi), grad_img, (R * np.sin(T), R * np.cos(T)),
            method="linear", bounds_error=False, fill_value=0
        )

    # Update search direction
    nsteps_reset = 100
    if iter % nsteps_reset == 0:
        beta = 0
    else:
        betaPR = np.dot(gradient_img.flatten(), gradient_img.flatten() - gradient_img_prev.flatten()) / \
                 np.dot(gradient_img_prev.flatten(), gradient_img_prev.flatten())
        betaFR = np.dot(gradient_img.flatten(), gradient_img.flatten()) / \
                 np.dot(gradient_img_prev.flatten(), gradient_img_prev.flatten())
        beta = min(max(betaPR, 0), betaFR)
    search_dir = beta * search_dir - gradient_img
    gradient_img_prev = gradient_img

    # Forward projection of search direction
    drecording_x_f = np.zeros((xelem.size, fused.size, rotAngle.size), dtype="complex64")
    for rot_idx in range(rotAngle.size):
        Ti = np.arctan2(Zi, Xi) - rotAngle[rot_idx]
        C = interpn(
            (z, x), 1 / slowness, (Ri * np.sin(Ti), Ri * np.cos(Ti)),
            method="linear", bounds_error=False, fill_value=c_bkgnd
        )
        SEARCH_DIR = interpn(
            (z, x), search_dir, (Ri * np.sin(Ti), Ri * np.cos(Ti)),
            method="linear", bounds_error=False, fill_value=c_bkgnd
        )
        ddwf_x_z_f = downward_continuation(
            xi, zi, C, fused, np.zeros((xi.size, fused.size)),
            ddwf_x_z_f_rot[:, :, :, rot_idx] * np.tile(SEARCH_DIR[:, :, np.newaxis], (1, 1, fused.size)), aawin
        )
        drecording_x_f[:, :, rot_idx] = ddwf_x_z_f[-1, x_src_idx, :]

    # Compress search direction projection
    compressed_drecording_x_f = np.zeros((M, fused.size, rotAngle.size), dtype="complex64")
    for rot_idx in range(rotAngle.size):
        compressed_drecording_x_f[:, :, rot_idx] = Phi @ drecording_x_f[:, :, rot_idx]

    # Line search step size
    perc_step_size = 1
    numerator = -np.dot(gradient_img.flatten(), search_dir.flatten())
    denominator = np.dot(compressed_drecording_x_f.flatten(), np.conj(compressed_drecording_x_f.flatten()))
    alpha = numerator / denominator

    # Update slowness model
    slowness += perc_step_size * np.real(alpha) * grid_conv_factor * search_dir

# Save the final slowness model to the specified output file
np.save(args.output_file, slowness)