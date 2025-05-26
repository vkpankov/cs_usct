# cs_usct

A Python project for **Ultrasound Tomography with Compressed Sensing** using **Full Waveform Inversion (FWI)**. It reconstructs sound speed (slowness) models from compressed ultrasound measurements

## Usage

Run the script with command-line arguments:

```bash
python ultrasound_tomography.py --compression_ratio 0.25 --lambda_reg 0.001 --c_bkgnd 1525 --input_file "sim_breast.mat" --output_file "slowness_4x.npy" --Niter 10
```

## Command-Line Arguments

- `--compression_ratio` (float, default: 0.25): Ratio of compressed measurements to original (0 < ratio < 1).
- `--lambda_reg` (float, default: 0.001): L1 regularization parameter for sparsity.
- `--c_bkgnd` (float, default: 1525): Background speed of sound (m/s).
- `--input_file` (str, default: "sim_breast.mat" from https://github.com/rehmanali1994/FullWaveformInversionUSCT/releases/download/v1.0.0/sim_breast.mat): Path to input `.mat` file with simulated data.
- `--output_file` (str, default: "slowness_4x.npy"): Path to save reconstructed slowness model (.npy).
- `--Niter` (int, default: 10): Number of optimization iterations.

## Requirements

- Python 3.x
- NumPy, SciPy, Matplotlib, TQDM

Install dependencies:
```bash
pip install numpy scipy matplotlib tqdm
```

## Citation

This project is based on code and methods from:

Ali, R. "Open-Source Full-Waveform Ultrasound Computed Tomography Based on the Angular Spectrum Method Using Linear Arrays". *Medical Imaging 2022: Ultrasonic Imaging and Tomography*. Vol. 12038. SPIE, 2022.

**Baseline Code**: [https://github.com/rehmanali1994/FullWaveformInversionUSCT](https://github.com/rehmanali1994/FullWaveformInversionUSCT)

## License

MIT License.