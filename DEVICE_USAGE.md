# Device Configuration Usage Guide

This guide demonstrates how to use the new device (CUDA/CPU) flexibility features in SMPL-X Fitting.

## Configuration File (config.yaml)

The main configuration file now includes a device setting:

```yaml
general:
  device: "cuda"  # Options: "cuda", "cpu", "auto"
```

Device options:
- `"cuda"` - Use CUDA if available, fallback to CPU if not
- `"cpu"` - Force CPU usage
- `"auto"` - Automatically select CUDA if available, otherwise CPU
- `"cuda:0"`, `"cuda:1"`, etc. - Use specific GPU device

## Command Line Usage

All main scripts now accept a `--device` parameter that overrides the config file:

### fit_body_model.py

```bash
# Use CPU
python fit_body_model.py onto_scan --scan_path scan.obj --landmark_path landmarks.txt --device cpu

# Use CUDA
python fit_body_model.py onto_scan --scan_path scan.obj --landmark_path landmarks.txt --device cuda

# Auto-select device
python fit_body_model.py onto_scan --scan_path scan.obj --landmark_path landmarks.txt --device auto

# Use specific GPU
python fit_body_model.py onto_scan --scan_path scan.obj --landmark_path landmarks.txt --device cuda:1

# Dataset fitting
python fit_body_model.py onto_dataset -D FAUST --device cpu
```

### fit_vertices.py

```bash
# Use CPU
python fit_vertices.py onto_scan --scan_path scan.obj --landmark_path landmarks.txt --device cpu

# Use CUDA
python fit_vertices.py onto_scan --scan_path scan.obj --landmark_path landmarks.txt --device cuda

# Dataset fitting
python fit_vertices.py onto_dataset -D CAESAR --device auto
```

### refine_fitting.py

```bash
# Use CPU
python refine_fitting.py onto_scan --scan_path scan.obj --landmark_path landmarks.txt --device cpu

# Use CUDA
python refine_fitting.py onto_scan --scan_path scan.obj --landmark_path landmarks.txt --device cuda

# Dataset fitting  
python refine_fitting.py onto_dataset -D FAUST --device auto
```

### evaluate_fitting.py

```bash
# Evaluate with CPU
python evaluate_fitting.py chamfer /path/to/results /path/to/scan.obj --device cpu

# Evaluate with CUDA
python evaluate_fitting.py chamfer /path/to/results /path/to/scan.obj --device cuda
```

## Programmatic Usage

If you're calling the functions directly in Python:

```python
from utils import load_config, process_device_config

# Load config and set device
cfg = load_config()
cfg["device"] = "cpu"  # Override device setting
cfg = process_device_config(cfg)

# Now cfg["device"] contains a torch.device object
print(f"Using device: {cfg['device']}")

# Use in fitting functions
fit_body_model(input_dict, cfg)  # Will use the configured device
```

## Device Fallback Behavior

The system includes intelligent fallback behavior:

1. If you request CUDA but it's not available, it falls back to CPU with a warning
2. If you request a specific GPU (e.g., cuda:1) but only cuda:0 is available, it falls back to cuda:0
3. If you request a GPU ID that doesn't exist, it falls back to cuda:0 or CPU
4. The "auto" option automatically selects the best available device

## Migration from Old Code

If you have existing scripts or configurations:

1. **Config files**: Add `device: "auto"` to the `general` section
2. **Command line**: Add `--device auto` to maintain current behavior
3. **Python code**: The functions remain the same, just pass device in the config

The changes are backward compatible - if no device is specified, the system defaults to CUDA if available, CPU otherwise.