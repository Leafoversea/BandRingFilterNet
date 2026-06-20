# BandRingFilterNet

This repository contains the official implementation and experimental materials for the paper:

**BandRingFilterNet: Lightweight Global Modeling and Deployable Implementation via Ring-Band Spectral Modulation**

BandRingFilterNet (BRFNet) is a lightweight visual backbone based on ring-band spectral modulation. The main idea is to replace point-wise frequency-domain filter parameterization with compact ring-band sharing, reducing spectral filter storage while preserving global filtering behavior.

## Repository Structure

```text
BandRingFilterNet/
├── train/
│   ├── BRFNet_M/          # ImageNet-1K non-distilled BRFNet-M training code/results
│   ├── BRFNet_S/          # ImageNet-1K distilled BRFNet-S training code/results
│   ├── BRFNet_T/          # CIFAR-100 BRFNet-T training code/results
│   └── BRFNet_UT/         # BloodMNIST BRFNet-UT training/export/test files
├── no_modulation_5seed/   # Five-seed CIFAR-100 ablation: no modulation
├── pointwise_5seed/       # Five-seed CIFAR-100 ablation: point-wise spectral filtering
├── ring_5seeds/           # Five-seed CIFAR-100 ablation: ring-band spectral modulation
├── controlled deployment comparison/
│   ├── controlled deployment comparison.py
│   └── out/               # Deployment-side comparison outputs
├── throughout_brf_m/      # BRFNet-M throughput test materials
├── draw_power/            # Scripts/materials for visualization
├── code.zip               # Archived auxiliary code
├── ring2.zip              # Archived auxiliary materials
└── rings_radial_heatmap_true_outer_geometry.png
```

## Main Experiments

This repository provides materials for the main experiments reported in the paper:

* **ImageNet-1K classification**

  * BRFNet-M: non-distilled training setting
  * BRFNet-S: hard distillation setting with RegNetY-16GF teacher

* **CIFAR-100 ablation**

  * No modulation
  * GFNet-style point-wise spectral filtering
  * Ring-band spectral modulation

* **Controlled deployment comparison**

  * Matched point-wise vs. ring-band comparison under the same BRFNet-T-style backbone and CIFAR-100 training protocol
  * Reported metrics include frequency-domain parameter storage, GPU latency, peak GPU memory, batch latency, and throughput

* **BRFNet-UT / FPGA-oriented evaluation**

  * BRFNet-UT training/export/test materials for lightweight deployment-oriented evaluation

## Environment

The code is based on Python and PyTorch. A typical environment includes:

```bash
python >= 3.9
pytorch
torchvision
timm
numpy
pandas
matplotlib
fvcore
```

You may install the common dependencies with:

```bash
pip install torch torchvision timm numpy pandas matplotlib fvcore
```

Depending on your CUDA version and hardware platform, please install the appropriate PyTorch build from the official PyTorch website.

## Usage

### Train or evaluate BRFNet variants

Each model variant is placed under `train/`:

```text
train/BRFNet_M/
train/BRFNet_S/
train/BRFNet_T/
train/BRFNet_UT/
```

For example, BRFNet-T related code and outputs are located in:

```text
train/BRFNet_T/
```

BRFNet-UT export and test materials are located in:

```text
train/BRFNet_UT/
```

### Five-seed ablation experiments

The five-seed CIFAR-100 ablation materials are organized as:

```text
no_modulation_5seed/
pointwise_5seed/
ring_5seeds/
```

These folders correspond to the ablation comparison among no modulation, point-wise spectral filtering, and the proposed ring-band spectral modulation.

### Controlled deployment comparison

The matched deployment-side comparison is provided in:

```text
controlled deployment comparison/
```

This part compares GFNet-style point-wise spectral filtering and ring-band spectral modulation under the same BRFNet-T-style backbone and the same CIFAR-100 training recipe.

## Datasets

The paper uses public datasets including ImageNet-1K, CIFAR-100, and BloodMNIST. The datasets themselves are not included in this repository. Please download them from their official sources and modify the dataset paths in the scripts accordingly.

## Notes

This repository is intended to support reproducibility of the reported experiments and to provide reference implementations of the proposed ring-band spectral modulation design. Some auxiliary files are provided as archived materials for completeness.

## Citation

If you find this work useful, please cite:



## License

This project is released under the MIT License. See the `LICENSE` file for details.
