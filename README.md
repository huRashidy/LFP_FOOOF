# Aperiodicity in Mouse CA1 and DG Power Spectra

This repository contains the code and data modifications used in the study **"Aperiodicity in Mouse CA1 and DG Power Spectra"**. The study focuses on analyzing the aperiodic components of hippocampal power spectra in mice, specifically in the CA1 and dentate gyrus (DG) regions. We modified the open-source [FOOOF toolbox](https://github.com/fooof-tools/fooof) to improve the assessment of periodic and aperiodic components in electrophysiological signals.


## Repository Structure

The repository is organized as follows:

├── Fitting examples/ # Example data and scripts for fitting

│ ├── data/ # Example datasets

│ │ ├── CA1_example.mat # CA1 LFP recording example

│ │ └── DG_example.mat # DG LFP recording example

│ └── Example_fitting.ipynb # Jupyter notebook for analyzing CA1 and DG LFP recordings

├── FOOOF codes/ # Modified FOOOF toolbox


│ ├── funcs.py # Modified helper functions (originally in specparam/core)
│ └── fit.py # Modified fitting functionality (originally in specparam/objs)


├── README.md # This file


## Key Modifications to the FOOOF Toolbox

We made the following critical modifications to the FOOOF toolbox:

1. **Improved Accuracy**:
   - Reduced the assessment error of periodic components from **3% to 0.1%** using simulated data.
   - Enhanced the fitting algorithm to better capture aperiodic components.

2. **Aperiodic Component Analysis**:
   - Added 3 aperiodic functions:
     - **2 Exponents (up to 200 Hz)**: For fitting lower frequency ranges.
     - **2 Exponents + Flattening (up to 500 Hz)**: For fitting broader frequency ranges with a flattening component.
     - **3 Exponents (up to 500 Hz)**: For fitting broader frequency ranges with an additional exponent.

---

## How to Use This Repository

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo


### 2. Install Dependencies
Ensure you have the following dependencies installed:
Python 3.7+
NumPy
SciPy
Matplotlib (for visualization)



## Related Publication
This work is part of the study "Aperiodicity in Mouse CA1 and DG Power Spectra". For more details, refer to the preprint link 
[Biorxiv preprint]([https://github.com/fooof-tools/fooof)](https://www.biorxiv.org/content/10.1101/2025.01.30.635678v1)
