# Human Activity Recognition with PyTorch: CPU, GPU and TPU Benchmarking

A reproducible sensor time-series classification project using PyTorch, a 1D Convolutional Neural Network (CNN), and comparative inference benchmarking across CPU, NVIDIA GPU, and Google TPU hardware.

## 1. Project Overview

This project implements a Human Activity Recognition (HAR) pipeline using smartphone inertial sensor data from the UCI HAR dataset.

The primary objectives are:

- Explore and preprocess multichannel inertial sensor time-series data.
- Develop and train a 1D CNN using PyTorch.
- Evaluate the trained model on an independent test set.
- Execute the same trained model on CPU, NVIDIA GPU, and Google TPU.
- Compare inference latency, throughput, and execution variability using a consistent benchmarking protocol.

**This is an experimental sensor-data classification and hardware benchmarking project, not an automotive perception system.** It provides practical experience with time-series processing and accelerator-based inference relevant to sensor-data applications.

## 2. Dataset

**Dataset:** [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

The dataset contains smartphone accelerometer and gyroscope measurements collected from 30 participants performing six activities.

| Property | Description |
|---|---|
| Participants | 30 |
| Activities | 6 |
| Sampling frequency | 50 Hz |
| Window length | 128 time steps |
| Window duration | 2.56 seconds |
| Selected channels | 6 |
| Training samples | 7,352 |
| Test samples | 2,947 |
| Total samples | 10,299 |

### Selected sensor channels

- Body acceleration: X, Y, Z
- Body angular velocity (gyroscope): X, Y, Z

The body acceleration channels are processed signals with the gravity component separated, rather than raw accelerometer measurements.

### Activity classes

1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

The original participant-independent training and test split was preserved. A separate participant-grouped validation subset was created from the training participants to avoid participant overlap between training and validation.

## 3. Data Processing Pipeline

The workflow includes:

1. Dataset loading and structural inspection.
2. Exploratory data analysis (EDA).
3. Activity-wise sensor signal visualization.
4. Statistical analysis of sensor channels.
5. Participant-independent training and test separation.
6. Participant-grouped training and validation split.
7. Conversion of sensor windows into PyTorch tensors.
8. Preparation of PyTorch Dataset and DataLoader objects.

The CNN receives input tensors in the following format:

`(batch_size, channels, sequence_length)`

For this project:

`(64, 6, 128)`

## 4. Model Architecture

A baseline 1D CNN was implemented using PyTorch.

| Layer | Configuration |
|---|---|
| Input | 6 sensor channels |
| Conv1D | 6 → 32 channels, kernel size 5 |
| ReLU | Activation |
| MaxPool1D | Kernel size 2 |
| Conv1D | 32 → 64 channels, kernel size 5 |
| ReLU | Activation |
| MaxPool1D | Kernel size 2 |
| Flatten | Feature flattening |
| Fully Connected | 2,048 → 128 |
| ReLU | Activation |
| Output | 128 → 6 classes |

**Trainable parameters:** 274,342

**Loss function:** CrossEntropyLoss

**Optimizer:** Adam

**Learning rate:** 0.001

The model produces six class logits. Softmax is not applied inside the model because CrossEntropyLoss operates directly on logits.

### Model evaluation

The final evaluation on the original held-out test set produced:

| Metric | Result |
|---|---:|
| Test samples | 2,947 |
| Test loss | 0.4671 |
| Test accuracy | 89.21% |

The trained checkpoint is stored in:

`models/har_cnn_baseline.pth`

## 5. Hardware Benchmarking

The same trained model checkpoint and test dataset were used for all three hardware experiments.

### Hardware platforms

| Platform | Environment | Execution backend |
|---|---|---|
| CPU | Local Windows machine | PyTorch CPU |
| GPU | Google Colab, NVIDIA Tesla T4 | CUDA |
| TPU | Google Colab, TPU v5e | Google TorchTPU |

### Benchmark configuration

| Parameter | Configuration |
|---|---|
| Model | HAR_CNN |
| Test samples | 2,947 |
| Batch size | 64 |
| Number of batches | 47 |
| Warm-up | 1 complete test-set pass |
| Measured runs | 3 |
| Execution mode | `torch.inference_mode()` |
| Timing | `time.perf_counter()` |

The benchmark measures the time required to process the complete test dataset in batches.

For GPU and TPU experiments, CPU-to-accelerator batch transfers are included in the measured time. Accelerator synchronization is performed before and after each timed pass to account for asynchronous execution.

The CPU experiment uses the same batching and warm-up protocol without accelerator transfers.

### Software environment

The environments were not identical:

| Platform | PyTorch version | Additional backend |
|---|---|---|
| CPU | 2.15.0+cpu | None |
| GPU | 2.14.1+cu130 | CUDA 13.0 |
| TPU | 2.14.1+cpu | TorchTPU 0.2.0 |

The TPU experiment used Google's TorchTPU backend. It did not use PyTorch/XLA.

## 6. Benchmark Results

### Performance comparison

| Metric | Local CPU | NVIDIA Tesla T4 | Google TPU v5e |
|---|---:|---:|---:|
| Average total time (ms) | 96.353 | **49.338** | 78.957 |
| Median total time (ms) | 97.061 | **46.847** | 78.872 |
| Standard deviation (ms) | 1.801 | 5.487 | **1.391** |
| Average batch time (ms) | 2.050 | **1.050** | 1.680 |
| Throughput (samples/s) | 30,585.36 | **59,730.93** | 37,323.94 |

### Relative throughput

Using the local CPU as the baseline:

| Platform | Relative throughput | Improvement over CPU |
|---|---:|---:|
| CPU | 1.00× | 0% |
| NVIDIA Tesla T4 | 1.95× | 95.3% |
| Google TPU v5e | 1.22× | 22.0% |

### Key observations

- The NVIDIA Tesla T4 achieved the highest throughput for this workload.
- The TPU v5e also outperformed the local CPU in measured throughput.
- The TPU had the lowest standard deviation across the three measured runs.
- The GPU showed higher run-to-run variability than the CPU and TPU.

These findings apply to the selected CNN, batch size, dataset, and runtime configurations. They do not establish that GPUs are generally faster than TPUs.

## 7. Experimental Limitations

The following limitations are important when interpreting the results:

1. **Different execution environments:** CPU benchmarking was performed locally, while GPU and TPU benchmarking were performed in Google Colab.
2. **Different software versions:** The three platforms used different PyTorch versions and execution backends.
3. **Limited repetitions:** Only three measured runs were performed per platform.
4. **Small model and workload:** The CNN has approximately 274K trainable parameters and uses a batch size of 64. Different workloads may produce different hardware rankings.
5. **Data transfer overhead:** GPU and TPU measurements include CPU-to-device transfers, whereas CPU execution does not require such transfers.
6. **Hardware scope:** The results represent one local CPU, one Tesla T4, and one TPU v5e configuration.

The benchmark should therefore be interpreted as a practical comparison of these three execution configurations, rather than a controlled general-purpose hardware evaluation.

## 8. Repository Structure

```text
sensor-har-pytorch-accelerators/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── har_preprocessed.pt
│
├── models/
│   └── har_cnn_baseline.pth
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_hardware_benchmarking.ipynb
│   ├── 05_gpu_benchmarking.ipynb
│   ├── 06_tpu_benchmarking.ipynb
│   └── 07_hardware_comparison_report.ipynb
│
├── results/
│   ├── inference_benchmark_cpu_standardized.csv
│   ├── inference_benchmark_gpu_standardized.csv
│   ├── inference_benchmark_tpu_standardized.csv
│   └── hardware_comparison.csv
│
├── reports/
│
├── paths.py
└── README.md
```

## 9. Reproducing the Experiments

### Local CPU environment

The project uses Conda to manage its Python dependencies and local CPU development environment.
Prerequisites:

- Anaconda or Miniconda
- Git

1. Clone the repository
```bash
git clone https://github.com/SujanKarna/sensor-har-pytorch-accelerators.git
cd sensor-har-pytorch-accelerators
```
2. Create the Conda environment
```bash
conda env create -f environment.yml
```
3. Activate the environment
```bash
conda activate sensor-har
```
4. Register the Jupyter kernel
```bash
python -m ipykernel install --user --name sensor-har --display-name "Python (sensor-har)"
```

### NVIDIA GPU

The GPU benchmarking notebook is designed to run in Google Colab with an NVIDIA T4 runtime.

1. Open `05_gpu_benchmarking.ipynb` in Colab.
2. Select a T4 GPU runtime.
3. Upload the processed dataset and trained model checkpoint.
4. Run the setup and benchmarking cells.
5. Download the resulting CSV file.

### Google TPU

The TPU benchmarking notebook is designed for Google's TorchTPU environment.

1. Open `06_tpu_benchmarking.ipynb` in Colab.
2. Connect to a supported TPU runtime.
3. Initialize the TorchTPU backend.
4. Upload the processed dataset and trained model checkpoint.
5. Run the setup and benchmarking cells.
6. Download the resulting CSV file.

The comparison notebook loads the three standardized CSV files from the local `results/` directory.

## 10. Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Google Colab
- NVIDIA CUDA
- Google TorchTPU

## 11. Project Scope and Learning Outcomes

This project demonstrates practical experience in:

- Multichannel inertial sensor time-series analysis.
- Sensor data preprocessing and exploratory data analysis.
- Deep learning model development with PyTorch.
- CNN-based time-series classification.
- Model evaluation using an independent test set.
- CPU, GPU, and TPU inference execution.
- Hardware benchmarking and performance analysis.
- Reproducible experimentation and result documentation.
---

**Project focus:** Practical sensor-data analysis, PyTorch-based deep learning, and comparative hardware inference benchmarking.