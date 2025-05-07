# Future-Guided Learning: A Predictive Approach To Enhance Time-Series Forecasting

![Figure 1](fig_1.png)

**Figure 1:** (A) In the FGL framework, a teacher model operates in the relative future of a student model that focuses on long-term forecasting. After training the teacher on its future-oriented task, both models perform inference during the student’s training phase. The probability distributions from the teacher and student are extracted, and a loss is computed based on Equation 1. (A1) Knowledge distillation transfers information via the Kullback-Leibler (KL) divergence between class distributions. (B) In an event prediction setting, the teacher is trained directly on the events themselves, while the student is trained to forecast these events. Future labels are distilled from the teacher to the student, guiding the student to align more closely with the teacher model’s predictions, despite using data from the relative past. (C) In a regression forecasting scenario, the teacher and student perform short-term and long-term forecasting, respectively. Similar to event prediction, the student gains insights from the teacher during training, enhancing its ability to predict further into the future.

## Requirements

All module dependencies for this project can be found in the `requirements.txt` file. The experiments were conducted using:
- **Python Version:** 3.11.7
- **CUDA Version:** 12.1

Ensure that all required dependencies are installed before running the experiments.

---

## Dataset Description

This repository contains experiments conducted on three datasets:

### 1. AES Dataset (American Epilepsy Society Seizure Prediction Challenge)
This dataset consists of **5 dogs and 2 human patients** with **intracranial EEG (iEEG) recordings**. The recordings vary in length, seizure count, and number of channels. The full dataset (~50GB unzipped) can be downloaded from [Kaggle](https://www.kaggle.com/c/seizure-prediction). After downloading, place the patient data folders inside a `Dataset` directory.

### 2. CHB-MIT Dataset (Children's Hospital Boston - MIT)
This dataset contains **23 patients** with **intracranial EEG recordings**. The dataset (~50GB) is available on [PhysioNet](https://physionet.org/content/chbmit/1.0.0/). We train only on select patients with sufficient preictal data (specified in `main.py`). Place the downloaded patient data inside the `Dataset` folder.

### 3. Mackey Glass Dataset
This dataset differs from the seizure datasets, as it consists of samples generated from the **Mackey-Glass differential equation**. It provides a controlled setting for time-series forecasting experiments.

---

## Code Execution

### **Seizure Prediction & Detection (AES & CHB-MIT Datasets)**
The main files for executing experiments on the **AES** and **CHB-MIT** datasets are:
- `seizure_prediction.py` (Runs seizure prediction models)
- `seizure_detection.py` (Runs seizure detection models)
- `FGL_{AES/CHBMIT}.py` (Performs Future Guided Learning)
- `create_teacher.py` (Generates universal seizure detection models for AES)

For AES experimentation, before running `FGL_AES.py`, execute `create_teacher.py` to save the universal seizure detector models, which are required during forward passes in knowledge distillation on AES.

Example execution:
```bash
python FGL_CHBMIT.py --target Patient_1 --epochs 30 --trials 3 --optimizer_type Adam --alpha 0.5 --temperature 4
```

### **Mackey Glass Dataset**
For experiments on the **Mackey Glass** dataset, execute:
- `FGL_MG.py` (Runs teacher, baseline, and student models)

Example execution:
```bash
python FGL_MG.py --horizon 5 --alpha 0.5 --num_bins 50 --epochs 20 --optimizer SGD --temperature 4
```

---

## Parameter Tuning

Future Guided Learning used Knowledge Distillation, which involves two main hyperparameters:

- **Alpha (\(\alpha\))**: Controls the weighting between cross-entropy loss and KL divergence during distillation.
- **Temperature (T)**: Scales the softmax output for KL loss, regulating knowledge transfer.

Experimentation with different values of **\(\alpha\)** and **T** allows fine-tuning of the distillation process for better performance.

---

## Preprocessing Details

All preprocessing steps are implemented in the `utils/` folder for each dataset. The preprocessing varies across datasets:

### **AES & CHB-MIT Datasets:**
1. Classify EEG signals as **preictal, ictal, or interictal**.
2. Apply **Short-Time Fourier Transforms (STFT)**:
   - AES: Sampling rates of **200 Hz** or **1000 Hz**.
   - CHB-MIT: Sampling rate of **256 Hz**.
3. Convert EEG signals into **numpy arrays**.
4. For the **student model**, set:
   - **Seizure occurrence period:** 30 minutes
   - **Seizure prediction horizon:** 5 minutes

### **Mackey Glass Dataset:**
1. Generate **time series values** using the Mackey-Glass equation.
2. Bin the targets into discrete categories.
3. Create **data loaders** for training/testing.
4. A visualization of the binning process is available and can be uncommented for better interpretability.

---
## Citation
If you use this repository, please cite our work:
```
@article{gunasekaran2024future,
  title={Future-Guided Learning: A Predictive Approach To Enhance Time-Series Forecasting},
  author={Gunasekaran, Skye and Kembay, Assel and Ladret, Hugo and Zhu, Rui-Jie and Perrinet, Laurent and Kavehei, Omid and Eshraghian, Jason},
  journal={arXiv preprint arXiv:2410.15217},
  year={2024}
}
```