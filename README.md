# EEG Signal Classification for Motor Imagery using Deep Learning

This repository contains the MATLAB code developed as part of my M.Eng. thesis at the University of West Attica. The goal was to classify EEG signals during motor imagery tasks using deep learning models such as CNNs, LSTMs, and Feedforward Networks (FFNs).

📄 **Read the full thesis:**  
🔗 [Polynoe Library - University of West Attica](https://polynoe.lib.uniwa.gr/xmlui/handle/11400/10006)

---

## 🧠 Project Overview

The system classifies EEG signals recorded during motor imagery (MI) tasks using data from **BCI Competition IV – Dataset 1**. The focus was on analyzing models that can be used in Brain-Computer Interfaces (BCIs), with potential applications in neuroprosthetics.

### Objectives:
- Preprocess raw EEG signals with bandpass filters
- Segment data and apply trial-based classification
- Train and evaluate:
  - A shallow **CNN**
  - A **LSTM** model
  - A **Feedforward Neural Network (FFN)**

---

## 📁 Repository Structure

	•	cnn_best.m – CNN model implementation
	•	Swallow_LSTM.m – LSTM model implementation
	•	ffn_trial.m – Feedforward Neural Network (FFN) with trial-based testing
	•	create_data_stacked.m – Trial-based data creation script used for FFN
	•	preprocessEEG.m – EEG preprocessing using a 4th-order Butterworth bandpass filter (8–35 Hz)
	•	visualizeEEGSegment.m – Visualization of EEG signal before and after preprocessing
	•	swallow_create_data.asv – Archived/older version of EEG data creation
	•	visual.m – Utility functions for EEG signal plotting
---

## 🧪 Dataset Used

- **Dataset**: [BCI Competition IV - Dataset 1](http://www.bbci.de/competition/iv/)
- **Subjects**: Single-subject data used for calibration and evaluation
- **Channels**: 59 EEG channels
- **Sampling Rate**: 100 Hz
- **Task**: Motor imagery (left vs right hand)

---

## ⚙️ How to Run the Code

1. Clone the repository or download the `.m` files
2. Download the dataset and place it in the appropriate folder
3. In MATLAB:
   ```matlab
   run('cnn_best.m')       % To train/test CNN model
   run('Swallow_LSTM.m')   % To train/test LSTM model
   run('ffn_trial.m')      % To train/test FFN model
