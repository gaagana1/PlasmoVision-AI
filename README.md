# PlasmoVision AI: High-Throughput Malaria Parasite Screening
# By Gabriel Agana Anongwin
# Doctor of Pharmacy Class of 2026
# Data Science Student at ALX, Cohort 10
An automated digital pathology dashboard designed to accelerate malaria screening by enabling healthcare workers to evaluate large batches of blood cell microscopic images simultaneously.



## Problem Statement
Malaria remains a critical global epidemiology issue, demanding rapid mass screening in regions with limited pathological resources. Traditional diagnosis relies heavily on manual blood smear microscopy, where field health workers must examine cells one by one under a microscope. This process is time-consuming, mentally exhausting, and prone to human diagnostic fatigue.



**PlasmoVision AI** solves this problem. By introducing deep learning computer vision to the frontline, this system allows health workers to drop multiple cell specimens into a high-throughput pipeline and screen thousands of cells at once—dramatically reducing screening time and maximizing early detection capacity.



## Dataset Reference
The custom Convolutional Neural Network (CNN) model was trained using the official, gold-standard cell image collection hosted by the National Institutes of Health (NIH).
* **Dataset Link:** [NLM/NIH Malaria Dataset on Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)



## 🛠️ Features
* **High-Throughput Batch Uploading:** Upload multiple thin blood smear specimens simultaneously.
* **Deep Learning Pathology Core:** Employs a customized CNN optimized for binary classification (Parasitized vs. Uninfected).
* **Stakeholder Alignment:** Designed to conform to rapid deployment guidelines outlined by the World Health Organization (WHO) and Africa CDC.



## Model Architecture & Training Specifications
The core of PlasmoVision AI is a custom-built Convolutional Neural Network (CNN) engineered from scratch using TensorFlow and Keras, later optimized via ONNX Runtime for deployment.

### Training Hyperparameters
* **Architecture:** 3-Layer Sequential CNN (Conv2D -> MaxPooling2D -> Dropout -> Dense Output)
* **Input Resolution:** 150x150x3 (RGB)
* **Optimization Function:** Adam Optimizer
* **Loss Function:** Binary Crossentropy (Optimized for Infected vs. Healthy classification)
* **Training Duration:** 5 Epochs
* **Regularization:** Dropout (0.5) to prevent overfitting and ensure real-world generalization



### Evaluation Metrics (Final Epoch Results)
The network achieved outstanding metrics by the end of its training cycle, demonstrating clinical-grade reliability:
* **Training Accuracy:** 95.2%
* **Validation Accuracy:** 94.5%
* **Recall (Sensitivity):** 96.8%

> **Note on Metrics:**
 In medical diagnostics, minimizing False Negatives is critical. The high **Recall (96.8%)** score ensures that the model successfully catches nearly all true positive malaria infections, making it an excellent high-throughput screening tool for frontline workers.