# Speech Emotion Recognition (SER) using CNN & LSTM / GRU

This repository contains the source code for a **Speech Emotion Recognition (SER)** system, developed as a department project. The model predicts human emotions from audio speech using deep learning techniques, specifically a hybrid architecture combining **Convolutional Neural Networks (CNN)** and **Recurrent Neural Networks (LSTM/GRU)**.

A user-friendly web interface is additionally provided using **Gradio**, allowing real-time emotion prediction from uploaded `.wav` audio files.

---

## 📊 Dataset

The model is trained on the **[RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)](https://zenodo.org/record/1188976)** dataset. The dataset consists of emotional speech utterances covering 8 distinct emotions:
- 😐 Neutral
- 😌 Calm
- 😊 Happy
- 😢 Sad
- 😡 Angry
- 😨 Fearful
- 🤢 Disgust
- 😲 Surprised

---

## 🛠️ Tech Stack & Libraries
- **Language**: Python 3.x
- **Deep Learning**: TensorFlow, Keras, Keras Tuner
- **Audio Processing**: Librosa
- **Data Preprocessing & Metrics**: Scikit-Learn, Imbalanced-Learn (SMOTE)
- **Visualization**: Matplotlib, Seaborn
- **Web Interface**: Gradio

---

## 🧠 Model Architecture & Methodology

1. **Feature Extraction**: 
   - Uses `Librosa` to extract **MFCCs (Mel-Frequency Cepstral Coefficients)** (40 coefficients).
   - Computes delta and delta-delta features to capture dynamic audio characteristics.
   - Pads and trims sequences to a uniform length of 150 frames.
2. **Data Preprocessing**:
   - Classes are encoded using `LabelEncoder`.
   - Feature normalization using `StandardScaler` / `MinMaxScaler`.
   - Addressed class imbalance within the dataset utilizing **SMOTE (Synthetic Minority Over-sampling Technique)**.
3. **Deep Learning Model**:
   - Designed using `tf.keras.Sequential`.
   - **2D Convolutional Layers (CNN)** coupled with Max Pooling to capture spatial acoustic representations.
   - **LSTM / GRU Layers** to capture sequential context and temporal dependencies of speech over time.
   - Regularization applied through `Dropout` and `BatchNormalization` to prevent overfitting.
   - Best hyperparameters found using **Keras Tuner**.
4. **Early Stopping**: Halts training if validation loss does not improve for a designated number of epochs, ensuring optimum generalization.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Yesh-219/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Install Dependencies
```bash
pip install numpy librosa gradio matplotlib seaborn tensorflow scikit-learn imbalanced-learn keras-tuner
```

### 3. Setup Dataset
- Download the RAVDESS dataset and place the audio files in a suitable directory.
- Open the `.ipynb` notebook or python script and ensure that `dataset_path` correctly points to your RAVDESS dataset.

### 4. Train the Model
Run the Jupyter Notebook, or if exported to a script, execute:
```bash
python train.py
```
*(The trained model will be saved as `saved_model.h5` or `ser_model.h5`)*

### 5. Launch the Web App
Run the Gradio interface to start predicting using your own voice inputs!
```bash
python app.py
```
A local URL will be generated (e.g. `http://127.0.0.1:7860/`). Open it in your browser and upload audio files to test the SER model.

---

## 📈 Results
- **Training Accuracy**: ~99% 
- **Validation/Test Accuracy**: ~93 - 96%
*(Using optimum CNN-GRU architecture + SMOTE applied data)*

---

## 🤝 Acknowledgments
This project is completed as a part of the university Department Project curriculum. We acknowledge the authors of the RAVDESS dataset for making the data publicly available for research.


