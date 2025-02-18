# Handwritten Digit Recognition (MNIST Dataset)
This project trains a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0-9) using the **MNIST dataset**.

## Installation
Run the following command to install dependencies:
```bash
pip install numpy pandas matplotlib tensorflow keras seaborn scikit-learn
```

## Running the Project
### 1. Train the Model
```bash
python src/train.py
```
### 2. Predict a Digit
```bash
python src/predict.py
```

## Project Structure
```
handwritten_digit_recognition/
│── dataset/               # (Optional: If downloading manually)
│── models/                # (Save trained models)
│── notebooks/             # (Jupyter Notebook experiments)
│── src/                   # (Main Python files)
│   ├── train.py           # (Training script)
│   ├── predict.py         # (Prediction script)
│── README.md              # (Project description)
│── requirements.txt       # (Dependencies)
│── .gitignore             # (Git ignored files)
```

## GitHub Deployment
To push this project to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/handwritten-digit-recognition.git
git push -u origin main
```
