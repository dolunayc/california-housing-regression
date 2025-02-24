# California Housing Price Prediction

This project implements a **Linear Regression model** to predict **housing prices** in California using the **California Housing Dataset** from `sklearn.datasets`.

## 📂 Project Structure

```
.
├── california_housing_regression.py  # Main script for regression
├── README.md                         # Project documentation
```

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/dolunayc/california-housing-regression.git
cd california-housing-regression
```

### 2️⃣ Install Dependencies

Ensure that you have the required dependencies installed:

```sh
pip install numpy pandas scikit-learn matplotlib
```

### 3️⃣ Run the Regression Script

```sh
python california_housing_regression.py
```

## 🏠 Dataset Information

- **Source:** `sklearn.datasets.fetch_california_housing()`
- **Features:** 8 continuous variables (e.g., MedInc, HouseAge, AveRooms, etc.)
- **Target:** Median house value in \$100,000s
- **Objective:** Predict house prices based on feature attributes.

## 🧠 Machine Learning Model Used

The script implements **Linear Regression** for prediction.

## 📊 Model Evaluation Metrics

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**

## 📈 Visualization

- The script generates **scatter plots** comparing actual vs. predicted values.

## 📝 Notes

- The dataset is preprocessed before training.
- Train-test split ratio is **80%-20%**.



