# 🏠 House Price Prediction using Linear Regression

This project implements a simple Machine Learning model to predict house prices based on three main features:
- **Living Area (sq ft)**
- **Number of Bedrooms**
- **Number of Bathrooms**

The dataset is taken from the [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition.

---

## 🔧 Technologies Usedk

- Python 3.10+
- pandas
- matplotlib
- seaborn
- scikit-learn

---

## 📊 Features Used

- `GrLivArea` – Above ground living area
- `BedroomAbvGr` – Bedrooms above ground
- `TotalBathrooms` – Computed as `FullBath + 0.5 * HalfBath`

---

## 📈 Model & Evaluation

We used **Linear Regression** from `scikit-learn`.  
The dataset was split into training and validation sets (80/20), and RMSE (Root Mean Squared Error) was used as the evaluation metric.

Example Output:

```bash
✅ Validation RMSE: 53371.56

🔍 Sample Actual vs Predicted House Prices:

 Actual Price  Predicted Price
      208500         195321.67
      181500         172404.29
      223500         230188.42
      ...
