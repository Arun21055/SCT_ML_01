import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df['TotalBathrooms'] = train_df['FullBath'] + 0.5 * train_df['HalfBath']
test_df['TotalBathrooms'] = test_df['FullBath'] + 0.5 * test_df['HalfBath']

features = ['GrLivArea', 'BedroomAbvGr', 'TotalBathrooms']
X = train_df[features]
y = train_df['SalePrice']
X_test = test_df[features]

X = X.fillna(0)
X_test = X_test.fillna(0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
print(f"Validation RMSE: {rmse:.2f}")

comparison_df = pd.DataFrame({
    'Actual Price': y_val.values[:10],
    'Predicted Price': y_pred[:10].round(2)
})

print("🔍 Sample Actual vs Predicted House Prices:\n")
print(comparison_df.to_string(index=False))

plt.figure(figsize=(6, 4))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train_df, alpha=0.6)
plt.title("Living Area vs Sale Price")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual House Prices")
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')  # Reference line
plt.tight_layout()
plt.show()

test_predictions = model.predict(X_test)
submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_predictions})
submission.to_csv("house_price_predictions.csv", index=False)
print("Predictions saved to house_price_predictions.csv")
