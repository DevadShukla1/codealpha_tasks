import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data = pd.read_csv("Code Alpha/Advertising.csv").iloc[:, 1:]

data['Total_Spend'] = data['TV'] + data['Radio'] + data['Newspaper']
X = data[['TV', 'Radio', 'Newspaper', 'Total_Spend']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

print("\nFeature Coefficients (Ad Impact on Sales):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

print("\nINSIGHTS:")
top_feature = X.columns[model.coef_.tolist().index(max(model.coef_, key=abs))]
print(f"{top_feature} advertising has the strongest impact on sales.")
print("Increasing TV and Radio spend together can boost results.")
print("Newspaper ads have lower ROI; consider reducing or reallocating that budget.")
print("Total_Spend helps understand combined effect of all channels.")

plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.tight_layout()
plt.show()
