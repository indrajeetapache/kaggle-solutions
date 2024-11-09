Used Car Price Prediction

📋 Table of Contents

Project Overview
Dataset
Installation & Setup
Modeling Approach
Evaluation
Results
Conclusion & Future Work
📌 Project Overview
This project is part of the Kaggle Playground Series, Season 4, Episode 9 competition. The task is to predict the prices of used cars based on various features using regression techniques. The goal is to minimize the error in predicting car prices and achieve a high rank on the leaderboard.

📂 Dataset
The dataset consists of information on used cars, including:

ID: Unique identifier
Model: Car model name
Year: Year of manufacture
Mileage: Distance the car has been driven
Engine Size: Engine capacity in liters
Fuel Type: Type of fuel used (e.g., Petrol, Diesel, Electric)
Transmission: Transmission type (Manual, Automatic, etc.)
Price: The target variable (price of the car in dollars)
The dataset was provided by the competition organizers on the Kaggle competition page.

⚙️ Installation & Setup
To run this project, you'll need the following dependencies:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn

If you're using Jupyter or Google Colab, make sure you have all the necessary libraries installed.

🛠 Modeling Approach
Data Preprocessing
Data Cleaning: Handled missing values and removed duplicates.
Feature Engineering:
Created new features like car_age from the Year.
Converted categorical variables using Label Encoding and One-Hot Encoding.
Scaling: Applied StandardScaler for numerical features to normalize the dataset.
Model Selection
Implemented various regression models including:
Linear Regression
Ridge and Lasso Regression
XGBoost Regressor (used as the final model due to its high performance)
Hyperparameter Tuning
Used GridSearchCV and RandomizedSearchCV for optimizing the hyperparameters of the models.
📊 Evaluation
The model was evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

python
Copy code
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
🏆 Results
The best performing model was XGBoost Regressor with the following metrics on the test set:


The model achieved satisfactory performance in predicting used car prices. Future improvements may include:

Utilizing Ensemble methods like stacking multiple models.
Experimenting with deep learning models for feature extraction.
Exploring more sophisticated feature engineering techniques.


👤 Author
Developed by Indrajit. Feel free to reach out for any queries or collaboration opportunities!
