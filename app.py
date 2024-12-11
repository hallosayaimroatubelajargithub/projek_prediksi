import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title
st.title("Orange Quality Prediction App")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data:")
    st.write(data.head())

    # Preprocessing: Encode categorical variables
    data = pd.get_dummies(data, columns=['Color', 'Variety', 'Blemishes (Y/N)'], drop_first=True)

    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = ['Size (cm)', 'Weight (g)', 'Brix (Sweetness)', 
                          'pH (Acidity)', 'Softness (1-5)', 'HarvestTime (days)', 'Ripeness (1-5)']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Feature Engineering: Add Sweetness/Acidity ratio
    data['Sweetness_Acidity_Ratio'] = data['Brix (Sweetness)'] / (data['pH (Acidity)'] + 1e-6)

    # Separate features and target
    X = data.drop(columns=['Quality (1-5)'])
    y = data['Quality (1-5)']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    st.write("### Select Algorithm for Training")
    model_choice = st.selectbox("Choose a model:", ["Decision Tree", "Random Forest", "Gradient Boosting"])

    if st.button("Train and Evaluate Model"):
        if model_choice == "Decision Tree":
            # Decision Tree with Grid Search for Hyperparameter Tuning
            param_grid = {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            decision_tree = DecisionTreeRegressor(random_state=42)
            grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

        elif model_choice == "Random Forest":
            # Random Forest Regressor
            best_model = RandomForestRegressor(random_state=42)
            best_model.fit(X_train, y_train)

        elif model_choice == "Gradient Boosting":
            # Gradient Boosting Regressor
            best_model = GradientBoostingRegressor(random_state=42)
            best_model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # Display metrics
        st.write("### Model Evaluation Metrics:")
        st.write(f"**R-squared:** {r2:.3f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.3f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.3f}")

        # Predicted vs Actual Plot
        st.write("### Predicted vs Actual Values")
        fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axis
        ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values')
        st.pyplot(fig)  # Pass the figure object to st.pyplot

        # Residual Plot
        st.write("### Residual Plot")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axis
        sns.residplot(x=y_pred, y=residuals, lowess=True, color="green", line_kws={'color': 'red', 'lw': 2}, ax=ax)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predicted Values')
        st.pyplot(fig)  # Pass the figure object to st.pyplot

        # Feature Importance (if using tree-based models)
        if model_choice in ["Random Forest", "Gradient Boosting"]:
            st.write("### Feature Importance")
            feature_importances = best_model.feature_importances_
            feature_names = X.columns
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)
            st.write(importance_df)

            # Plot Feature Importance
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)  # Pass the figure object to st.pyplot

else:
    st.write("Please upload a CSV file to proceed.")

