# Data Science Challenge: Memprediksi Kualitas Jeruk Berdasarkan Dataset
![gambar](https://github.com/hallosayaimroatubelajargithub/projek_prediksi/blob/main/orange.jpg)

Merupakan website aplikasi yang dibangun dengan Streamlit, dirancang untuk memprediksi kualitas jeruk berdasarkan berbagai fitur seperti size, weight, sweetness, acidity, softness, ripeness, dan banyak lagi. Aplikasi ini menggunakan model pembelajaran mesin untuk membuat prediksi dan mengevaluasi performa model.

## Features
`1.` <b>Data Upload:</b> Pengguna dapat mengunggah file CSV yang berisi fitur-fitur relevan dari dataset jeruk.\
`2.` <b>Model Selection:</b> Decision Tree Regressor, Random Forest Regressor, dan Gradient Boosting Regressor.\
`3.` <b>Model Evaluation:</b> R-squared (R²), Mean Absolute Error (MAE), dan Mean Squared Error (MSE).\
`4.` <b>Predicted vs Actual Plot:</b> Aplikasi memvisualisasikan nilai prediksi vs aktual untuk memberikan wawasan tentang kinerja model.\
`5.` <b>Residual Plot:</b> Plot residual ditampilkan untuk membantu pengguna mengevaluasi distribusi kesalahan model.\
`6.` <b>Feature Importance:</b> Untuk model berbasis pohon seperti Random Forest dan Gradient Boosting, aplikasi menampilkan pentingnya setiap fitur dalam membuat prediksi.

## Project Structure
```bash
├── app.py                # The main homepage of the application
└── requirements.txt       # List of dependencies for the project
```

## Requirements
```bash
streamlit
pandas
scikit-learn
numpy
matplotlib
seaborn
statsmodels
```

## How to Run
```bash
# Step 1 -> Clone repository
git clone https://github.com/your-username/orange-quality-prediction-app.git

# Step 2 -> Navigate to project directory
cd orange-quality-prediction-app

# Step 3 -> Install dependencies
pip install -r requirements.txt

# Last step -> Run the application
streamlit run app.py
```
## Dataset
<b>Link:</b> https://www.kaggle.com/datasets/shruthiiiee/orange-quality

## Online Access
<b>Link:</b> https://final-project-analyst-sentiment-250602.streamlit.app/

## Full Version on Medium
<b>Link:</b> https://medium.com/@imroatuslch/lets-try-fd404ef114c9

## MIT License
Copyright (c) 2024 Imroatu Solicah
