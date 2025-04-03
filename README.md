# TipPredictor 🚖💰  

TipPredictor is a machine learning model that predicts taxi tip amounts using a **Decision Tree Regressor**. The dataset is analyzed, preprocessed, and trained using **Scikit-Learn** to optimize performance.

## 🚀 Features  
- **Regression Tree Model:** Uses `DecisionTreeRegressor` to predict taxi tip amounts.  
- **Feature Importance Analysis:** Identifies key factors affecting tip amounts.  
- **Hyperparameter Tuning:** Compares different `max_depth` values to prevent overfitting.  
- **Dataset Preprocessing:** Normalizes numeric features and removes low-impact features.  

## 📊 Dataset  
The model uses the **NYC Yellow Taxi Trip Dataset** for training and evaluation.  
Data is sourced from:  
[Yellow Taxi Trip Data](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv)  

## 🛠 Installation  
1. Clone the repository:  
   ```sh
   git clone https://github.com/your-username/TipPredictor.git  
   cd TipPredictor  
   ```
2. Install dependencies:  
   ```sh
   pip install -r requirements.txt  
   ```
3. Run the script:  
   ```sh
   python Regression_Trees_Taxi_Tip.py  
   ```

## ⚡ Model Performance  
| Model Depth | MSE Score | R² Score |
|-------------|----------|---------|
| `max_depth=8` | `X.XXX` | `X.XXX` |
| `max_depth=4` | `X.XXX` | `X.XXX` |

Reducing `max_depth` to **4** improves generalization and prevents overfitting.

## 📜 License  
This project is licensed under the MIT License.  

---

🔍 **Want to contribute?** Fork the repo and submit a pull request! 🚀  
```

---
