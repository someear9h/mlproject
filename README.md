# MStudent Exam Performance Prediction: Flask Application for Data Analysis, Model Training, and Deployment

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Setup Instructions](#setup-instructions)
- [Model Training Workflow](#model-training-workflow)
- [Data Preprocessing Techniques](#data-preprocessing-techniques)
- [Model Evaluation Metrics](#model-evaluation-metrics)
- [Flask Application](#flask-application)
- [Frontend Development](#frontend-development)
- [Deployment](#deployment)
- [Conclusion](#conclusion)

---

## Project Overview
This project demonstrates a complete end-to-end machine learning pipeline that includes:
- **Data ingestion**, preprocessing, and transformation.
- Exploratory Data Analysis (EDA) to uncover insights.
- Training and evaluation of machine learning models.
- Deployment of a machine learning model using a Flask web application.

The project involves both backend (model pipeline) and frontend (HTML/CSS integration) development, showcasing how to build a deployable machine learning solution.

---

## Technologies Used

### Programming Languages & Tools
- **Python**: Core programming language.
- **Jupyter Notebook**: For EDA and model training.
- **Flask**: Backend framework for model deployment.
- **HTML/CSS**: Frontend development.

### Python Libraries
- **Data Preprocessing**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning**: scikit-learn, XGBoost
- **Model Evaluation**: Metrics like RMSE, R2 score, accuracy
- **Other Utilities**: Logging, exception handling

---

## Features
- **End-to-End Machine Learning Pipeline**: Covers everything from data ingestion to model deployment.
- **Exploratory Data Analysis (EDA)**: Insightful visualizations and statistics.
- **Robust Preprocessing**: Handling missing data, feature engineering, and scaling.
- **Model Training**: Linear Regression and XGBoost.
- **Hyperparameter Tuning**: Optimized model performance using techniques like cross-validation.
- **Deployed Flask App**: Interactive web application to make predictions.
- **Frontend Development**: Clean and functional interface using HTML/CSS.

---

## Folder Structure
```plaintext
mlproject/
|-- app.py                  # Main Flask application
|-- requirements.txt        # Project dependencies
|-- artifacts/              # Contains processed datasets and model artifacts
|   |-- preprocessor.pkl
|   |-- model.pkl
|   |-- train.csv
|   |-- test.csv
|-- src/                    # Source code for backend
|   |-- components/
|       |-- data_ingestion.py
|       |-- data_transformation.py
|       |-- model_trainer.py
|   |-- pipeline/
|       |-- predict_pipeline.py
|       |-- train_pipeline.py
|   |-- utils.py
|   |-- logger.py
|   |-- exception.py
|-- templates/              # HTML templates for Flask app
|   |-- index.html
|-- static/                 # Static files (CSS, images, etc.)
|   |-- style.css
|-- notebooks/
    |-- data/
         |-- stud.csv            # Jupyter notebooks for EDA and model training
    |-- EDA STUDENT PERFORMANCE.ipynb
    |-- MODEL TRAINING.ipynb

```

---

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd mlproject
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask App**:
   ```bash
   python app.py
   ```
   Access the app at `http://127.0.0.1:8000/` in your browser.

---

## Model Training Workflow
### Steps:
1. **Data Ingestion**:
   - Load raw datasets.
   - Split into training and testing datasets.

2. **Data Transformation**:
   - Handle missing values.
   - Perform feature scaling and encoding.
   - Save the preprocessor as `preprocessor.pkl`.

3. **Model Training**:
   - Train models using algorithms such as Logistic Regression, Linear Regression, and XGBoost.
   - Perform hyperparameter tuning and cross-validation.
   - Save the best model as `model.pkl`.

4. **Model Evaluation**:
   - Evaluate models using metrics such as RMSE, R2 score, and accuracy.

---

## Data Preprocessing Techniques
- **Feature Engineering**: Created derived features for better model performance.
- **Handling Missing Data**: Imputation techniques for null values.
- **Scaling**: Standardized numerical features for uniformity.
- **Encoding**: Transformed categorical variables into numerical formats.

---

## Model Evaluation Metrics
- **Root Mean Squared Error (RMSE)**: Measures prediction errors.
- **R2 Score**: Indicates model goodness-of-fit.
- **Accuracy**: Evaluates classification performance.

---

## Flask Application
- **Endpoints**:
  - `/`: Displays the home page.
  - `/predictdata`: Accepts user inputs and provides predictions.

- **Backend**:
  - Integrates the pre-trained model (`model.pkl`) and preprocessor (`preprocessor.pkl`).
  - Uses `predict_pipeline.py` for inference.

---

## Frontend Development
- **HTML**:
  - Clean layout for user input and result display.
- **CSS**:
  - Added styling to improve user experience.

---

## Deployment
The project is deployed as a Flask application and can be further enhanced by deploying it on cloud platforms such as AWS, Azure, or Heroku.

---

## Conclusion
This project demonstrates the implementation of a complete machine learning pipeline, combining backend logic, frontend development, and model deployment. It showcases:
- Real-world data analysis and machine learning model development.
- End-to-end integration from preprocessing to deployment.
- Practical experience with Flask, HTML/CSS, and machine learning libraries.

## Contact
Feel free to reach out if you have any questions or want to collaborate!

- LinkedIn: [My LinkedIn](https://www.linkedin.com/in/samarth-tikotkar-7532b0328/)

- Twitter: [My Twitter](https://x.com/someear9h)

