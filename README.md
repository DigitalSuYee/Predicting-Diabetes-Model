# Predicting-Diabetes-Model
Diabetes is a prevalent health concern affecting individuals worldwide. Early identification of individuals at risk is crucial for implementing preventive measures and promoting better health outcomes. This project aims to develop a predictive model for diabetes risk assessment using a dataset containing relevant health parameters.

# Problem Statement
Many people around the world have diabetes, and we need a way to find out who might get it early on. Right now, there aren't easy tools for regular people or doctors to check someone's risk of getting diabetes using important health information. So, we want to make a simple and reliable computer program to do just that – help people and doctors know the risk of diabetes early, so they can take action and plan personalized ways to stay healthy.

# Dataset
- gender: Gender of the individual (categorical: Female/Male).
- age: Age of the individual.
- hypertension: Whether the individual has hypertension (binary: 0 for No, 1 for Yes).
- heart_disease: Whether the individual has heart disease (binary: 0 for No, 1 for Yes).
- smoking_history: Smoking history of the individual (categorical: never/current/No Info).
- bmi: Body Mass Index (numeric).
- hba1c_level: HbA1c level, a measure of average blood glucose over a few months (numeric).
- blood_glucose_level: Fasting blood glucose level (numeric).
- diabetes: Target variable indicating the presence of diabetes (binary: 0 for No, 1 for Yes).

# EDA
Explored dataset structure and types, summarized numerical and categorical features, assessed missing data, analyzed relationships between features, and examined class distribution for insights into potential imbalances.

# Model Training

## Model Training with Decision Tree:
A Decision Tree model was trained to predict diabetes based on features like age, BMI, and medical history. The scikit-learn library facilitated both model training and hyperparameter tuning through a grid search. Hyperparameters, including tree depth and minimum samples per leaf, were optimized to enhance the model's predictive performance. 

## Model Training with Logistic Regression:
Employing the scikit-learn toolkit, we trained a Logistic Regression model on a dataset comprising features like age, BMI, and medical history for diabetes prediction. Parameter tuning involved exploring values of C (inverse of regularization strength) including 1, 0.1, 0.01, and 10. The optimal parameter was identified as C = 10, and this tuned model, showcasing superior performance, was selected for deployment. AUC-ROC curve analysis was instrumental in making this determination.


# Environment Management

I used pipenv for the virtual environment. 

```
pip install pipenv
```
To replicate the environment, on your command line, use

```
pipenv install numpy scikit-learn==0.24.2 flask waitress
```

For the required versions for libraries, use requirements.txt

```
$pip install -r requirements.txt 
```

# Containerization

Use this command for container building.

```
$Docker build -t 'container_name'
```

Use this container running

```
$Docker run -it --rm -p 9696:9696 'container_name'
```

