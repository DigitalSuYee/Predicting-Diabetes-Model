import requests

url = 'http://localhost:9696/predict'

user_id = 'xyz-123'

user = {
'gender': "Female",
'age': 40.0,
'hypertension': 0,
'heart_disease': 1,
'smoking_history': "never",
'bmi': 27.32,
'hba1c_level': 5.0,
'blood_glucose_level': 145,

}

response = requests.post(url, json=user).json()
print(response)

if response['diabetes'] == True:
    print('The user has %s' % user_id)
else:
    print('The user has no %s' % user_id)