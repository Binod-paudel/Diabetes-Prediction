from django.shortcuts import render
from django.http import HttpRequest
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from .forms import BMICalculatorForm


data = pd.read_csv('static/diabetes_dataset.csv')

X=data.drop(columns='Outcome',axis=1)
Y=data['Outcome']

scaler=StandardScaler()
scaler.fit(X)

standardized_data=scaler.transform(X)

X=standardized_data
Y=data['Outcome']
#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)







SVMmodel = joblib.load('static/svc')
lrmodel = joblib.load('static/LRmodel')


scaler = joblib.load('static/scaler.pkl')

def index(request):
    return render(request, 'index.html')

def contact(request):
    return render(request, 'contact.html')
def login(request):
    return render(request, 'login.html')

def registration(request):
    return render(request, 'registration.html')

def faq(request):
    return render(request, 'faq.html')

def prediction(request):
    if request.method == "POST":
        pregnancies = int(request.POST.get('pregnancies'))
        glucose = float(request.POST.get('glucose'))
        bloodPressure = float(request.POST.get('bloodPressure'))
        skinThickness = float(request.POST.get('skinThickness'))
        insulin = float(request.POST.get('insulin'))
        bmi = float(request.POST.get('bmi'))
        diabetesPedigreeFunction = float(request.POST.get('diabetesPedigreeFunction'))
        age = int(request.POST.get('age'))
        model_select = request.POST.get('model_select')

       
        input_data = np.array([pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]).reshape(1, -1)

        
        std_data = scaler.transform(input_data)
        print(std_data)
  
        if model_select == 'SVM':
            prediction = SVMmodel.predict(std_data)
            
            if prediction[0] == 0:# 
                result = 'The person is not diabetic'
            else:
                result = 'The person is diabetic'
        elif model_select == 'LR':
            prediction = lrmodel.predict(std_data)
            X_train_prediction = lrmodel.predict(X_train)
            accuracy = accuracy_score(X_train_prediction, Y_train)
            if prediction[0] == 0:
                result = 'The person is not diabetic'
            else:
                result = 'The person is diabetic'
        elif model_select == 'Both':
             prediction = SVMmodel.predict(std_data)
            # accuracy score on the training data
             X_train_prediction = SVMmodel.predict(X_train)
             accuracy = accuracy_score(X_train_prediction, Y_train)
            
            
             if prediction[0] == 0:# Assuming prediction is a single value
                result1 = 'From SVM: The person is not diabetic'
             else:
                result1 = 'From SVM: The person is diabetic'

             prediction = lrmodel.predict(std_data)
             X_train_prediction = lrmodel.predict(X_train)
             accuracy = accuracy_score(X_train_prediction, Y_train)
             if prediction[0] == 0:# Assuming prediction is a single value
                result2 = 'From LR: The person is not diabetic'
             else:
                result2 = 'From LR:The person is diabetic'
             result=result1+""+result2
        output = {
            'pred': result 
        }

        return render(request, 'prediction.html', output)
    else:
        return render(request, 'prediction.html')
    
def about(request):
    return render(request, 'about.html')
def accuracy(request):
    
    X_test_prediction_svm = SVMmodel.predict(X_test)
    precision_svm = precision_score(Y_test, X_test_prediction_svm)
    recall_svm = recall_score(Y_test, X_test_prediction_svm)
    f1_svm = f1_score(Y_test, X_test_prediction_svm)
    accuracy_svm = accuracy_score(X_test_prediction_svm, Y_test)
    
    X_test_prediction_lr = lrmodel.predict(X_test)
    precision_lr = precision_score(Y_test, X_test_prediction_lr)
    recall_lr = recall_score(Y_test, X_test_prediction_lr)
    f1_lr = f1_score(Y_test, X_test_prediction_lr)
    accuracy_lr = accuracy_score(X_test_prediction_lr, Y_test)
     

    precision_svm = "{:.2f}%".format(precision_svm * 100)
    recall_svm = "{:.2f}%".format(recall_svm * 100)
    f1_svm = "{:.2f}%".format(f1_svm * 100)
    accuracy_svm = "{:.2f}%".format(accuracy_svm * 100)
    
    precision_lr = "{:.2f}%".format(precision_lr * 100)
    recall_lr = "{:.2f}%".format(recall_lr * 100)
    f1_lr = "{:.2f}%".format(f1_lr * 100)
    accuracy_lr = "{:.2f}%".format(accuracy_lr * 100)
    
    
    labels = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    svm_scores = [precision_svm, recall_svm, f1_svm, accuracy_svm]
    lr_scores = [precision_lr, recall_lr, f1_lr, accuracy_lr]
    
    # Plotting the bar graph
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x, svm_scores, width, label='SVM')
    ax.bar([i + width for i in x], lr_scores, width, label='LR')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of SVM and LR Models')
    ax.legend()
    

    
    output1 = {
        'pre': precision_svm,
        'rec': recall_svm,
        'fscor': f1_svm,
        'acc': accuracy_svm,
        'pre1': precision_lr,
        'rec1': recall_lr,
        'fscor1': f1_lr,
        'acc1': accuracy_lr
    }
    return render(request, 'accuracy.html', output1)




def calculate_bmi(weight_kg, height_m):
    bmi = weight_kg / (height_m ** 2)
    return bmi

def interpret_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def prediction_view(request):
    if request.method == 'POST':
        form = BMICalculatorForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            weight = data['weight']
            height = data['height']
            bmi = calculate_bmi(weight, height)
            interpretation = interpret_bmi(bmi)
            return render(request, 'prediction.html', {'pred': interpretation})
    else:
        form = BMICalculatorForm()
    
    return render(request, 'prediction.html', {'form': form})


