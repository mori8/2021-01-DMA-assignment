import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score #score evaluation

train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')
passenger_id = test_data['PassengerId']

# print(train_data.isnull().sum())
# print(test_data.isnull().sum())


def titanic_model():
    # 1. 데이터 전처리
    data = [train_data, test_data]
    for d in data:
        d['Initial'] = 0
        for i in d:
            d['Initial'] = d.Name.str.extract('([A-Za-z]+)\.')  # lets extract the Salutations
        d['Initial'].replace(
            ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Dona', 'Capt', 'Sir', 'Don'],
            ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr'], inplace=True)

        # print(d.groupby('Initial')['Age'].mean())

        d.loc[(d.Age.isnull()) & (d.Initial == 'Mr'), 'Age'] = 32
        d.loc[(d.Age.isnull()) & (d.Initial == 'Mrs'), 'Age'] = 39
        d.loc[(d.Age.isnull()) & (d.Initial == 'Master'), 'Age'] = 7
        d.loc[(d.Age.isnull()) & (d.Initial == 'Miss'), 'Age'] = 22
        d.loc[(d.Age.isnull()) & (d.Initial == 'Other'), 'Age'] = 42

        d['Age_band'] = 0
        d.loc[d['Age'] <= 16, 'Age_band'] = 0
        d.loc[(d['Age'] > 16) & (d['Age'] <= 32), 'Age_band'] = 1
        d.loc[(d['Age'] > 32) & (d['Age'] <= 48), 'Age_band'] = 2
        d.loc[(d['Age'] > 48) & (d['Age'] <= 64), 'Age_band'] = 3
        d.loc[d['Age'] > 64, 'Age_band'] = 4

        d['Embarked'].fillna('S', inplace=True)

        d['Family_Size'] = 0
        d['Family_Size'] = d['Parch'] + d['SibSp']  # family size
        d['Alone'] = 0
        d.loc[d.Family_Size == 0, 'Alone'] = 1  # Alone

        d['Fare_cat'] = 0
        d.loc[d['Fare'] <= 7.91, 'Fare_cat'] = 0
        d.loc[(d['Fare'] > 7.91) & (d['Fare'] <= 14.454), 'Fare_cat'] = 1
        d.loc[(d['Fare'] > 14.454) & (d['Fare'] <= 31), 'Fare_cat'] = 2
        d.loc[(d['Fare'] > 31) & (d['Fare'] <= 513), 'Fare_cat'] = 3

        d['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
        d['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
        d['Initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other'], [0, 1, 2, 3, 4], inplace=True)

        d.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin', 'PassengerId'], axis=1, inplace=True)


    # print(train_data.isnull().sum())
    # print(test_data.isnull().sum())

    # 2. 모델 학습과 예측
    X = train_data[train_data.columns[1:]]
    Y = train_data['Survived']

    ada = AdaBoostClassifier(n_estimators=250, learning_rate=0.05)
    ada.fit(X, Y)
    predictions = ada.predict(test_data)

    result = cross_val_score(ada, X, Y, cv=10, scoring='accuracy')
    # print('The cross validated score for AdaBoost is:',result.mean())

    # 3. 예측 결과를 담은 submission.csv 파일 생성
    output = pd.DataFrame({'PassengerId': passenger_id, 'Survived': predictions})
    output.to_csv('titanic/submission.csv', index=False)

    print("'submission.csv' is successfully created.")
