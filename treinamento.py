import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.linear_model import LogisticRegression

test = pd.read_csv("static/test.csv")
train = pd.read_csv("static/train.csv")

dataframes = [train, test]
for dataset in dataframes:

    dataset['Sex'] = dataset['Sex'].replace('female', 1)
    dataset['Sex'] = dataset['Sex'].replace('male', 0)

    dataset['Embarked'] = dataset['Embarked'].replace('S', 0)
    dataset['Embarked'] = dataset['Embarked'].replace('C', 1)
    dataset['Embarked'] = dataset['Embarked'].replace('Q', 2)

    dataset['Family'] = dataset['SibSp'] + train['Parch']
    dataset['IsAlone'] = 0
    dataset.loc[dataset['Family'] == 1, 'IsAlone'] = 1

    dataset['Age'] = dataset['Age'].fillna(30)
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)



variaveis = ['Title', 'Sex', "Age", "Pclass", "Embarked", "IsAlone", "Fare"]

X = train[variaveis]
y = train["Survived"]

X = X.fillna(-1)

resultados = []
kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=10)

for linhas_treino, linhas_valid in kf.split(X):
    print("Treino:", linhas_treino.shape[0])
    print("Valid:", linhas_valid.shape[0])

    X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]
    y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]

    modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    # modelo = LogisticRegression(max_iter=200)
    modelo.fit(X_treino, y_treino)

    p = modelo.predict(X_valid)
    acc = np.mean(y_valid == p)
    resultados.append(acc)
    print("Acc:", acc)

print(np.mean(resultados))

