from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data.drop(columns=['ID','Candidate','Constituency ∇'], inplace=True)
test_data.drop(columns=['ID','Candidate','Constituency ∇'], inplace=True)

def convert_to_lakhs(value):
    value = str(value)
    if value == "0":
        return 0
    elif 'Crore+' in value:
        number, unit = value.split()
        number = float(number.replace(",", ""))
        return int(number * 100)
    elif 'Lac+' in value:
        number, unit = value.split()
        number = float(number.replace(",", ""))
        return int(number)
    elif 'Thou+' in value:
        number, unit = value.split()
        number = float(number.replace(",", ""))
        return int(number / 100)
    elif 'Hund+' in value:
        number, unit = value.split()
        number = float(number.replace(",", ""))
        return int(number / 1000)
    else:
        return int(value)

train_data["Total Assets"] = train_data["Total Assets"].apply(convert_to_lakhs)
train_data["Liabilities"] = train_data["Liabilities"].apply(convert_to_lakhs)
test_data["Total Assets"] = test_data["Total Assets"].apply(convert_to_lakhs)
test_data["Liabilities"] = test_data["Liabilities"].apply(convert_to_lakhs)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
train_data['Education'] = label_encoder.fit_transform(train_data['Education'])

train_data = pd.get_dummies(train_data, columns=['Party','state'])
test_data = pd.get_dummies(test_data, columns=['Party','state'])

X = train_data.drop(columns=['Education'])
y = train_data['Education']

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, test_size=0.2, random_state=42)

model = BernoulliNB(alpha=0.5, binarize=0.1, fit_prior=True, class_prior=None)

model.fit(X_train_split, y_train_split)
y_pred = model.predict(test_data)
y_pred = label_encoder.inverse_transform(y_pred)

df_pred = pd.DataFrame(y_pred, columns=["Education"])
df_pred.to_csv('submission.csv', index_label='ID')
