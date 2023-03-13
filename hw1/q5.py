import pickle
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Q5.1
# DiabetesPedigreeFuncition has the largest impact on the prediction
print("Q5.1 -------------------------")
train_df = pd.read_pickle('./HW1_Q5/train_1.pkl')
train_x = train_df.loc[:, train_df.columns != 'Outcome']
train_y = train_df.loc[:,'Outcome']
logisticRegr = LogisticRegression(max_iter=100000)
logisticRegr.fit(train_x, train_y)

test_df = pd.read_pickle('./HW1_Q5/test_1.pkl')
test_x = test_df.loc[:, test_df.columns != 'Outcome']
test_y = test_df.loc[:,'Outcome']
predictions = logisticRegr.predict(test_x)
print(accuracy_score(test_y, predictions))
pred_df = pd.DataFrame(predictions, columns=['Prediction'])
pred_df_with_context = pd.merge(test_x, pred_df, left_index=True, right_index=True)
print(pred_df_with_context)
weights = pd.DataFrame(logisticRegr.coef_, columns=train_x.columns)
print(weights)

print("Q5.2-------------------------")
train_df2 = pd.read_pickle('./HW1_Q5/train_2.pkl')
train_x2 = train_df2.loc[:, train_df2.columns != 'Outcome']
train_y2 = train_df2.loc[:,'Outcome']
logisticRegr2 = LogisticRegression(max_iter=100000)
logisticRegr2.fit(train_x2, train_y2)

test_df2 = pd.read_pickle('./HW1_Q5/test_2.pkl')
test_x2 = test_df2.loc[:, test_df2.columns != 'Outcome']
test_y2 = test_df2.loc[:,'Outcome']
predictions2 = logisticRegr2.predict(test_x2)
print(accuracy_score(test_y2, predictions2))
pred_df2 = pd.DataFrame(predictions2, columns=['Prediction'])
pred_df_with_context2 = pd.merge(test_x2, pred_df2, left_index=True, right_index=True)
print(pred_df_with_context2)
weights2 = pd.DataFrame(logisticRegr2.coef_, columns=train_x2.columns)
print(weights2)

# The new feature's weight is quite low at .016896, same as glucose's. This makes sense
# because the new feature is identical to Glucose.
# The values are the same because the new feature is identical to Glucose.

print("Q5.3-------------------------")
train_df3 = pd.read_pickle('./HW1_Q5/train_3.pkl')
train_x3 = train_df3.loc[:, train_df3.columns != 'Outcome']
train_y3 = train_df3.loc[:,'Outcome']
logisticRegr3 = LogisticRegression(max_iter=100000)
logisticRegr3.fit(train_x3, train_y3)

test_df3 = pd.read_pickle('./HW1_Q5/test_3.pkl')
test_x3 = test_df3.loc[:, test_df3.columns != 'Outcome']
test_y3 = test_df3.loc[:,'Outcome']
predictions3 = logisticRegr3.predict(test_x3)
print(accuracy_score(test_y3, predictions3))
pred_df3 = pd.DataFrame(predictions3, columns=['Prediction'])
pred_df_with_context3 = pd.merge(test_x3, pred_df3, left_index=True, right_index=True)
print(pred_df_with_context3)
weights3 = pd.DataFrame(logisticRegr3.coef_, columns=train_x3.columns)
print(weights3)