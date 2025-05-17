import numpy as np
import pandas as pd
import warnings
import plotly.express as px
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as logisticModel
from sklearn.metrics import *

class LogisticRegression:

    def __init__(self):
        pass

    def read_csv(self):
        warnings.filterwarnings('ignore') #control how warning messages are handeled
        data = pd.read_csv("src/data/bank-additional-full.csv", sep=';')
        return data
    
    def shape_inspection(self, bankData):
        shape = bankData.shape
        print(f'The rows are: {shape[0]} and columns are : {shape[1]}')

    def data_cleaning(self, data):
        #Data analyzing
        print("check all columns ", data.describe(include='all').T) #stats includes object and numeric values
        print("check numeric columns ", data.describe().T) # numeric values
        print("check object columns ", data.describe(include='O').T) #stats of object columns

        #NaN check and update for numric data
        print("Column wise NaN value ", data.isnull().sum()) #columns wise NaN check
        print("Total NaN values ", data.isnull().sum().sum()) #Total NaN values
        data.dropna(inplace=True) #It removes data that contains missing values(represented as NaN - Not a number)

        #duplicate check and delete duplicate data
        print("Duplicate check ", data.duplicated().sum()) #duplicate check
        data.drop_duplicates(inplace=True) #duplicates removed
        return data
    
    def shape_inspection_after_data_cleaning(self, bankData):
        shape = bankData.shape
        print(f'After Data cleaning the rows are: {shape[0]} and columns are : {shape[1]}')

    def draw_boxplot(self, data):
        colList = data.columns
        for column in colList:
            if data[column].dtype != 'object':
                fig = px.box(data[column], title = f'Box plot of {column} column')
                fig.update_layout(width=500, height=500)
                fig.show()

    def draw_outlier_for_requried_columns(self, data):
        outlierCols = ['age', 'duration', 'campaign', 'cons.conf.idx']
        #Q1=25% Q2=50% Q3=75% IQR - Interquartile Range. Data arrange in ascending order.
        for col in outlierCols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3- Q1
            #1.5 is commonly used convention in statistics for outlier detection using the IQR method
            LF = Q1 - (1.5*IQR)
            UF = Q3 + (1.5*IQR)
            data = data[(data[col] >= LF) & (data[col] <= UF)]
        return data
    
    def shape_inspection_after_outlier(self, data):
        shape = data.shape
        print(f'After outlier the rows are: {shape[0]} and columns are : {shape[1]}')

    def correlation_data(self, data):
        corrMatrics = data.select_dtypes(include=['number']).corr()
        plot.figure(figsize=(18,8))
        sns.heatmap(corrMatrics, annot=True, cmap='coolwarm')
        plot.show()    

    def encoding_data(self, data):
        categoryColmns = []
        for column in data.columns:
            if data[column].dtype == 'object':
                categoryColmns.append(column)
        
        encoder = LabelEncoder()
        labelMap = {}
        for column in categoryColmns:
            data[column] = encoder.fit_transform(data[column])
            labelMap[column] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            print(f'{column} : {labelMap[column]}')

        plot.figure(figsize=(18, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        plot.show()

    def get_vif_data(self, data):
        columns = data.columns
        tempData = data.drop(columns=['y'])
        vifData = pd.DataFrame()
        vifData['Features'] = tempData.columns
        vifData['VIF_Score'] = [vif(tempData.values, i) for i in range(len(tempData.columns))]
        maxVifScore = vifData.loc[vifData['VIF_Score'] == vifData['VIF_Score'].max()]
        a = maxVifScore['Features']
        maxValue = pd.to_numeric(maxVifScore['VIF_Score'].max(), errors='coerce')
        if(maxValue >= 5):
            data = data.drop(a, axis=1)
            data = self.get_vif_data(data)
        return data
    
    def sigmoid(self, data):
        result = 1/(1+(np.exp(-data)))
        print(f'result : {result}')
        return result
    
    def logisticRegressionModel(self, data):
        #x_train - Learing Quesions y_train - Answers to learn my model x_test - exam_questions y_test - Key
        x_train, x_test, y_train,y_test = train_test_split(data, data['y'], test_size=0.3, random_state=42)
        model = logisticModel()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)

        #evolution: confusion metrics, classification report, accuracy score
        accuracyScore = accuracy_score(y_test, y_predict)
        print(f'accuracyScore : {accuracyScore}')
        confusionMetrics = confusion_matrix(y_test, y_predict)
        print(f'confusionMetrics : {confusionMetrics}')
        classificationReport = classification_report(y_test, y_predict)
        print(f'classificationReport : {classificationReport}')

        #classification report describe
        #precision predicts positive values
        #True Positive/(True Positive + False Positive)
        #Recall(Sensitivity)
        #True Positive/(True Positive + False Negetive)
        #F1 Score - A balanced between precision and recall
        # 2*(precision * recall)/(precision+recall)
        #support - no of actual values occuring each class in dataset

        y_score = model.predict_proba(x_test)[:, 1]
        sortInd = np.argsort(y_score)
        sortLabel = y_test.iloc[sortInd]
        sortScore = y_test[sortInd]

        x_values = np.linspace(-10, 10, 100)
        y_sigmoid = logisticRegression.sigmoid(x_values)
        plot.figure(figsize=(7,5))
        plot.plot(x_values, y_sigmoid, color='red')
        plot.scatter(sortScore, sortLabel)
        plot.show()
        

if __name__ == "__main__":
    logisticRegression = LogisticRegression()
    data = logisticRegression.read_csv()
    logisticRegression.shape_inspection(data)
    data = logisticRegression.data_cleaning(data)
    logisticRegression.shape_inspection_after_data_cleaning(data)
    logisticRegression.draw_boxplot(data)
    data = logisticRegression.draw_outlier_for_requried_columns(data)
    logisticRegression.shape_inspection_after_outlier(data)
    logisticRegression.correlation_data(data)
    logisticRegression.encoding_data(data)
    data = logisticRegression.get_vif_data(data)
    print("vifData : ", data)
    logisticRegression.logisticRegressionModel(data)