import pandas as pd
import numpy as nu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error


class LinearRegression:

    def __init__(self):
        pass

    def read_csv_file(self):
        insurance = pd.read_csv("src/data/new_insurance_data.csv")
        return insurance
    
    def shape_inspection(self, insurance):
        shape = insurance.shape
        print("shape : ", shape)

    def column_info(self, insurance):
        info = insurance.info()
        print("Column Info : ", info)

    def column_list(self, insurance):
        columns = list(insurance.columns)
        return columns
        

    #data preprocessing to fill missing values
    def pre_processing(self, insurance, columns):
        insurance.smoker.mode()
        for cname in columns:
            if cname == 'children':
                insurance[cname] = insurance[cname].fillna(insurance[cname].mode()[0])
            elif insurance[cname].dtype == 'object':
                insurance[cname] = insurance[cname].fillna(insurance[cname].mode()[0])
            else:
                insurance[cname] = insurance[cname].fillna(insurance[cname].mean())

    #identify outlier data points
    def detect_outlier(self, insurance, columns):
        for cname in columns:
            if(insurance[cname].dtype == 'float64'):
                plt.boxplot(insurance[cname])
                plt.xlabel(cname)
                plt.show()

    #outlier data filter
    def filter_outlier(self, insurance):
        columns = ['bmi','past_consultations','Hospital_expenditure','Anual_Salary']

        for column in columns:
            Q1 = insurance[column].quantile(0.25)
            Q3 = insurance[column].quantile(0.75)
            IQR = Q3 - Q1
            lowerFence = Q1 - (1.5*IQR)
            upperFence = Q3 + (1.5*IQR)
            insurance = insurance[(insurance[column] >= lowerFence) & (insurance[column] <= upperFence)]
        return insurance

    # Shape of the data after outlier
    def shape_inspection_after_filter_outlier(self, insurance):
        shape = insurance.shape
        print(" shape after outlier : ", shape)

    #andas to compute the correlation matrix between numerical columns
    def data_correlation(self, insurance):
        numCols = insurance.select_dtypes(include = ['number'])
        correlateMatrix = numCols.corr()
        plt.figure(figsize=[10, 5]) 
        sns.heatmap(correlateMatrix,annot=True,cmap="bwr")
        plt.show()

    ###**Preprocessing:** Encoding, Standardization, Normalization, Feature selection/ Feature elemination, Feature Extraction, Log Transformation
    #Documentation: https://scikit-learn.org/stable/api/sklearn.preprocessing.html
    
    def data_encoding(self, insurance):
        category_columns = insurance.select_dtypes(include = ['object'])
        le = LabelEncoder()
        for column in category_columns:
            insurance[column] = le.fit_transform(insurance[column])
        return insurance
    
    def linearRegression(self, insurance):
        x = insurance.drop('charges', axis=1) #independent column
        y = insurance['charges'] # Dependent column/Target

        x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=0)
        model = SklearnLinearRegression()
        model.fit(x_train, y_train)

        print("m slope : ", model.coef_)
        print("c : ", model.intercept_)
        y_predict = model.predict(x_test)
        rSquaredScore = r2_score(y_test, y_predict)
        print("rSquaredScore : ", rSquaredScore)
        rSquaredEorrs = root_mean_squared_error(y_test, y_predict)
        print("rSquaredEorrs : ", rSquaredEorrs)

        #BestFit LinearRegression
        sns.regplot(x=y_predict, y=y_test)
        plt.xlabel('Prediction')
        plt.title('Best Fit Line')
        plt.ylabel('Acutal Value')
        plt.show()

    


if __name__ == "__main__":

    lrModel = LinearRegression()
    insurance = lrModel.read_csv_file()
    lrModel.shape_inspection(insurance)
    lrModel.column_info(insurance)
    columnList = lrModel.column_list(insurance)
    lrModel.pre_processing(insurance, columnList)
    lrModel.detect_outlier(insurance, columnList)
    insurance = lrModel.filter_outlier(insurance)
    lrModel.shape_inspection_after_filter_outlier(insurance)
    lrModel.data_correlation(insurance)
    lrModel.data_encoding(insurance)
    lrModel.linearRegression(insurance)
