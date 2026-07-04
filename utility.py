from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


#FIND BEST POLYNOIMIAL DEGREE SUIT MODEL ON DATASET
def find_best_model(x_train,y_train,x_cv,y_cv,degrees,baseline):

    #list of all sclaer,polynomial ,mses list and cross validation error list
    polys=[]
    scalers=[]
    models=[]
    train_mses=[]
    cv_mses=[]
    for degree in degrees:
        poly=PolynomialFeatures(degree,include_bias=False)
        x_mapped=poly.fit_transform(x_train)
        polys.append(poly)

        #scale train data
        scaler=StandardScaler()
        x_mapped_scaled=scaler.fit_transform(x_mapped)
        scalers.append(scaler)

        #fit model 
        model=LinearRegression()
        model.fit(x_mapped_scaled,y_train)
        models.append(model)

        #predict train data on trained model
        y_hat=model.predict(x_mapped_scaled)
        train_mse=mean_squared_error(y_train,y_hat)/2
        train_mses.append(train_mse)

        x_cv_mapped=poly.transform(x_cv)
        x_cv_mapped_scaled=scaler.transform(x_cv_mapped)

        #predict oncross validation data
        y_cap=model.predict(x_cv_mapped_scaled)
        cv_mse=mean_squared_error(y_cv,y_cap)/2
        cv_mses.append(cv_mse)

    print(f"{train_mses},{cv_mses}") 

    #plot graph
    plt.plot(degrees,train_mses,color='r',marker='o')
    plt.plot(degrees,cv_mses,color='b',marker='o')
    #plt.plot(degrees,np.repeat(baseline,len(degrees)),linestyle='--',label='baseline')
    plt.xlabel('degree')
    plt.ylabel('mean squared error')
    plt.show()
    index_cvmse=np.argmin(cv_mses)+1
    return index_cvmse,models[index_cvmse-1],polys[index_cvmse-1],scalers[index_cvmse-1]


#FIND BEST REGULARIZED PARAMETER

def find_regularized_parameter(x_train,y_train,x_cv,y_cv,degree,regularized,baseline):
    polys=[]
    scalers=[]
    train_mses=[]
    cv_mses=[]
    models=[]
    #run loop to find best Ridge model fitted with best regularized parameter
    for reg in regularized:
        poly=PolynomialFeatures(degree,include_bias=False)
        x_train_mapped=poly.fit_transform(x_train)
        polys.append(poly)

        #scale train data
        scaler=StandardScaler()
        x_train_mapped_scaled=scaler.fit_transform(x_train_mapped)
        scalers.append(scaler)

        #scale ridge model
        model=Ridge(alpha=reg)
        model.fit(x_train_mapped_scaled,y_train)
        models.append(model)

        #predict on train data 
        y_hat=model.predict(x_train_mapped_scaled)
        train_mse=mean_squared_error(y_train,y_hat)/2
        train_mses.append(train_mse)

        x_cv_mapped=poly.transform(x_cv)
        x_cv_mapped_scaled=scaler.transform(x_cv_mapped)

        #predict on cross validation data
        y_cap=model.predict(x_cv_mapped_scaled)
        cv_mse=mean_squared_error(y_cv,y_cap)/2
        cv_mses.append(cv_mse)
        

    #plot graph
    plt.plot(regularized,train_mses,color='r',marker='o')
    plt.plot(regularized,cv_mses,color='b',marker='o')
    plt.xlabel('lambda')
    plt.ylabel('mean squared error')
    print(f"train_mses={train_mses}")
    print(f"cv_mses={cv_mses}")
    
    #find index with minimum mean square error of cross vlaidation data
    index_cvmse=np.argmin(cv_mses)+1

    

    return index_cvmse,models[index_cvmse-1],polys[index_cvmse-1],scalers[index_cvmse-1]
    

            
            

            
            
        
        