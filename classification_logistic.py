from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.metrics import accuracy_score 


def logistic_model(degrees,x_train,y_train,x_val,y_val):
    polys_list=[]
    scaler_list=[]
    model_list=[]
    acc_train_list=[]
    acc_val_list=[]
    for degree in degrees:
        print(degree)
        #define polynomial and scalar object
        poly=PolynomialFeatures(degree,include_bias=False)
        scaler=StandardScaler()

        #convert feature into polynomial and scale it for train and val data
        x_mapped=poly.fit_transform(x_train)
        x_mapped_scaled=scaler.fit_transform(x_mapped)

        x_val_mapped=poly.transform(x_val)
        x_val_mapped_scaled=scaler.transform(x_val_mapped)

        #add polys and scaler object in list
        polys_list.append(poly)
        scaler_list.append(scaler)


        #now train model here 
        model=LogisticRegression()
        model.fit(x_mapped_scaled,y_train)
        model_list.append(model)


        #now predict outcome through model
        y_train_predict=model.predict(x_mapped_scaled)
        y_val_predict=model.predict(x_val_mapped_scaled)

        #now predict accuracy
        acc_train=accuracy_score(y_train,y_train_predict)
        acc_val=accuracy_score(y_val,y_val_predict)

        #add accuracy in list
        acc_train_list.append(acc_train)
        acc_val_list.append(acc_val)

    return acc_train_list,acc_val_list,model_list,polys_list,scaler_list
        






def optimal_intercept(C_list,x_train,y_train,x_val,y_val,optimal_degree):

    C_value_list=[]
    acc_train_list=[]
    acc_val_list=[]
    
    #define polynomila feature ans scaler object
    poly=PolynomialFeatures(optimal_degree,include_bias=False)
    scaler=StandardScaler()
    
    #mapped and scale dall data
    x_train_mapped=poly.fit_transform(x_train)
    x_val_mapped=poly.transform(x_val)

    #scale data
    x_train_mapped_scaled=scaler.fit_transform(x_train_mapped)
    x_val_mapped_scaled=scaler.transform(x_val_mapped)
    
    
    for C in C_list:

        #train model with optimal degree
        model=LogisticRegression(C=C,random_state=55)
        model.fit(x_train_mapped_scaled,y_train)
        C_value_list.append(C)

        #predict train and val data
        y_train_predict=model.predict(x_train_mapped_scaled)
        y_val_predict=model.predict(x_val_mapped_scaled)

        #accuracy of train and val data
        acc_train=accuracy_score(y_train,y_train_predict)
        acc_val=accuracy_score(y_val,y_val_predict)

        #add in to list
        acc_train_list.append(acc_train)
        acc_val_list.append(acc_val)

    return acc_train_list,acc_val_list

        
        
    

        
        