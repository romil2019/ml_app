import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer







# preparing dataset object for House prediction
dataset={"Housing Dataset":{"file":"Housing.csv",
                         "type":"Regression",
                         "model":["XGBoost",
                                 "Random Forest Regressor",
                                  "Ridge"]
                         
},
    "Medical insurnace":{
         "file":"insurance.csv",
        "type":"Regression",
        "model":["Random Forest Regressor","XGBoost","Ridge"]
},
"Heart Diagnisis":{"file":"heart.csv",
                   "type":"Classification",
                   "model":["XGBoost",
                                 "Random Forest",
                                  "Logistic Regression"]

},
       
"Spam Dataset":{"file":"spam.csv",
                "type":"Classification",
                "model":["XGBoost",
                         "Random Forest",
                         "Logistic Regression"]
    
}
        }



#define functionto  take feauture input from user as  paramaeter by using slider button
#for Hosuing predcition

def add_input_slider(info):
              param=dict()
              data=pd.read_csv(info["file"])
    
              if info["file"]=="Housing.csv":
                  X=data.drop(columns=["price","date","id","condition"],axis=1)
                   
               
              
              
    

        
              bedroom=st.sidebar.slider("bedroom",X["bedrooms"].min(),X["bedrooms"].max())
              bathroom=st.sidebar.slider("bathroom",X["bathrooms"].min(),X["bathrooms"].max())
              sqft_living=st.sidebar.slider("sqft_living",X["sqft_living"].min(),X["sqft_living"].max())
              sqft_lot=st.sidebar.slider("sqft lot",X["sqft_lot"].min(),X["sqft_lot"].max())
              floors=st.sidebar.slider("floors",X["floors"].min(),X["floors"].max())
              waterfront=st.sidebar.slider("waterfront",X["waterfront"].min(),X["waterfront"].max())
              view=st.sidebar.slider("view",X["view"].min(),X["view"].max())
              grade=st.sidebar.slider("grade",X["grade"].min(),X["grade"].max())
              sqft_above=st.sidebar.slider("sqft_above",X["sqft_above"].min(),X["sqft_above"].max())
              sqft_basement=st.sidebar.slider("sqft_basement",X["sqft_basement"].min(),
                                        X["sqft_basement"].max())
              yr_built=st.sidebar.slider("yr_built",X["yr_built"].min(),X["yr_built"].max())
                  
              yr_renovated=st.sidebar.slider("yr_renovated",X["yr_renovated"].min(),
                                                      X["yr_renovated"].max())
              zipcode=st.sidebar.slider("zipcode",X["zipcode"].min(),X["zipcode"].max())
              lat=st.sidebar.slider("lat",X["lat"].min(),X["lat"].max())
              long=st.sidebar.slider("long",X["long"].min(),X["long"].max())
              sqft_living15=st.sidebar.slider("sqft_living15",X["sqft_living15"].min(),
                                        X["sqft_living15"].max())
              sqft_lot15=st.sidebar.slider("sqft_lot15",X["sqft_lot15"].min(),X["sqft_lot15"].max())


#    input paramreter attribute defined
              input_data=pd.DataFrame(
                 {"bedrooms":[bedroom],
                 "bathrooms":[bathroom],
                 "sqft_living":[sqft_living],
                 "sqft_lot":[sqft_lot],
                 "floors":[floors],
                 "waterfront":[waterfront],
                 "view":[view],
                 "grade":[grade],
                 "sqft_above":[sqft_above],
                 "sqft_basement":[sqft_basement],
                 "yr_built":[yr_built],
                 "yr_renovated":[yr_renovated],
                 "zipcode":[zipcode],
                 "lat":[lat],
                 "long":[long],
                 "sqft_living15":[sqft_living15],
                 "sqft_lot15":[sqft_lot15]
             
             
             
        
             
             
            }
        )
              return input_data 



#define function for creating Input slider for dataset Insurance
def insurance_input_slider(X):
    
    age=st.sidebar.slider("Age",X["age"].min(),X["age"].max())
    bmi=st.sidebar.slider("bmi",X["bmi"].min(),X["bmi"].max())
    children=st.sidebar.slider("children",X["children"].min(),X["children"].max())
    sex=st.sidebar.selectbox("sex",["male","female"])
    sex_male=1 if sex=="male" else 0
    sex_female=1 if sex=="female" else 0
    smoker=st.sidebar.selectbox("smoker",['Yes','No'])
    smoker_yes=1 if smoker=="Yes" else 0
    smoker_no=1 if smoker=="No" else 0
    region=st.sidebar.selectbox("region",['Northeast','Northwest','Southeast','Southwest'])
    region_northwest = 1 if region == "Northwest" else 0
    region_southwest=1 if region=="Southwest" else 0
    region_southeast=1 if region=="Southeast" else 0
    region_northeast=1 if region=="Northeast" else 0
    
    


#create dataframe for input parameter
    input_data=pd.DataFrame(
      {"age":[age],
       "bmi":[bmi],
       "children":[children],
       "sex_female":[sex_female],
       "sex_male":[sex_male],
       "smoker_no":[smoker_no],
       "smoker_yes":[smoker_yes],
       "region_northeast":[region_northeast],
       "region_northwest":[region_northwest],
       "region_southeast":[region_southeast],
       "region_southwest":[region_southwest],
       
      }
           
       

    )
    return input_data
      
    
    
    


#define function for creating Input slider for dataset Heart 

def add_heart_input_slider(X):
    Age=st.sidebar.slider("Age",X["Age"].min(),X["Age"].max())
    RestingBP=st.sidebar.slider("Resting blood pressure",X["RestingBP"].min(),X["RestingBP"].max())
    Cholesterol=st.sidebar.slider("Cholesterol",X["Cholesterol"].min(),X["Cholesterol"].max())
    MaxHR=st.sidebar.slider("Max Heart Rate",X["MaxHR"].min(),X["MaxHR"].max())
    Oldpeak=st.sidebar.slider("Oldpeak",X["Oldpeak"].min(),X["Oldpeak"].max())
    
    Sex=st.sidebar.selectbox("Sex",['Male','Female'])
    Sex_F=1 if Sex=="Male" else 0
    Sex_M=1 if Sex=="Female" else 0

    FastingBS=st.sidebar.selectbox("Fasting BS",['Yes','No'])
    FastingBS=1 if FastingBS=='Yes' else 0

    
    ChestPainType=st.sidebar.selectbox("Chest painType",['ATA', 'NAP','ASY','TA'])
    ChestPainType_ASY=1 if ChestPainType=="ASY" else 0
    ChestPainType_ATA=1 if ChestPainType=="ATA" else 0
    ChestPainType_NAP=1 if ChestPainType=="NAP" else 0
    ChestPainType_TA=1 if ChestPainType=="TA" else 0

    RestingECG=st.sidebar.selectbox("Resting ECG",['Normal', 'ST', 'LVH'])
    RestingECG_LVH=1 if RestingECG=="LVH" else 0
    RestingECG_Normal=1 if RestingECG=="Normal" else 0
    RestingECG_ST=1 if RestingECG=="ST" else 0

    ExerciseAngina=st.sidebar.selectbox("Exercise Angina",['N', 'Y'])
    ExerciseAngina_N=1 if ExerciseAngina=="N" else 0
    ExerciseAngina_Y=1 if ExerciseAngina=="Y" else 0

    ST_Slope=st.sidebar.selectbox("ST Slope",['Up','Flat', 'Down'])
    ST_Slope_Down=1 if ST_Slope=="Down" else 0 
    ST_Slope_Flat=1 if ST_Slope=="Flat" else 0 
    ST_Slope_Up=1 if ST_Slope=="Up" else 0 
    

    
    #create data frame for input data
    input_data=pd.DataFrame(
{ "Age":[Age],
  "RestingBP":[RestingBP],
  "Cholesterol":[Cholesterol],
  "FastingBS":[FastingBS],
  "MaxHR":[MaxHR],
  "Oldpeak":[Oldpeak],
  "Sex_F":[Sex_F],
  "Sex_M":[Sex_M],
  "ChestPainType_ASY":[ChestPainType_ASY],
  "ChestPainType_ATA":[ChestPainType_ATA],
  "ChestPainType_NAP":[ChestPainType_NAP],
  "ChestPainType_TA":[ChestPainType_TA],
  "RestingECG_LVH":[RestingECG_LVH],
  "RestingECG_Normal":[RestingECG_Normal],
  "RestingECG_ST":[RestingECG_ST],
  "ExerciseAngina_N":[ExerciseAngina_N],
  "ExerciseAngina_Y":[ExerciseAngina_Y],
  "ST_Slope_Down":[ST_Slope_Down],
  "ST_Slope_Flat":[ST_Slope_Flat],
  "ST_Slope_Up":[ST_Slope_Up]
  
  
}
        
    )

    return input_data





#****************************************CHOOSE DATASET*********************************************************************


#choose dataset to work on
data_click=st.sidebar.selectbox("Choose Dataset",list(dataset.keys()))
info=dataset[data_click]     #click data set
data=pd.read_csv(info["file"],encoding="latin-1")
st.write("Problem type is :",info["type"])

      

#make select button to choose which model to use
model=st.sidebar.selectbox("choose model",info["model"])



#********************************************DATASET HOUSING*****************************************************************


#check if choosen datset is Housing 
if(info["file"]=="Housing.csv"):

    #preparing raw data
    y_raw=data["price"]
    X_raw=data.drop(["price","id","date","condition"])
  
  

    #split data in to train,test and validation data
    x_train,x_test,y_train,y_test=train_test_split(X_raw,y_raw,test_size=0.2,random_state=1234)
    
     #Display raw data and train data
    with st.expander('Data') :
            st.write('**Raw data**')
            X_raw
            st.write('**Price**')
            y_raw

            st.write('**X**')
            x_train
            st.write('**y**')
            y_train

     #display data visulaization chart for housing data
    with st.expander('Data Visulaization'):
           st.scatter_chart(data=data, x='sqft_living',y='price')

    #It take input  by the user from slider button 
    input_data=add_input_slider(info)
            
    #display input parameter  for housing data
    with st.expander('Input Parameter'):
           input_data 

#extract pre trained model of Ridge model from .pkl file
    if (model=="Ridge"):
       model=pickle.load(open("house_model.pkl","rb"))
       poly=pickle.load(open("house_polys.pkl","rb"))
       scaler=pickle.load(open("house_scaler.pkl","rb")) 


       #scale and map train data 
       x_train_mapped=poly.transform(x_train)
       x_train_scaled=scaler.transform(x_train_mapped)

       #scale and map test data
       x_test_mapped=poly.transform(x_test)
       x_test_scaled=scaler.transform(x_test_mapped)

#map and scale input given by user
       input_mapped=poly.transform(input_data)
       input_mapped_scaled=scaler.transform(input_mapped)
    
# sextract pre trained model of Random Forest Regresor from .pkl file and scale it
    elif(model=="Random Forest Regressor"):
       model=pickle.load(open("Housing_RandomForestRegressor.pkl","rb"))
        
       scaler=StandardScaler()
       x_train_scaled=scaler.fit_transform(x_train)
       x_test_scaled=scaler.transform(x_test)
       input_mapped_scaled=scaler.transform(input_data)

   #For XGBoost classifier midel is extract  and sclaed for train,test an input data
    elif(model=="XGBoost"):
        model=pickle.load(open("Housing_XGBRegressor.pkl","rb"))

        scaler=StandardScaler()
        x_train_scaled=scaler.fit_transform(x_train)
        x_test_scaled=scaler.transform(x_test)
        input_mapped_scaled=scaler.transform(input_data)









#*************************************************DATASET INSURANCE********************************************************************



#Check if chosen dataset by user  is insurance
    
elif (info["file"]=="insurance.csv"):

  #Preparing Raw data    

    insurance_data=pd.read_csv("insurance.csv")
    y_raw=insurance_data["charges"]
    X_raw=insurance_data.drop(columns=["charges"])
    variable=['sex','smoker','region']
    X=pd.get_dummies(data=X_raw,prefix=variable,columns=variable)

    #scaler=StandardScaler()
    #X_scaled=scaler.fit_transform(X)
    
    x_train,x_test,y_train,y_test=train_test_split(X,y_raw,test_size=0.4,random_state=55)

#Display Raw Data
    with st.expander('Data'):
        
        st.write('**Raw Data**')
        X_raw
        st.write('Charges')
        y_raw
        st.write('**X**')
        X
        st.write('**y**')
        y_train
# Display Graph
    with st.expander('Data Visualization'):
        st.scatter_chart(data=insurance_data,x='age',y='charges')


# Take input from user and Display it
    with st.expander('Input parameter'):

       input_data=insurance_input_slider(X_raw)
       
       
       input_data
       
       

        

    
        
#Check  model type,load it and scale it

    if(model=="Random Forest Regressor"):

        model=pickle.load(open("insurance_RandomForestRegressor.pkl","rb"))


        
        scaler=StandardScaler()
        x_train_scaled=scaler.fit_transform(x_train)
        x_test_scaled=scaler.transform(x_test)
        
        
        input_mapped_scaled=scaler.transform(input_data)
        

    
    
    #For XGBOOST model   
    elif(model=="XGBoost"):

        #load Xgboost model from pkl file
        model=pickle.load(open("Insurance_XGBRegressor.pkl","rb"))

        #scale train,test data and input data
        scaler=StandardScaler()
        x_train_scaled=scaler.fit_transform(x_train)
        x_test_scaled=scaler.transform(x_test)

        input_mapped_scaled=scaler.transform(input_data)
        
    else:


       #For RIDGE model
       model=pickle.load(open("insurance_Ridgemodel.pkl","rb"))

        # define scale and poly object
       polys=pickle.load(open("insurance_Ridgepolys.pkl","rb"))
       scaler=pickle.load(open("insurance_Ridgescaler.pkl","rb"))

       #scale Train,Test dataand input data
       x_train_mapped=polys.transform(x_train)
       x_train_scaled=scaler.transform(x_train_mapped)

       #scale and map test data
       x_test_mapped=polys.transform(x_test)
       x_test_scaled=scaler.transform(x_test_mapped)


      #map and scale input given by user
       input_mapped=polys.transform(input_data)
       input_mapped_scaled=scaler.transform(input_mapped)
        
    


#***********************************************DATASET HEART**********************************************************************

#prepare train,test data for heart dataset
elif (info["file"]=="heart.csv"):

    st.title('❤️ Heart Disease Prediction')
    #prepare raw data
    heart_data=pd.read_csv('heart.csv')
    y_raw=heart_data["HeartDisease"]
    X_raw=heart_data.drop(columns=["HeartDisease"])
    variable=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
    X=pd.get_dummies(data=X_raw,prefix=variable,columns=variable)

    #split data into train ,test
    x_train,x_test,y_train,y_test=train_test_split(X,y_raw,test_size=0.4,random_state=55)

    #display Data
    with st.expander('Data'):
        st.write('**Raw data**')
        X_raw
        st.write('Heart Disease')
        y_raw
        st.write('**X**')
        X
        st.write('y')
        y_raw

    
    #display graph
    with st.expander('Data Visualization'):
        st.scatter_chart(data=heart_data,x='Age',y='Cholesterol')

    with st.expander('Input parameter'):
        input_data=add_heart_input_slider(X)

        input_data

    if(model=="Logistic Regression"):
        model=pickle.load(open("heart_model.pkl","rb"))

        polys=pickle.load(open("heart_poly.pkl","rb"))
        scaler=pickle.load(open("heart_scaler.pkl","rb"))

       #scale Train,Test dataand input data
        x_train_mapped=polys.transform(x_train)
        x_train_scaled=scaler.transform(x_train_mapped)

       #scale and map test data
        x_test_mapped=polys.transform(x_test)
        x_test_scaled=scaler.transform(x_test_mapped)


      #map and scale input given by user
        input_mapped=polys.fit_transform(input_data)
        input_mapped_scaled=scaler.transform(input_mapped)

    elif(model=="Random Forest"):
        
        #unpack the model and load it
        model=pickle.load(open("Heart_Classification_model.pkl","rb"))

        #scale trian,test data and input data
        scaler=StandardScaler()
        x_train_scaled=scaler.fit_transform(x_train)
        x_test_scaled=scaler.transform(x_test)
        input_mapped_scaled=scaler.transform(input_data)
        
    elif(model=="XGBoost"):
        
        #unpack the model and load it
        model=pickle.load(open("heart_XGBClassifier.pkl","rb"))

        #scale trian,test data and input data
        scaler=StandardScaler()
        x_train_scaled=scaler.fit_transform(x_train)
        x_test_scaled=scaler.transform(x_test)
        input_mapped_scaled=scaler.transform(input_data)



#*************************************************DATASET SPAM************************************************************************



#prepare train,test data for spam dataset
elif(info["file"]=="spam.csv"):
    st.title('📩 SMS Spam Detection')
    #Prepare Raw dataset
    spam_dataset=pd.read_csv("spam.csv",encoding="latin-1")
    y=spam_dataset["v1"]
    y=y.map({"ham":0,
             "spam":1
        
    })
    message=spam_dataset["v2"]

    #convert message into numerical using tfId
    tfid=TfidfVectorizer()
    X=tfid.fit_transform(message)

    #split data into train,test and val data
    x_train,x_,y_train,y_=train_test_split(X,y,test_size=0.4,random_state=1234)
    x_val,x_test,y_val,y_test=train_test_split(x_,y_,test_size=0.5,random_state=1234)


    #map train,test data
    x_train_scaled=x_train
    x_test_scaled=x_test

   
     #load random forest model for spam data
    if(model=="Random Forest"):
        model=pickle.load(open("spam_tree_ensemble.pkl","rb"))
    elif(model=="XGBoost"):
        model=pickle.load(open("spam_xgbclassifier.pkl","rb"))
    elif(model=="Logistic Regression"):
        model=pickle.load(open("spam_Logistic_regression.pkl","rb"))

        





#predict output of train data
y_train_cap=model.predict(x_train_scaled)

#predict output of test data 
y_test_cap=model.predict(x_test_scaled)





#********************************************FIND ACCURACY AND DISPLAY RESULT************************************************************
  

#predict output of input by user data for spam data and display result
if(info["file"]=="spam.csv"):
   
    #predict accuracy of test data
   st.markdown(f"<h3 style=text-align:center;>Model Performance </h3>",unsafe_allow_html=True)
   acc=accuracy_score(y_test,y_test_cap)
   st.markdown(f"<h6 style='text-align: center;'>🎯 Model Test Accuracy: {acc*100:.2f}%</h6ss>",unsafe_allow_html=True)    
    
    #Take Input from user
   input_message=st.text_area(
        "Enter Message here",
        placeholder="⚠️ Please enter a message before clicking Predict."
    )
    

        
   input_mapped_scaled =tfid.transform([input_message])
   
   if st.button("Predict"):  
       if input_message.strip()=="":
           st.warning("Please Enter message as you still didnt gave Input")
           st.stop()
       else:
           input_cap=model.predict(input_mapped_scaled)
   else:
      st.stop()
     
   #predict probqbality of output for given input
   st.markdown(f"<h3 style=text-align:center;> Prediction </h3>",unsafe_allow_html=True)
   out=model.predict_proba(input_mapped_scaled)
   if(input_cap[0]==1):
        st.error( f"🚨 Spam\n\nConfidence: {out[0][1]*100:.2f}%")
   else:
       st.error(f"✅ Not Spam\n\nConfidence:{out[0][0]*100:.2f}%")
   df_prediction_proba=pd.DataFrame(out)
   df_prediction_proba.columns=['Not Spam','Spam']
  
   st.dataframe(df_prediction_proba,
                 column_config={"Not Spam":st.column_config.ProgressColumn(
                     "Not Spam",
                     format="%.4f",
                     width='medium',
                     min_value=0,max_value=1),
                    "Spam":st.column_config.ProgressColumn(
                        "Spam",
                        format="%.4f",
                        width='medium',
                        min_value=0,max_value=1
                    )
                                
                                },hide_index=True
                )
        
       
   






#*******************************************************************************************************

    

#Display Result like overall accuracy and  for house ,insurance dataset
if info["file"]in ["Housing.csv" ,"insurance.csv"]:
    r2 = r2_score(y_test, y_test_cap)
    
    #Display R**2 score accuracy 
    st.metric(label="📊 Test R² Score",value=f"{r2:.3f}") 
    input_cap=model.predict(input_mapped_scaled)
    if input_cap[0] < 0:
        st.warning("Input combination is outside the model's reliable range.")
    else:
        price=max(0,input_cap[0])
        if(info["file"]=="Housing.csv"):
           st.success(f"🏡 Predicted House Price: ${price:,.2f}")
        elif (info["file"]=="insurance.csv"):
           st.success(f"🏡 Predicted medical  charge: ${price:,.2f}")
       

#Display Result like overall accuracy and  for heart dataset   
elif(info["file"]=="heart.csv"):
    
    #find accuracy of test data and display probabality value
    st.markdown(f"<h3 style=text-align:center;>Model Performance </h3>",unsafe_allow_html=True)
    acc=accuracy_score(y_test,y_test_cap)
    st.markdown(f"<h6 style='text-align: center;'>🎯 Model Test Accuracy: {acc*100:.2f}%</h6ss>",unsafe_allow_html=True)
    out=model.predict_proba(input_mapped_scaled)
    input_cap=model.predict(input_mapped_scaled)
    if(input_cap[0]==1):
        st.error( f"🚨 Heart Disease\n\nConfidence: {out[0][1]*100:.2f}%")
    else:
       st.error(f"✅ No Heart Disease\n\nConfidence:{out[0][0]*100:.2f}%")
    df_prediction_proba=pd.DataFrame(out)
    df_prediction_proba.columns=['No Heart Disease','Heart Disease']
    
    st.subheader('Prediction')
    
    st.dataframe(df_prediction_proba,
                 column_config={'No Heart Disease':st.column_config.ProgressColumn(
                     "No Heart Disease",
                     format="%.5f",
                     width='medium',
                     min_value=0,
                     max_value=1),
                 "Heart Disease":st.column_config.ProgressColumn(
                     'Heart Disease',
                     format="%.5f",
                     width='medium',
                     min_value=0,max_value=1),
                 },hide_index=True
        
    )

    
    







      
    
    
    
















   
   
