#Core Pkgs
import streamlit as st #Streamlit is used for build our Data App.

#Exploratory Data Analysis Packages
import pandas as pd #Panel data-For data analysis & Manipulation
import numpy as np #Numerical Computation
st.set_option('deprecation.showPyplotGlobalUse', False)

# Utils
import os
import joblib #for pipelining our model
import hashlib

#Data viz pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ML Interpretation
import lime
import lime.lime_tabular



#DB
from managed_db import *

#This Function is used to Hash Our Password
def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hashes(password,hashed_text):
    if generate_hashes(password) == hashed_text:
        return hashed_text
    return False
 
feature_names_best = ['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']
 
 
gender_dict = {'male':1,'female':2}
feature_dict = {'No':1,'Yes':2}


def get_value(val,my_dict):
    	for key,value in my_dict.items():
         if val == key:
             return value 

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key

def get_fvalue(val):
	feature_dict = {"No":1,"Yes":2}
	for key,value in feature_dict.items():
		if val == key:
			return value 
#Load Ml models        
def load_model(model_file):
    loaded_model  = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

# ML Interpretation
import lime
import lime.lime_tabular


html_temp = """
		<div style="background-color:black;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Disease Mortality Prediction</h1>
		<h5 style="color:white;text-align:center;">Hepatitis B </h5>
		</div>
		"""

#Avatar Image using a url
avatar1 ="https://www.w3schools.com/howto/img_avatar2.png"
avatar2 ="https://www.w3schools.com/howto/img_avatar2.png"

result_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

result_temp2 ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/{}" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

prescriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Avoid Alchol</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""


descriptive_message_temp ="""
	<div style="background-color:black;overflow-x: auto; padding:10px;border-radius:10px;margin:10px;">
		<h3 style="text-align:justify;color:white;padding:10px">What is Hepatitis B?</h3>
		<p style="color:white;">Hepatitis B is an infection of your liver. It’s caused by a virus. There is a vaccine that protects against it. For some people, hepatitis B is mild and lasts a short time. These “acute” cases don’t always need treatment. But it can become chronic. If that happens, it can cause scarring of the organ, liver failure, and cancer, and it even can be life-threatening.It’s spread when people come in contact with the blood, open sores, or body fluids of someone who has the hepatitis B virus.
It's serious, but if you get the disease as an adult, it shouldn’t last a long time. Your body fights it off within a few months, and you’re immune for the rest of your life. That means you can't get it again. But if you get it at birth, it’ unlikely to go away.
</p></div>

<div style="background-color:black;overflow-x: auto; padding:10px;border-radius:10px;margin:10px;">
<p><h3 style="color:white;" ><bold>Hepatitis B Symptoms</bold></h3>
<p style="color:white;">Short-term (acute) hepatitis B infection doesn’t always cause symptoms. For instance, it’s uncommon for children younger than 5 to have symptoms if they’re infected.
If you do have symptoms, they may include:</p>
<li style="color:white;" >Jaundice (Your skin or the whites of the eyes turn yellow, and your pee turns brown or orange.)</li>
<li style="color:white;" >Light-colored poop</li>
<li style="color:white;" >Fever</li>
<li style="color:white;" >Fatigue that persists for weeks or months</li>
<li style="color:white;" >Stomach trouble like loss of appetite, nausea, and vomiting</li>
<li style="color:white;" >Belly pain</li>
<li style="color:white;" >Joint pain</li>
<p style="color:white;">Symptoms may not show up until 1 to 6 months after you catch the virus. You might not feel anything. About a third of the people who have this disease don’t. They find out only through a blood test.
Symptoms of long-term (chronic) hepatitis B infection don’t always show up, either. If they do, they may be like those of short-term (acute) infection.
<br>
<h3><bold style="color:white;">Hepatitis B Causes and Risk Factors</bold></h3>
<p style="color:white;">It’s caused by the hepatitis B virus, and it can spread from person to person in certain ways. You can spread the hepatitis B virus even if you don’t feel sick.The most common ways to get hepatitis B include:</p>
<li style="color:white;" >Sex</li>
<li style="color:white;" >Sharing needles</li>
<li style="color:white;" >Accidental needle sticks</li>
<li style="color:white;" >Mother to child(Pregnancy)</li></p>
<p style="color:white;">The number of people who get this disease is down, the CDC says. Rates have dropped from an average of 200,000 per year in the 1980s to around 20,000 in 2016. People between the ages of 20 and 49 are most likely to get it.
About 90% of infants and 25-50% of children between the ages of 1-5 will become chronically infected. In adults, approximately 95% will recover completely and will not go on to have a chronic infection.</p>
<br/>
<br/></div>
<div style="background-color:black;overflow-x: auto; padding:10px;border-radius:10px;margin:10px;">
<li><a href="https://www.webmd.com/hepatitis/digestive-diseases-hepatitis-b">Learn more about Hepatitis B</a></li>    
</div>


	"""
#@st.cache
#def load_image(img):
#    im = Image.open(os.path.join(img))
 #   return im
	

def change_avatar(sex):
	if sex == "male":
		avatar_img = 'img_avatar.png'
	else:
		avatar_img = 'img_avatar2.png'
	return avatar_img
 
 

#Python Function
def main():
    """Hepatitis Mortality Prediction Application""" 
    st.markdown(html_temp.format('royalblue'),unsafe_allow_html=True)
    
    menu = ['Home','Login','Signup']
    submenu = ['Plot','Prediction']
    
    
    choice = st.sidebar.selectbox('Menu',menu)
    if choice ==  'Home':
        st.subheader('Home')
        #st.text("What is Hepatitis")
        st.markdown(descriptive_message_temp,unsafe_allow_html=True)
        #st.image(load_image('images/hepimage.jpeg'))
        
    elif choice =='Login':
        username = st.sidebar.text_input('Username')
        password = st.sidebar.text_input('Password', type='password')
        if st.sidebar.checkbox('Login'):
            create_usertable()
            hashed_pswd = generate_hashes(password)
            result = login_user(username,verify_hashes(password,hashed_pswd))
            if result:
            #if password == '12345':
                st.success('Welcome {}'.format(username))
                
                activity = st.selectbox('Activity',submenu)
                if activity == 'Plot':
                    st.subheader('Data Viz Plot')
                    df = pd.read_csv('data/clean_hepatitis_dataset.csv')
                    st.dataframe(df)
                    
                    df['class'].value_counts().plot(kind='bar')
                    st.pyplot()
                    
                    #Frequency Distribution Plot
                    freq_df = pd.read_csv('data/freq_df_hepatitis_dataset.csv')
                    st.bar_chart(freq_df['count'])
                    
                    if st.checkbox('Area Chart'):
                        all_columns = df.columns.to_list()
                        feat_choices = st.multiselect('Choose a Feature',all_columns)
                        new_df = df[feat_choices]
                        st.area_chart(new_df)
                 
                 
                    
                elif activity == 'Prediction':
                    st.subheader('Predictive Analytics')
                    
                    age = st.number_input('Age',7,80)
                    sex = st.radio('Sex',tuple(gender_dict.keys()))
                    steroid = st.radio("Do You Take Steriods?",tuple(feature_dict.keys()))
                    antivirals = st.radio("Do You Take Anti-virus?",tuple(feature_dict.keys()))
                    fatigue = st.radio("Do You Have Fatigue",tuple(feature_dict.keys()))
                    spiders = st.radio('Presence of Spider Naevi', tuple(feature_dict.keys()))
                    ascites = st.selectbox('Ascities',tuple(feature_dict.keys()))
                    varices = st.selectbox("Presence of Varices",tuple(feature_dict.keys()))
                    bilirubin = st.number_input("bilirubin Content",0.0,8.0)
                    alk_phosphate = st.number_input("Alkaline Phosphate Content",0.0,296.0)
                    sgot = st.number_input("Sgot",0.0,648.0)
                    albumin = st.number_input("Albumin",0.0,6.4)
                    protime = st.number_input("Prothrombin Time",0.0,100.0)
                    histology = st.selectbox("Histology",tuple(feature_dict.keys()))
                    feature_list = [age,get_value(sex,gender_dict),get_fvalue(steroid),
                                    get_fvalue(antivirals),get_fvalue(fatigue),
                                    get_fvalue(spiders),get_fvalue(ascites),
                                    get_fvalue(varices),bilirubin,alk_phosphate,sgot,albumin,int(protime),
                                    get_fvalue(histology)]
                    st.write(len(feature_list))
                    st.write(feature_list)
                    pretty_result = {"age":age,"sex":sex,"steroid":steroid,"antivirals":antivirals,"fatigue":fatigue,
                     "spiders":spiders,"ascites":ascites,"varices":varices,"bilirubin":bilirubin,
                     "alk_phosphate":alk_phosphate,"sgot":sgot,"albumin":albumin,"protime":protime,
                     "histolog":histology}
                    st.json(pretty_result)
                    single_sample = np.array(feature_list).reshape(1,-1)
                    
                    
                    
                    #Machine Learning model
                    model_choice = st.selectbox("Select Model",["KNN","DecisionTree","Logistic Tree"])
                    if st.button('Predict'):
                        if model_choice == 'KNN':
                            loaded_model = load_model("models/knn_hepB_model.pkl")
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(single_sample)
                        elif model_choice == "DecisionTree":
                            loaded_model = load_model("models/decision_tree_clf_hepB_model.pkl")
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(single_sample)
                        else:
                            loaded_model = load_model("models/logistic_regression_hepB_model.pkl")
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(single_sample)
                            
                            
                        if prediction == 1:
                                st.warning('Patient Dies')
                                pred_probability_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}
                                st.subheader("Prediction Probability Score using {}".format(model_choice))
                                st.json(pred_probability_score)
                                st.subheader("Prescriptive Analytics")
                                st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
                                
                                
                        else :
                                st.success('Patient Lives')
                                pred_probability_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}
                                st.subheader("Prediction Probability Score using {}".format(model_choice))
                                st.json(pred_probability_score)
                                #st.subheader("Prescriptive Analytics")
                                #st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
                                
                                
                                
                    if st.checkbox("Interpret"):
                        if model_choice == "KNN":
                                loaded_model = load_model("models/knn_hepB_model.pkl")
                        elif model_choice == "DecisionTree":
                                loaded_model = load_model("models/decision_tree_clf_hepB_model.pkl")
                        else:
                                loaded_model = load_model("models/logistic_regression_hepB_model.pkl")
                                
                                
                                df = pd.read_csv("data/clean_hepatitis_dataset.csv")
                                x = df[['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']]
                                feature_names = ['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']
                                class_names = ['Die(1)','Live(2)']
                                explainer = lime.lime_tabular.LimeTabularExplainer(x.values,feature_names=feature_names, class_names=class_names,discretize_continuous=True)
                                
                                
                                # The Explainer Instance
                                exp = explainer.explain_instance(np.array(feature_list),
                                loaded_model.predict_proba,num_features=13, top_labels=1)
                                exp.show_in_notebook(show_table=True, show_all=False)
                                #exp.save_to_file('lime_oi.html')
                                st.write(exp.as_list())
                                new_exp = exp.as_list()
                                label_limits = [i[0] for i in new_exp]
                                
                                # st.write(label_limits)
                                label_scores = [i[1] for i in new_exp]
                                plt.barh(label_limits,label_scores)
                                st.pyplot()
                                plt.figure(figsize=(20,10))
                                fig = exp.as_pyplot_figure()
                                st.pyplot()
                                
                    
                               
                    
                
            else:
                st.warning('Incorrect Username/Password')
                
    elif choice =='Signup':
        new_username = st.text_input('User name')
        new_password = st.text_input('Password', type='password')
        
        confirm_password = st.text_input('Confirm Password',type='password')
        if new_password == confirm_password:
            st.success('Password Confirmed')
        else:
            st.warning('Password not the same')
            
        if st.button('Submit'):
            create_usertable()
            hashed_new_password = generate_hashes(new_password)
            add_userdata(new_username,hashed_new_password)
            st.success('You have sucessfully created a New Account')
            st.info('Login to Get started')
                

if __name__ == '__main__':
	main()