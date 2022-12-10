import streamlit as st
from streamlit_option_menu import option_menu
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt
import validators
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import shutil
from st_aggrid import AgGrid

# col1, col2 = st.columns([1, 3])
# st.sidebar.title("UltraSec Phishing Detector")
# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Navigation Menu", ["Home", 'Phishing Prediction','Project Details','Example',
    'Model Test Result','Data Set','Setting','Extension Download'], 
        icons=['house','caret-right','book','list-task','bar-chart-line','caret-right','caret-right','download'], menu_icon="cast", default_index=1)
if selected == "Home":
    st.title('Phishing Detection')
    # st.write('This ML-based app is developed for educational purposes. Objective of the app is detecting phishing websites only using content data. Not URL!'
    #          ' You can see the details of approach, data set, and feature set if you click on _"See The Details"_. ')
    choice = st.selectbox("Please select your machine learning model",
                    [
                        'Logistic Regression','Gaussian Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
                        'AdaBoost', 'Neural Network', 'K-Neighbours'
                    ]
                    )
    model = ml.nb_model
    if choice == 'Logistic Regression':
        model = ml.lr_model
        st.write('LR model is selected!')
    elif choice == 'Gaussian Naive Bayes':
        model = ml.nb_model
        st.write('GNB model is selected!')
    elif choice == 'Support Vector Machine':
        model = ml.svm_model
        st.write('SVM model is selected!')
    elif choice == 'Decision Tree':
        model = ml.dt_model
        st.write('DT model is selected!')
    elif choice == 'Random Forest':
        model = ml.rf_model
        st.write('RF model is selected!')
    elif choice == 'AdaBoost':
        model = ml.ab_model
        st.write('AB model is selected!')
    elif choice == 'Neural Network':
        model = ml.nn_model
        st.write('NN model is selected!')
    else:
        model = ml.kn_model
        st.write('KN model is selected!')
    url = st.text_input('Enter the URL')
    # check the url is valid or not
    if st.button('Check!'):
        my_bar = st.progress(0)
        with st.spinner('Wait for it...'):
            try:
                valid=validators.url(url)
                if valid==True:
                    st.success("Url Format is valid")
                else:
                    st.warning("Invalid url")
                response = re.get(url, verify=False, timeout=4)
                if response.status_code == 404:
                    st.warning("404 Client Error. HTTP connection was not successful for the URL: "+url)
                elif response.status_code != 200:
                    print(". HTTP connection was not successful for the URL: ", url)
                    st.warning("Attention! This web page is a potential PHISHING! HTTP connection was not successful for the URL: "+url)
                else:
                    soup = BeautifulSoup(response.content, "html.parser")
                    vector = [fe.create_vector(soup)]  # it should be 2d array, so I added []
                    result = model.predict(vector)
                    if result[0] == 0:
                        st.success("This web page seems a legitimate!")
                        st.balloons()
                    else:
                        st.warning("Attention! This web page is a potential PHISHING!")
                        st.snow()
            except re.exceptions.RequestException as e:
                print("--> ", e)
                st.error(e,icon=None)   
        my_bar = st.progress(100)
        st.success('Done!')   
        # except re.exceptions.HTTPError as err:
        #     raise SystemExit(err)
if selected == "Phishing Prediction":
    st.title('Phishing Detection')
    # check the url is valid or not
    def evaluate(index,model,url):
        with st.spinner('Wait for it...'):
            try:
                valid=validators.url(url)
                if valid==True:
                    # st.success("Url Format is valid")
                    url_valid = "Url Format is valid"
                else:
                    # st.warning("Invalid url")
                    url_valid = "Invalid url"
                
                response = re.get(url, verify=False, timeout=4)
                if response.status_code == 404:
                    # st.warning("404 Client Error. HTTP connection was not successful for the URL: "+url)
                    url_response = "404 Client Error. HTTP connection was not successful for the URL: "
                elif response.status_code != 200:
                    print(". HTTP connection was not successful for the URL: ", url)
                    # st.warning("Attention! This web page is a potential PHISHING! HTTP connection was not successful for the URL: "+url)
                    url_response = "Attention! This web page is a potential PHISHING! HTTP connection was not successful for the URL: "
                else:
                    soup = BeautifulSoup(response.content, "html.parser")
                    vector = [fe.create_vector(soup)]  # it should be 2d array, so I added []
                    result = model.predict(vector)
                    if result[0] == 0:
                        # st.success("This web page seems a legitimate!")
                        url_response = "This web page seems a legitimate!"
                        # st.balloons()
                    else:
                        # st.warning("Attention! This web page is a potential PHISHING!")
                        url_response ="Attention! This web page is a potential PHISHING!"
                        # st.snow()
                data_return = {}
                data_return = {
                    index : [
                        url_valid,
                        url_response
                    ]
                } 
                # data_return = {
                #     index : index,
                #     index+ ' Format':url_valid,
                #     index+ ' Result':url_response,
                # } 

            except re.exceptions.RequestException as e:
                print("--> ", e)
                st.error(e,icon=None)   
        return data_return
    url = st.text_input('Enter the URL')
    if st.button('Check!'):
        my_bar = st.progress(0)
        complete_progress = 0

        # st.write('Logistic Regression')
        model = ml.lr_model
        index = 'LR'
        lr_result = evaluate(index,model,url)
        complete_progress+=13
        my_bar.progress(complete_progress)
        
        # st.write('Gaussian Naive Bayes')
        model = ml.nb_model
        index = 'NB'
        gnb_result = evaluate(index,model,url)
        complete_progress+=13
        my_bar.progress(complete_progress)

        # st.write('Support Vector Machine')
        model = ml.svm_model
        index = 'SVM'
        svm_result = evaluate(index,model,url)
        complete_progress+=13
        my_bar.progress(complete_progress)

        # st.write('Decision Tree')
        model = ml.dt_model
        index = 'DT'
        dt_result = evaluate(index,model,url)
        complete_progress+=13
        my_bar.progress(complete_progress)

        # st.write('Random Forest')
        model = ml.rf_model
        index = 'RF'
        rf_result = evaluate(index,model,url)
        complete_progress+=13
        my_bar.progress(complete_progress)

        # st.write('AdaBoost')
        model = ml.ab_model
        index = 'AB'
        ab_result = evaluate(index,model,url)
        complete_progress+=13
        my_bar.progress(complete_progress)

        # st.write('Neural Network')
        model = ml.nn_model
        index = 'NN'
        nn_result = evaluate(index,model,url)
        complete_progress+=13
        my_bar.progress(complete_progress)

        # st.write('K-Neighbours')
        model = ml.kn_model
        index = 'KN'
        kn_result = evaluate(index,model,url)
        complete_progress+=9
        my_bar.progress(complete_progress)

        # def Merge(dict1, dict2,dict3,dict4,dict5,dict6,dict7,dict8):
        #     res = {**dict1, **dict2,**dict3,**dict4,**dict5,**dict6,**dict7,**dict8}
        #     return res

        column_index = ['Logistic Regression','Gaussian Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
        'AdaBoost', 'Neural Network', 'K-Neighbours']
            
        # Driver code
        dict_result = {k:v for d in (lr_result,gnb_result,svm_result,dt_result,rf_result,ab_result,nn_result,kn_result) for k,v in d.items()}
        # dict_result= Merge(lr_result,gnb_result,svm_result,dt_result,rf_result,ab_result,nn_result,kn_result)
        # st.write(dict_result)
        # dict_result = {k:v for d in (lr_result,gnb_result) for k,v in d.items()}

        # lr = pd.DataFrame(lr_result)
        # nb = pd.DataFrame(gnb_result)

        # lr.reset_index(drop=True, inplace=True)
        # nb.reset_index(drop=True, inplace=True)

        # concated = pd.concat([lr,nb],axis=0)
        # st.write(dict_result) 
        
        # result_df = pd.concat([pd.DataFrame(lr_result),pd.DataFrame(gnb_result),
        # pd.DataFrame(svm_result),pd.DataFrame(dt_result),pd.DataFrame(rf_result),
        # pd.DataFrame(ab_result),pd.DataFrame(nn_result,kn_result)],axis=0,index=indexes)
        # # st.write(result_df)
        row_index = ['URL Format Validity','Result']
        result_df = pd.DataFrame(data=dict_result,index=row_index)
        result_df_T = result_df.transpose()
        result_concat = pd.concat([ml.df_results,result_df_T],axis=1)
        result_concat.index = column_index
        st.success('Done!')
        st.dataframe(result_concat)
        # AgGrid(result_concat)
        
if selected == "Project Details":
    # with st.expander("PROJECT DETAILS"):
    st.subheader('Approach')
    st.write('I used _supervised learning_ to classify phishing and legitimate websites. '
            'I benefit from content-based approach and focus on html of the websites. '
            'Also, I used scikit-learn for the ML models.'
            )
    st.write('For this educational project, '
            'I created my own data set and defined features, some from the literature and some based on manual analysis. '
            'I used requests library to collect data, BeautifulSoup module to parse and extract features. ')
    st.write('The source code and data sets are available in the below Github link:')
    st.write('_https://github.com/emre-kocyigit/phishing-website-detection-content-based_')
    st.subheader('Data set')
    st.write('I used _"phishtank.org"_ & _"tranco-list.eu"_ as data sources.')
    st.write('Totally 26584 websites ==> **_16060_ legitimate** websites | **_10524_ phishing** websites')
    st.write('Data set was created in October 2022.')
    # ----- FOR THE PIE CHART ----- #
    labels = 'phishing', 'legitimate'
    phishing_rate = int(ml.phishing_df.shape[0] / (ml.phishing_df.shape[0] + ml.legitimate_df.shape[0]) * 100)
    legitimate_rate = 100 - phishing_rate
    sizes = [phishing_rate, legitimate_rate]
    explode = (0.1, 0)
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)
    # ----- !!!!! ----- #
    st.write('Features + URL + Label ==> Dataframe')
    st.markdown('label is 1 for phishing, 0 for legitimate')
    number = st.slider("Select row number to display", 0, 100)
    st.dataframe(ml.legitimate_df.head(number))
    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    csv = convert_df(ml.df)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='phishing_legitimate_structured_data.csv',
        mime='text/csv',
    )
if selected == "Example":
    # with st.expander('EXAMPLE PHISHING URLs:'):
    st.title('EXAMPLE PHISHING URLs:')
    st.write('_https://rtyu38.godaddysites.com/_')
    st.write('_https://karafuru.invite-mint.com/_')
    st.write('_https://defi-ned.top/h5/#/_')
    st.write('_https://beeflash.net/_')
    st.caption('REMEMBER, PHISHING WEB PAGES HAVE SHORT LIFECYCLE! SO, THE EXAMPLES SHOULD BE UPDATED!')
if selected == "Model Test Result":
    st.subheader('Features')
    st.write('I used only content-based features. I didn\'t use url-based faetures like length of url etc.'
            'Most of the features extracted using find_all() method of BeautifulSoup module after parsing html.')
    st.subheader('Results')
    st.write('I used 8 different ML classifiers of scikit-learn and tested them implementing k-fold cross validation.'
            'Firstly obtained their confusion matrices, then calculated their accuracy, precision and recall scores.'
            'Comparison table is below:')
    st.table(ml.df_results)
    st.write('LR --> Logistic Regression')
    st.write('NB --> Gaussian Naive Bayes')
    st.write('SVM --> Support Vector Machine')
    st.write('DT --> Decision Tree')
    st.write('RF --> Random Forest')
    st.write('AB --> AdaBoost')
    st.write('NN --> Neural Network')
    st.write('KN --> K-Neighbours') 
if selected == "Data Set":
    @st.cache
    def load_data(nrows):
        data = pd.read_csv('malicious_phish.csv', nrows=nrows)
        return data
    # st.text_input(label, value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, *, placeholder=None, disabled=False, label_visibility="visible")
    # weekly_data = load_data(1000)
    # st.write(weekly_data)
    # st.subheader('Weekly Demand Data')
    # st.write(weekly_data)
    # #Bar Chart
    # st.bar_chart(weekly_data)
    st.sidebar.header('User Input Parameters')
    def user_input_features():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features
    df = user_input_features()
    st.subheader('User Input parameters')
    st.write(df)
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    clf = RandomForestClassifier()
    clf.fit(X, Y)
    prediction = clf.predict(df)
    prediction_proba = clf.predict_proba(df)
    st.subheader('Class labels and their corresponding index number')
    st.write(iris.target_names)
    st.subheader('Prediction')
    st.write(iris.target_names[prediction])
    #st.write(prediction)
    st.subheader('Prediction Probability')
    st.write(prediction_proba)
if selected == "Extension Download":
    st.title('Download Page')
    st.subheader('Download the latest version Ultrasec Phishing Detector')
    shutil.make_archive('phishing-detection-plugin-main', 'zip', 'phishing-detection-plugin-main')
    with open("phishing-detection-plugin-main.zip", "rb") as fp:
        btn1 = st.download_button(
            label="Download UltraSec Phishing Detector 1.0.1",
            data=fp,
            file_name='UltraSec-Phishing-Detector.zip',
            mime='application/zip',
            key="1",
        )   
    st.markdown("Looking for way to install extension? [**Watch Video Below**](#video-to-install-ultrasec-phishing-detector-chrome-extension)")
    st.subheader('Instructions')
    st.markdown("[1. Step 1 - download a zip file with the extension](#step-1-download-a-zip-file-with-the-extension)")
    st.markdown("[2. Step 2 - extract the contents of the zip file](#step-2-extract-the-contents-of-the-zip-file)")
    st.markdown("[3. Step 3 - open the extension page in google chrome](#step-3-open-the-extension-page-in-google-chrome)")
    st.markdown("[4. Step 4 - activate developer mode](#step-4-activate-developer-mode)")
    st.markdown("[5. Step 5 - load unpacked extension](#step-5-load-unpacked-extension)")
    st.header('Step 1 - download a zip file with the extension')
    with open("phishing-detection-plugin-main.zip", "rb") as fp:
        btn2 = st.download_button(
            label="Download UltraSec Phishing Detector 1.0.1",
            data=fp,
            file_name='UltraSec-Phishing-Detector.zip',
            mime='application/zip',
            key="2",
        )   
    st.header('Step 2 - extract the contents of the zip file')
    st.write('Right click on the downloaded zip file, then click "Extract Here".')
    st.header('Step 3 - open the extension page in google chrome')
    st.write('''There are several ways todo that.
    
    Option 1: type chrome://extensions in the url bar and press enter.
    Option 2: click on the tree dots in the top right of the browser, then click "More tools" then click "Extensions".
    ''')
    st.header('Step 4 - activate developer mode')
    st.write('Turn on the switch on the top right of the page that says "Developer mode"')
    st.header('Step 5 - load unpacked extension')
    st.write('''Click on the button on the top left of the page that says "Load unpacked".
    
    Then select a folder that contains the manifest.json file.
    ''')
    st.header('Video to install UltraSec Phishing Detector Chrome Extension.')
    video_file = open('install_extension.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)