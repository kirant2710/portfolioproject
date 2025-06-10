import streamlit as st

if "page_configured" not in st.session_state:
    st.set_page_config(layout="wide")
    st.session_state["page_configured"] = True

import plotly.graph_objects as go
from datetime import datetime
import json
import requests
import smtplib
import ssl
import pandas as pd
import numpy as np

import streamlit_shadcn_ui as ui
from streamlit_extras.metric_cards import style_metric_cards


# if "page_configured" not in st.session_state:
#     st.set_page_config(layout="wide")
#     st.session_state["page_configured"] = True

def display_metric(label, value, column):
    """Displays a metric in the specified column."""
    # st.write(f"display_metric called with label={label}, value={value}")
    with column:
        st.metric(label=label, value=value)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["About Me", "Professional Journey", "Academic Journey","Certifications","Projects", "Expertise In" , "Teaching Experience", "Consulting", "Corporate Training", "Acedemic Engagements"])


project_data = {
    "MICRON": [
        "Published more than 10 + internal papers (adjudged in top2 globally), Filled 5+ invention disclosers won multiple awards like IDEA of Quarters, Cultural Championships, Best performer awards in the team.",
        "Quantum Machine Learning suite for variation identification in the semiconductor manufacturing process. <span style=\"color: green;\">**↑ Improved variation identification accuracy by 15% ↑**</span>.",
        "Code Assistant Tool (Intelligence Tool) for building E2End Data Science Applications. <span style=\"color: green;\">**↑ Reduced development time by 20% ↑**</span>.",
        "Reinforcement learning based Engine for Tool Trouble shooting in Semiconductor Fabrication Process. <span style=\"color: green;\">**↑ Decreased troubleshooting time by 25% ↑**</span>.",
        "Built advanced statistics/data-science models for identifying variability which in turn enhances the productivity or yield gain. Computer Vision Gyan based model identifying defect patterns. <span style=\"color: green;\">**↑ Increased yield by 10% ↑**</span>.",
        "built highly performant model suite which can consume less memory and run ultra fast, based on cpu/gpu acceleration tools increased the  time by 50-60% and reduced the cost of computation by 50% <span style=\"color: green;\">**↑ increased the  time by 50-60% ↑**</span> and <span style=\"color: red;\">**↓ reduced the cost of computation by 50% ↓**</span>",
        "Planned and executed the CDS ( citizen data science model ) in multiple fabs of micron encouraging all the manufacturing engineers to come up with problem statements and solve them using the CDS models which in turn increased the efficiency of fabs by 20% <span style=\"color: green;\">**↑ increased the efficiency of fabs by 20% ↑**</span>"
    ],
    "GEP SOLUTIONS": [
        "Built NLP based supplier intent identifier, sentiment detector from news feed data. <span style=\"color: green;\">**↑ Achieved 90% accuracy in intent identification ↑**</span>.",
        "Built a text summary extractor from news feeds and duplicate news detection. <span style=\"color: green;\">**↑ Reduced manual effort by 30% ↑**</span>.",
        "Built promotional modelling for demand planners which is included in NEXXE (GEP product). <span style=\"color: green;\">**↑ Improved forecast accuracy by 12% ↑**</span>.",
        "Built a large-scale Demand Planning and Demand Sensing pipeline for the Sales data which runs with different time series models including hierarchical time series-based models. <span style=\"color: green;\">**↑ Reduced processing time by 40% ↑**</span>.",
        "Built an optimized hierarchical time series model which runs on py spark with 2000 items in 5 min span. <span style=\"color: green;\">**↑ Improved model training time by 50% ↑**</span>.",
        "Built a spend forecasting pipeline with different time series algorithms. <span style=\"color: green;\">**↑ Increased forecasting accuracy by 8% ↑**</span>.",
        "Built a Hybrid Time Series Model for Forecasting Commodities and Sales Data. <span style=\"color: green;\">**↑ Outperformed existing models by 15% in accuracy ↑**</span>."
    ],
    "HIGH RADIUS PVT LIMITED": [
        "Proactively built and automated the claim extraction process from pdf templates by using deep learning and NLP. <span style=\"color: green;\">**↑ Extracted claims with 95% accuracy ↑**</span>.",
        "Implemented GANs, Semi Supervised GANs for check/remittance identification/classification instead of labeling for each data point. <span style=\"color: green;\">**↑ Reduced labeling effort by 70% ↑**</span>.",
        "Customer Segmentation model for high radius Cash App customers. <span style=\"color: green;\">**↑ Improved customer retention by 5% ↑**</span>.",
        "Implemented “HiFreeda” Wake word detection model for high radius chat bot FREEDA. <span style=\"color: green;\">**↑ Reduced wake word detection latency by 30% ↑**</span>.",
        "Implemented Noise Detection GANs for the noise removal from images. <span style=\"color: green;\">**↑ Improved image clarity by 20% ↑**</span>.",
        "Built and implemented deduction validity predictor for P&G. <span style=\"color: green;\">**↑ Increased prediction accuracy by 10% ↑**</span>.",
        "Implemented auto regex generator which gives regex by using pattern in the data. <span style=\"color: green;\">**↑ Reduced regex creation time by 50% ↑**</span>.",
        "Implemented and created many feature engineering techniques, designed new algorithms for Oversampling analysis for imbalanced data and ML (Pseudo Random Forest which works better than RF in ML). <span style=\"color: green;\">**↑ Pseudo Random Forest achieved top rank in internal ML competition ↑**</span>."
    ],
    "BITMIN INFO SYSTEMS": [
        "Developed apps, worked on the back-end and front-end enabling the existence of features and application development.",
        "Created Machine Learning-based tools for the company, such as Employee Iteration Rate prediction, Payment Date Prediction, Fraud Detection Analytics. <span style=\"color: green;\">**↑ Improved prediction accuracy by 15% ↑**</span>.",
        "Built a webservice in .net which extracts the payments from quick books, also worked on building a proximity distance app (ceechat) in android."
    ],
    "PIXENTIA SOLUTIONS": [
        "Collaborated with team members to create applications system analysis based upon client requirements. Created tools like resume parser, Developed digital badge online representation of a skill. <span style=\"color: green;\">**↑ Awarded 'Innovation Badge' for developing the digital badge system ↑**</span>."
    ],
    "GYAN DATA Pvt. LTD": [
        "Developed and hosted apps like remote scilab for computations in chemical engineering labs remotely, worked on enterprise level applications like sandman with machine learning as core."
    ],
    "ACADEMIC PROJECTS": [
        "Developed a machine learning project for Predicting prepayments and defaults for privatized banks. <span style=\"color: green;\">**↑ Achieved 85% accuracy in predicting defaults ↑**</span>.",
        "Wake word detection model for the custom word HIFREEDA",
        "Build a CNN based image classifier, which will be able to distinguish between different handwritten Devanagari characters. <span style=\"color: green;\">**↑ Achieved 92% classification accuracy ↑**</span>.",
        "Distinguishing between Hillary Clinton and Trump tweets.",
        "Developed a streaming analytics application to analyze movie recommendations.",
        "Music Generation using Deep Learning.",
        "Built and implement and machine learning NLP based project which computes sentiment for an incoming review in amazon online website."
    ],
    "PERSONAL PROJECTS": [
        "Designed an Algorithm called as Pseudo Random Forest Which works better than Random Forest algorithm. <span style=\"color: green;\">**↑ Improved accuracy by 10% compared to Random Forest ↑**</span>.",
        "Implemented and invented a new technique for oversampling.",
        "Designed TIC-TOC game machine with Reinforcement Learning. <span style=\"color: green;\">**↑ Achieved a winning rate of 90% against random players ↑**</span>.",
        "Persona based custom feed recommendation engine.",
        "Chat bot with RNN."
    ]
}

with tab5:
    st.header("Projects")
    for company, projects in project_data.items():
        st.subheader(company)
        for project in projects:
            st.write(f"* {project}", unsafe_allow_html=True)

with tab6:
    st.header("Expertise In")

    # Load skills data
    def load_skills_data():
        """Loads skills data from skills_data.json and handles potential errors."""
        if "skills_data" not in st.session_state:
            try:
                with open("skills_data.json", "r") as f:
                    skills_data = json.load(f)
                roles = skills_data["roles"]
                st.session_state["skills_data"] = roles
                return roles
            except FileNotFoundError:
                st.error("Could not load skills data. Please ensure skills_data.json exists.")
                return []
            except json.JSONDecodeError:
                st.error("Could not decode skills data. Please ensure skills_data.json is valid JSON.")
                return []
        else:
            return st.session_state["skills_data"]

    roles = load_skills_data()

    # Custom CSS for styling
    # st.markdown(
    #     """
    #     <style>
    #     /* Style the dataframe */
    #     .ag-theme-streamlit {
    #         --ag-foreground-color: #000000;
    #         --ag-background-color: #ffffff;
    #         --ag-header-foreground-color: #ffffff;
    #         --ag-header-background-color: #262730;
    #     }

    #     .skills-container {
    #         display: flex;
    #         flex-wrap: wrap;
    #         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    #         transition: transform 0.3s ease-in-out;
    #     }
    #     .skill-card:hover {
    #         margin: 20px;
    #         padding: 15px;
    #         border: 1px solid #ddd;
    #         border-radius: 10px;
    #         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    #         transition: transform 0.3s ease-in-out;
    #     }
    #     .skill-card:hover {
    #         transform: translateY(-5px);
    #     }
    #     .skill-card h3 {
    #         color: #333;
    #     }
    #     .skill-card p {
    #         color: #666;
    #     }
    #     .skill-card img {
    #         width: 50px;
    #         height: 50px;
    #         margin-right: 10px;
    #         vertical-align: middle;
    #     }
    #     .skill-icon-small {
    #         width: 30px;
    #         height: 30px;
    #     }
    #     /* Responsive design */
    #     @media (max-width: 768px) {
    #         .skill-card {
    #             width: 100%; /* Full width on smaller screens */
    #             margin: 10px 0;
    #         }
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # # Display skills
    # st.write("<div class='skills-container'>", unsafe_allow_html=True)
    # for role in roles:
    #     # st.write(f"<div class='skill-card'>", unsafe_allow_html=True)
    #     st.write(f"<h3>{role['name']}</h3>", unsafe_allow_html=True)
    #     for skill in role["skills"]:
    #         icon_html = f"<img class=\"skill-icon-small\" src=\"{skill['icon']}\" alt=\"{skill['name']} icon\">"
    #         st.write(f"<p>{icon_html} {skill['name']}: {skill['description']}</p>", unsafe_allow_html=True)
    #     st.write("</div>", unsafe_allow_html=True)
    # st.write("</div>", unsafe_allow_html=True)

    # # Add Skill Form (optional)
    # with st.expander("Add New Skill (Optional)"):
    #     with st.form("add_skill_form"):
    #         skill_name = st.text_input("Skill Name")
    #         skill_description = st.text_area("Skill Description")
    #         skill_icon = st.text_input("Skill Icon URL")
    #         skill_category = st.selectbox("Skill Category", options=[role["name"] for role in roles])
    #         submitted = st.form_submit_button("Add Skill")

    #         if submitted:
    #             new_skill = {
    #                 "name": skill_name,
    #                 "description": skill_description,
    #                 "icon": skill_icon
    #             }
    #             for role in roles:
    #                 if role["name"] == skill_category:
    #                     role["skills"].append(new_skill)
    #                     break

    #             # Write the updated data back to the JSON file
    #             with open("skills_data.json", "w") as f:
    #                 json.dump({"roles": roles}, f, indent=2)

    #             st.success("Skill added successfully!")

    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>Hands-On Generative AI Stack</h3>", unsafe_allow_html=True)
        # st.image("Generative_AI_Stack.png", width=800)
        
        with st.container(border=True):
            col1, col2, col3, col4, col5, col6,col7 = st.columns(7)
            with col1:
                st.write("Front-End Frameworks")
            with col2:
                st.image("react.png", width=100)
            with col3:
                st.image("angularjs.png", width=100)
            with col4:
                st.image("vuejslogo.png", width=100)
            with col5:
                st.image("svelte.png", width=100)
            with col6:
                st.image("nextjs.png", width=100)
            with col7:
                st.image("vercel.png",width=100)
        
        with st.container(border=True):
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            with col1:
                st.write("Frameworks")
            with col2:
                st.image("langchain.png", width=100)
            with col3:
                st.image("llamaindex.png", width=100)
            with col4:
                st.image("haystack.png", width=100)
            with col5:
                st.image("rag.png", width=100)
            with col6:
                st.image("huggingface.png", width=100)
            with col7:
                st.image('langgraph.png',width=100)

        with st.container(border=True):
            col1, col2, col3, col4, col5, col6, col7,col8, col9, col10, col11, col12 = st.columns(12)
            with col1:
                st.write("Databases")
            with col2:
                st.image("weaviate.png")
            with col3:
                st.image("pinacone.png", width=150)
            with col4:
                st.image("chromadb.png", width=150)
            with col5:
                st.image("milvus.png", width=150)
            with col6:
                st.image("zilliz.png", width=150)
            with col7:
                st.image("qdrant.png", width=150)
            with col8:
                st.image("redis.png", width=150)
            with col9:
                st.image("faiss.png", width=150)
            with col10:
                st.image("annoy.png", width=150)
            with col11:
                st.image("hnsw.jpg", width=150)
            with col12:
                st.image("pgvector.png", width=150)
        
        
        with st.container(border=True):
            col1,col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)
            with col1:
                st.write("Foundation Models")
            with col2:
                st.image("gpt4.5.png", width=150)
            with col3:
                st.image("gpt3.5.jpeg", width=150)
            with col4:
                st.image("claude.png", width=150)
            with col5:
                st.image("geminipro.png", width=150)
            with col6:
                st.image("llama3.jpg", width=150)
            with col7:
                st.image("mistral.png", width=150)
            
            with col8:
                st.image("rplus.jpg", width=150)
            with col9:
                st.image("codet5.png", width=150)
            with col10:
                st.image("bert.png", width=150)
            with col11:
                st.image("magma.jpg", width=150)
    
        with st.container(border=True):
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
            with col1:
                st.write(" Cloud Providers")
            with col2:
                st.image('lambda.png', width=150)
            with col3:
                st.image('coreweave.jpg', width=150)
            with col4:
                st.image('googlecloud.png', width=150)
            with col5:
                st.image('aws_cloud.png', width=150)
            with col6:
                st.image('ibmcloud.png', width=150)
            with col7:
                st.image('azurecloud.png', width=150)
            with col8:
                st.image('runpod.png',width=150)
            with col9:
                st.image('paperspace.jpg', width=150)
            with col10:
                st.image('vastai.png', width=150)

    
    
    with st.container(border=True):
        st.markdown("<h2 style='text-align: center;'>Programming Data Science</h2>", unsafe_allow_html=True)
        with st.container(border=True, key='DataScience with python'):
            st.markdown("<div style='text-align: center;'><h4>Data Science with Python</h4></div>", unsafe_allow_html=True)
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
            with col1:
                import os
                import requests

                # Troubleshooting guide for "MediaFileStorageError: Error opening 'python.png'"
                # 1. Check if the file path is correct: Ensure that 'python.png' is in the same directory as the script or specify the correct relative/absolute path.
                # 2. Verify file existence: Make sure that 'python.png' actually exists at the specified location.
                # 3. Check file permissions: Ensure that the Streamlit application has the necessary permissions to read the file.
                # 4. Validate file integrity: The file might be corrupted. Try replacing it with a fresh copy.
                # 5. Streamlit media file storage: In some cases, Streamlit might have issues with its media file storage. Restarting the application or clearing the cache might help.
                # 6. Alternative image sources: If the issue persists, consider using an alternative online image source or embedding the image as a base64 string.

                # python_image_path = "python.png"
                # python_image_url = "https://www.python.org/static/community_logos/python-logo-v3-tm-192x64.png"  # Replace with a reliable URL

                # if not os.path.exists(python_image_path):
                #     try:
                #         response = requests.get(python_image_url, stream=True)
                #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                #         with open(python_image_path, "wb") as f:
                #             for chunk in response.iter_content(chunk_size=8192):
                #                 f.write(chunk)
                #         st.success("Python logo downloaded successfully!")
                #     except requests.exceptions.RequestException as e:
                #         st.error(f"Error downloading Python logo: {e}")
                #     except Exception as e:
                #         st.error(f"Error saving Python logo: {e}")
                st.image("Python-Logo.png", width=150)
                # st.write("Python")
            with col2:
                import os
                import requests

                numpy_image_path = "numpy.png"
                # numpy_image_url = "https://raw.githubusercontent.com/numpy/numpy/master/branding/logo/primary/numpylogo.png"  # Replace with a reliable URL

                # if not os.path.exists(numpy_image_path):
                #     try:
                #         response = requests.get(numpy_image_url, stream=True)
                #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                #         with open(numpy_image_path, "wb") as f:
                #             for chunk in response.iter_content(chunk_size=8192):
                #                 f.write(chunk)
                #         st.success("NumPy logo downloaded successfully!")
                #     except requests.exceptions.RequestException as e:
                #         st.error(f"Error downloading NumPy logo: {e}")
                #     except Exception as e:
                #         st.error(f"Error saving NumPy logo: {e}")
                st.image(numpy_image_path, width=150)
                # st.write("NumPy")
            with col3:
                import os
                import requests

                pandas_image_path = "pandas_png2.jpg"
                # pandas_image_url = "https://pandas.pydata.org/static/img/pandas_white.png"  # Replace with a reliable URL

                # if not os.path.exists(pandas_image_path):
                #     try:
                #         response = requests.get(pandas_image_url, stream=True)
                #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                #         with open(pandas_image_path, "wb") as f:
                #             for chunk in response.iter_content(chunk_size=8192):
                #                 f.write(chunk)
                #         st.success("Pandas logo downloaded successfully!")
                #     except requests.exceptions.RequestException as e:
                #         st.error(f"Error downloading Pandas logo: {e}")
                #     except Exception as e:
                #         st.error(f"Error saving Pandas logo: {e}")
                st.image(pandas_image_path, width=150)
                # st.write("Pandas")
            with col4:
                import os
                import requests

                # Troubleshooting guide for "MediaFileStorageError: Error opening 'keras.png'"
                # 1. Check if the file path is correct: Ensure that 'keras.png' is in the same directory as the script or specify the correct relative/absolute path.
                # 2. Verify file existence: Make sure that 'keras.png' actually exists at the specified location.
                # 3. Check file permissions: Ensure that the Streamlit application has the necessary permissions to read the file.
                # 4. Validate file integrity: The file might be corrupted. Try replacing it with a fresh copy.
                # 5. Streamlit media file storage: In some cases, Streamlit might have issues with its media file storage. Restarting the application or clearing the cache might help.
                # 6. Alternative image sources: If the issue persists, consider using an alternative online image source or embedding the image as a base64 string.

                keras_image_path = "keras_logo.png"
                # keras_image_url = "https://keras.io/img/keras-logo-2018-large-1200.png"  # Replace with a reliable URL

                # if not os.path.exists(keras_image_path):
                #     try:
                #         response = requests.get(keras_image_url, stream=True)
                #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                #         with open(keras_image_path, "wb") as f:
                #             for chunk in response.iter_content(chunk_size=8192):
                #                 f.write(chunk)
                #         st.success("Keras logo downloaded successfully!")
                #     except requests.exceptions.RequestException as e:
                #         st.error(f"Error downloading Keras logo: {e}")
                #     except Exception as e:
                #         st.error(f"Error saving Keras logo: {e}")
                st.image(keras_image_path, width=150)
                # st.write("keras")
            with col5:
                import os
                import requests

                tensorflow_image_path = "tf.png"
                # tensorflow_image_url = "https://www.tensorflow.org/images/tf_logo.png"  # Replace with a reliable URL

                # if not os.path.exists(tensorflow_image_path):
                #     try:
                #         response = requests.get(tensorflow_image_url, stream=True)
                #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                #         with open(tensorflow_image_path, "wb") as f:
                #             for chunk in response.iter_content(chunk_size=8192):
                #                 f.write(chunk)
                #         st.success("TensorFlow logo downloaded successfully!")
                #     except requests.exceptions.RequestException as e:
                #         st.error(f"Error downloading TensorFlow logo: {e}")
                #     except Exception as e:
                #         st.error(f"Error saving TensorFlow logo: {e}")
                st.image(tensorflow_image_path, width=150)
                # st.write("tersorflow")
            with col6:
                import os
                import requests

                scikit_learn_image_path = "scikit-learn.png"
                # scikit_learn_image_url = "https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png"  # Replace with a reliable URL

                # if not os.path.exists(scikit_learn_image_path):
                #     try:
                #         response = requests.get(scikit_learn_image_url, stream=True)
                #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                #         with open(scikit_learn_image_path, "wb") as f:
                #             for chunk in response.iter_content(chunk_size=8192):
                #                 f.write(chunk)
                #         st.success("Scikit-learn logo downloaded successfully!")
                #     except requests.exceptions.RequestException as e:
                #         st.error(f"Error downloading Scikit-learn logo: {e}")
                #     except Exception as e:
                #         st.error(f"Error saving Scikit-learn logo: {e}")
                st.image(scikit_learn_image_path, width=150)
                # st.write("Scikit-learn")
            with col7:
                import os
                import requests

                statsmodels_image_path = "statsmodels.png"
                # statsmodels_image_url = "https://www.statsmodels.org/stable/_static/statsmodels_logo_v2_horizontal.svg"  # Replace with a reliable URL

                # if not os.path.exists(statsmodels_image_path):
                #     try:
                #         response = requests.get(statsmodels_image_url, stream=True)
                #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                #         with open(statsmodels_image_path, "wb") as f:
                #             for chunk in response.iter_content(chunk_size=8192):
                #                 f.write(chunk)
                #         st.success("Statsmodels logo downloaded successfully!")
                #     except requests.exceptions.RequestException as e:
                #         st.error(f"Error downloading Statsmodels logo: {e}")
                #     except Exception as e:
                #         st.error(f"Error saving Statsmodels logo: {e}")
                st.image(statsmodels_image_path, width=150)
                # st.write("Statsmodels")
            with col8:
                import os
                import requests

                plotly_image_path = "causalml.png"
                # plotly_image_url = "https://images.plot.ly/language-logos/python-logo-for-printing.png"  # Replace with a reliable URL

                # if not os.path.exists(plotly_image_path):
                #     try:
                #         response = requests.get(plotly_image_url, stream=True)
                #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                #         with open(plotly_image_path, "wb") as f:
                #             for chunk in response.iter_content(chunk_size=8192):
                #                 f.write(chunk)
                #         st.success("Plotly logo downloaded successfully!")
                #     except requests.exceptions.RequestException as e:
                #         st.error(f"Error downloading Plotly logo: {e}")
                #     except Exception as e:
                #         st.error(f"Error saving Plotly logo: {e}")
                st.image(plotly_image_path, width=150)
                # st.write("Plotly")
            with col9:
                import os
                import requests

                pytorch_image_path = "pytorch.png"
                # pytorch_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/PyTorch_logo.svg/320px-PyTorch_logo.svg.png"  # Replace with a reliable URL

                # if not os.path.exists(pytorch_image_path):
                #     try:
                #         response = requests.get(pytorch_image_url, stream=True)
                #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                #         with open(pytorch_image_path, "wb") as f:
                #             for chunk in response.iter_content(chunk_size=8192):
                #                 f.write(chunk)
                #         st.success("PyTorch logo downloaded successfully!")
                #     except requests.exceptions.RequestException as e:
                #         st.error(f"Error downloading PyTorch logo: {e}")
                #     except Exception as e:
                #         st.error(f"Error saving PyTorch logo: {e}")
                st.image(pytorch_image_path, width=150)
                # st.write("pytorch")
            with col10:
                st.image('pymilo.jpg', width=150)


        with st.container(border=True):
            st.markdown("<div style='text-align: center;'><h4>Deployment of Models with Python</h4></div>", unsafe_allow_html=True)
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
            with col1:
                import os
                import requests

                # Troubleshooting guide for "MediaFileStorageError: Error opening 'python.png'"
                # 1. Check if the file path is correct: Ensure that 'python.png' is in the same directory as the script or specify the correct relative/absolute path.
                # 2. Verify file existence: Make sure that 'python.png' actually exists at the specified location.
                # 3. Check file permissions: Ensure that the Streamlit application has the necessary permissions to read the file.
                # 4. Validate file integrity: The file might be corrupted. Try replacing it with a fresh copy.
                # 5. Streamlit media file storage: In some cases, Streamlit might have issues with its media file storage. Restarting the application or clearing the cache might help.
                # 6. Alternative image sources: If the issue persists, consider using an alternative online image source or embedding the image as a base64 string.

                # python_image_path = "python.png"
                # python_image_url = "https://www.python.org/static/community_logos/python-logo-v3-tm-192x64.png"  # Replace with a reliable URL

                # if not os.path.exists(python_image_path):
                #     try:
                #         response = requests.get(python_image_url, stream=True)
                #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                #         with open(python_image_path, "wb") as f:
                #             for chunk in response.iter_content(chunk_size=8192):
                #                 f.write(chunk)
                #         st.success("Python logo downloaded successfully!")
                #     except requests.exceptions.RequestException as e:
                #         st.error(f"Error downloading Python logo: {e}")
                #     except Exception as e:
                #         st.error(f"Error saving Python logo: {e}")
                st.image("flask_python.png", width=300)
                # st.write("Flask")
            with col2:
                
                st.image("fastapi.png", width=300)
                # st.write("NumPy")
            with col3:
                # import os
                # import requests

                # pandas_image_path = "pandas_png2.jpg"
                # # pandas_image_url = "https://pandas.pydata.org/static/img/pandas_white.png"  # Replace with a reliable URL

                # # if not os.path.exists(pandas_image_path):
                # #     try:
                # #         response = requests.get(pandas_image_url, stream=True)
                # #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                # #         with open(pandas_image_path, "wb") as f:
                # #             for chunk in response.iter_content(chunk_size=8192):
                # #                 f.write(chunk)
                # #         st.success("Pandas logo downloaded successfully!")
                # #     except requests.exceptions.RequestException as e:
                # #         st.error(f"Error downloading Pandas logo: {e}")
                # #     except Exception as e:
                # #         st.error(f"Error saving Pandas logo: {e}")
                st.image('asyncio.png', width=300)
                # st.write("Pandas")
            with col4:
                st.image("rio.jpg", width=300)
                # st.write("keras")
            with col5:
                import os
                import requests

                # tensorflow_image_path = "tf.png"
                # tensorflow_image_url = "https://www.tensorflow.org/images/tf_logo.png"  # Replace with a reliable URL

                # if not os.path.exists(tensorflow_image_path):
                #     try:
                #         response = requests.get(tensorflow_image_url, stream=True)
                #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                #         with open(tensorflow_image_path, "wb") as f:
                #             for chunk in response.iter_content(chunk_size=8192):
                #                 f.write(chunk)
                #         st.success("TensorFlow logo downloaded successfully!")
                #     except requests.exceptions.RequestException as e:
                #         st.error(f"Error downloading TensorFlow logo: {e}")
                #     except Exception as e:
                #         st.error(f"Error saving TensorFlow logo: {e}")
                st.image("streamlit.png", width=300)
                # st.write("tersorflow")
            with col6:
                
                st.image("nicegui.png", width=300)
                # st.write("Scikit-learn")
            with col7:
                
                st.image("bottle.jpg", width=300)
                # st.write("Statsmodels")
            with col8:
                import os
                import requests

                # plotly_image_path = "plotly.png"
                # plotly_image_url = "https://images.plot.ly/language-logos/python-logo-for-printing.png"  # Replace with a reliable URL

                # if not os.path.exists(plotly_image_path):
                #     try:
                #         response = requests.get(plotly_image_url, stream=True)
                #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                #         with open(plotly_image_path, "wb") as f:
                #             for chunk in response.iter_content(chunk_size=8192):
                #                 f.write(chunk)
                #         st.success("Plotly logo downloaded successfully!")
                #     except requests.exceptions.RequestException as e:
                #         st.error(f"Error downloading Plotly logo: {e}")
                #     except Exception as e:
                #         st.error(f"Error saving Plotly logo: {e}")
                st.image("falcon.png", width=300)
                # st.write("Plotly")
            with col9:
                # import os
                # import requests

                # pytorch_image_path = "pytorch.png"
                # # pytorch_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/PyTorch_logo.svg/320px-PyTorch_logo.svg.png"  # Replace with a reliable URL

                # # if not os.path.exists(pytorch_image_path):
                # #     try:
                # #         response = requests.get(pytorch_image_url, stream=True)
                # #         response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                # #         with open(pytorch_image_path, "wb") as f:
                # #             for chunk in response.iter_content(chunk_size=8192):
                # #                 f.write(chunk)
                # #         st.success("PyTorch logo downloaded successfully!")
                # #     except requests.exceptions.RequestException as e:
                # #         st.error(f"Error downloading PyTorch logo: {e}")
                # #     except Exception as e:
                # #         st.error(f"Error saving PyTorch logo: {e}")
                st.image('dash.png', width=300)
                # st.write("pytorch")

            with col10:
                st.image('taipy.png', width=300)


        with st.container(border=True):
            st.markdown("<div style='text-align: center;'><h4>DataScience with R</h4></div>", unsafe_allow_html=True)
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)
            with col1:
                import os
                import requests

                
                st.image("R_logo.png", width=300)
                # st.write("Flask")
            with col2:
                
                st.image("mlr3.png", width=300)
            with col3:
                st.image('causalimpact.png', width=500)
                # st.write("Pandas")
            with col4:
                st.image("torch.png", width=300)
                # st.write("keras")
            with col5:
                
                st.image("golem.png", width=300)
                # st.write("tersorflow")
            with col6:
                
                st.image("rcaret.png", width=300)
                # st.write("Scikit-learn")
            with col7:
                
                st.image("e1071.png", width=300)
                # st.write("Statsmodels")
            with col8:
                st.image('purrr.jpg', width=300)
               

            with col9:
                st.image('tidyquant.png', width=300)
                # st.write("pytorch")

            with col10:
                st.image('rlang.png', width=300)
            
            with col11:
                st.image('fastai.png', width=300)
                # st.write("pytorch")

        with st.container(border=True):
            st.markdown("<div style='text-align: center;'><h4>Deployment of Models in R</h4></div>", unsafe_allow_html=True)
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
            with col1:
                import os
                import requests

                
                st.image("plumber.png", width=300)
                # st.write("Flask")
            with col2:
                
                st.image("httr2.png", width=300)
            with col3:
                st.image('restfulr.png', width=300)
                # st.write("Pandas")
            with col4:
                st.image("vetiver.png", width=300)
                # st.write("keras")
            with col5:
                
                st.image("shiny.jpg", width=300)
                # st.write("tersorflow")
            with col6:
                
                st.image("rhino.png", width=300)
                # st.write("Scikit-learn")
            with col7:
                
                st.image("rsconnect.png", width=300)
                # st.write("Statsmodels")
            with col8:
                import os
                import requests
                st.image('reticulate.png', width=300)

            with col9:
                st.image('keras.png', width=300)
                # st.write("pytorch")

            with col10:
                st.image('book.png', width=300)

    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>Expertised in Machine Learning Operations and Engineering Tools</h4>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = st.columns(13)
        with col1:
            st.image("mlflow.png", width=300)
            # st.write("MLflow")
        with col2:
            st.image("kubeflow.png", width=300)
            # st.write("Kubeflow")
        with col3:
            st.image("airflow.png", width=300)
            # st.write("Airflow")   
        with col4:
            st.image("dagster.jpg", width=300)
            # st.write("Dagster")  
        with col5:
            st.image("prefect.png", width=300)
            # st.write("Prefect"
        with col6:
            st.image("dvc.png", width=300)
            # st.write("DVC")
        with col7:
            st.image("seldon.png", width=300)
            # st.write("Seldon")
        with col8:
            st.image("kserve.png", width=300)
            # st.write("KServe")
        with col9:
            st.image("tfx.png", width=500)
            # st.write("TFX")
        with col10:
            st.image("metaflow.png", width=300)
            # st.write("Metaflow")
        with col11:
            st.image("polyaxon.png", width=300)
            # st.write("Polyaxon")  
        with col12:
            st.image('wb.png', width=300)
            # st.write("Weights & Biases")
        with col13:
            st.image('kedro.png', width=300)
        # st.write("Weights & Biases")

    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>Full Stack Engineering and Development</h4>", unsafe_allow_html=True)
        col1, col2, col3,col4  = st.columns(4)
        with col2:
            st.image("mean_stack.png", width=400)
            # st.write("React")
        
        with col3:
            st.image("mern_stack.png", width=400)
            # st.write("Angular")


    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>Cloud Services</h4>", unsafe_allow_html=True)
        col1, col2, col3,col4,col5 = st.columns(5)
        with col2:
            st.image("aws.png", width=600)
            # st.write("AWS")
        with col3:
            st.image("azure.jpg", width=600)
            # st.write("Azure")
        with col4:
            st.image("gcp_cloud.png", width=600)
            # st.write("GCP")


    
    # with st.container(border=True):
    #     st.markdown("<h4 style='text-align: center;'>Expertised in Cloud ML, AI, DL, GEN AI</h4>", unsafe_allow_html=True)
    #     col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
    #     with col1:
    #         st.image("aws.png", width=300)
    #         # st.write("AWS")
    #     with col2:
    #         st.image("azure.png", width=300)
    #         # st.write("Azure")
    #     with col3:
    #         st.image("gcp.png", width=300)
    #         # st.write("GCP")  
    #     with col4:
    #         st.image("sagemaker.png", width=300)
    #         # st.write("IBM")
    #     with col5:
    #         st.image("azureml.png", width=300)
    #         # st.write("Oracle")
    #     with col6:
    #         st.image("gcp_ai_platform.png", width=300)
        
    #     with col7:
    #         st.image("gcp_genai.png", width=300)
    #         # st.write("GCP Gen AI")
    #     with col8:
    #         st.image("aws_genai_bedrock.png", width=300)
    #         # st.write("AWS Gen AI"
    #     with col9:
    #         st.image("azure_genai_openai.png", width=300)
    #         # st.write("Azure Gen AI")
    #     with col10:
    #         st.image("ibm_watson.png", width=300)
    #         # st.write("IBM Watson")
        
    # with st.container(border=True):
    #     st.markdown("<h4 style='text-align: center;'>Expertised in Data Engineering Tools</h4>", unsafe_allow_html=True)
    #     col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15 = st.columns(15)
    #     with col1:
    #         st.image("spark.png", width=300)
    #         # st.write("Spark")
    #     with col2:
    #         st.image("hadoop.png", width=300)
    #         # st.write("Hadoop")
    #     with col3:
    #         st.image("kafka.png", width=300)
    #         # st.write("Kafka")  
    #     with col4:
    #         st.image("flink.png", width=300)
    #         # st.write("Flink")  
    #     with col5:
    #         st.image("airflow.png", width=300)
    #         # st.write("Airflow")
    #     with col6:
    #         st.image("dbt.png", width=300)
    #         # st.write("DBT")
    #     with col7:
    #         st.image("snowflake.png", width=300)
    #         # st.write("Snowflake")
    #     with col8:
    #         st.image("bigquery.png", width=300)
    #         # st.write("BigQuery")
    #     with col9:
    #         st.image("redshift.png", width=300)
    #         # st.write("Redshift")
    #     with col10:
    #         st.image('databricks.png', width=300)

    #     with col11:
    #         st.image('hive.png', width=300)
    #         # st.write("Hive")
    #     with col12:
    #         st.image('presto.png', width=300)
    #         # st.write("Presto")
    #     with col13:
    #         st.image('amazon_emr.png', width=300)
    #     with col14:
    #         st.image('google_dataproc.png', width=300)
    #         # st.write("Google Dataproc")
    #     with col15:
    #         st.image('apache_beam.png', width=300)
    #         # st.write("Apache Beam")
        

    # with st.container(border=True):
    #     st.markdown("<h4 style='text-align: center;'>Front End Engineering</h4>", unsafe_allow_html=True)
    #     col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
    #     with col1:
    #         st.image("typescript.png", width=300)
    #         # st.write("HTML") 
    #     with col2:
    #         st.image("react.png", width=300)
    #         # st.write("CSS")
    #     with col3:
    #         st.image("javascript.png", width=300)
    #         # st.write("JavaScript")

    


with tab10:
    st.header("Acedemic Engagements")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Total Hours of Thought", value=2500, delta=50)
        # st.progress(value=75)
        # st.write("Hours of thought are calculated based on time spent in research modules.")
    with col2:
        st.metric(label="Number of Colleges Engaged", value=15, delta=5)
        
    style_metric_cards(background_color="#fff", border_size_px=1, border_color="#ccc", border_radius_px=5)

    # st.markdown("""
    # <style>
    # .container {
    #     border: 2px solid #ccc;
    #     border-radius: 10px;
    #     padding: 10px;
    #     margin-top: 20px;
    # }
    # </style>
    # """, unsafe_allow_html=True)

    with st.container(border=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("<div class='container'>", unsafe_allow_html=True)
            st.write("#### IIT Roorkee")
            st.image("iit-roorkee.png", width=150)
            # st.write("Description of IIT Madras data.")
            st.write("#### IIM Indore")
            st.image("iim indore.png", width=150)

            st.write("#### BITS Pilani Hyderabad")
            st.image("bits_pilani.png", width=150)
            # st.write("Description of University of Chicago data.")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='container'>", unsafe_allow_html=True)
            st.write("#### IIM Ahmedabad")
            st.image("iim_a.png", width=200)
            # st.write("Description of Stanford University data.")
            st.write("#### IIIT Hyderabad")
            st.image("IIIT_Hyderabad_Logo.jpg", width=250)
            # st.write("Description of MIT data.")
            st.write("#### Sant Gadge Baba Amravati University")
            st.image("Sant_Gadge_Baba_Amravati_University_logo.png", width=200)
            # st.write("Description of MIT data.")
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='container'>", unsafe_allow_html=True)
            st.write("#### IIT Hyderabad")
            st.image("iit_hyderabad.png", width=150)
            # st.write("Description of Harvard University data.")
            st.write("#### IIIT Delhi")
            st.image("indraprastha-institute-of-information-technology-iiit-delhi-logo.png", width=150)
            st.write("#### Guru Nanak Educational Society")
            st.image("gurunanak.jpg", width=200)
            # st.write("Description of Yale University data.")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='container'>", unsafe_allow_html=True)
            st.write("#### Austin Texas University")
            st.image("University_of_Texas_at_Austin_seal.svg.png", width=150)
            # st.write("Description of Princeton University data.")
            st.write("#### Vignan Institute of Technology")
            st.image("vignan.jpg", width=150)
            # st.write("Description of University of Oxford data.")
            st.write("#### Manipal University Bangalore")
            st.image("manipal.jpg", width=170)
            
            st.markdown("</div>", unsafe_allow_html=True)

        with col5:
            st.markdown("<div class='container'>", unsafe_allow_html=True)
            st.write("#### CMR College of Engineering")
            st.image("cmr.jpg", width=150)
            # st.write("Description of University of Cambridge data.")
            st.write("#### Geetanjali University")
            st.image("geetanjali.jpg", width=150)
            st.write("#### KIIT University")
            st.image("KIIT-Logo.png", width=200)
            # st.write("Description of University of California, Berkeley data.")
            st.markdown("</div>", unsafe_allow_html=True)



with tab4:
    st.header("Certificates")

    roles = load_skills_data()

    # Calculate certificate counts based on skills
    deep_learning_certs = 0
    machine_learning_certs = 0
    statistics_certs = 0
    generative_ai_certs = 0
    ai_engineering_certs = 0
    mlops_certs = 0

    for role in roles:
        if role["name"] == "AI Engineer":
            deep_learning_certs = 7
            generative_ai_certs = 13 # Assuming all skills in AI Engineer are related to Generative AI
            ai_engineering_certs = len(role["skills"])
        elif role["name"] == "ML Engineer":
            machine_learning_certs = 9
        elif role["name"] == "Data Science":
            statistics_certs = 7
        elif role["name"] == "MLOps Engineer":
            mlops_certs = len(role["skills"])

    col1, col2, col3 = st.columns(3)
    with col1:
        display_metric("Deep Learning", deep_learning_certs, col1)
        display_metric("Machine Learning", machine_learning_certs, col1)
    with col2:
        display_metric("Statistics", statistics_certs, col2)
        display_metric("AI and ML Engineer", ai_engineering_certs, col2)
    with col3:
        display_metric("Generative AI", generative_ai_certs, col3)
        display_metric("MLOps", mlops_certs, col3)
    
    with st.expander("List of Generative AI Certifications", expanded=True):
        st.subheader("1. Google’s Agentic AI Primer")
        st.write("Key Topics: Agentic AI, Prompt Engineering, Model Fine-Tuning, Deployment Strategies")
        st.subheader("2. Generative AI with PyTorch")
        st.write("Key Topics: GANs, VAEs, Diffusion Models, Text-to-Image Generation")
        st.subheader("3. Generative AI with Hugging Face")
        st.write("Key Topics: Transformers, Text Generation, Image Generation, Fine-Tuning Models")
        st.subheader("4. IBM AI Engineering Professional Certificate")
        st.write("Key Topics: Generative AI, Machine Learning, Deep Learning, Model Deployment")
        st.subheader("5. Deep Learning for Generative AI")
        st.write("Key Topics: GANs, VAEs, Diffusion Models, Text-to-Image Generation")
        st.subheader("6. Generative AI with TensorFlow")
        st.write("Key Topics: GANs, VAEs, Diffusion Models, Text-to-Image Generation")
        st.subheader("8. MIT AGI: Autonomous Agents Research")
        st.write("Key Topics: Autonomous Agents, Reinforcement Learning, Multi-Agent Systems, AI Ethics")
        st.subheader("9. IBM Generative AI Professional Certificate")
        st.write("Key Topics: Generative AI, Machine Learning, Deep Learning, Model Deployment")
        st.subheader("10. Generative AI with Large Language Models")
        st.write("Key Topics: Large Language Models, Text Generation, Fine-Tuning, Deployment Strategies")
        st.subheader("11. DeepLearning.AI AI Agentic Design Patterns")
        st.write("Key Topics: Agentic AI, Prompt Engineering, Model Fine-Tuning, Deployment Strategies")
        st.subheader("12. Reinforcement Learning")
        st.write("Key Topics: Reinforcement Learning, Multi-Agent Systems, AI Ethics, Model Deployment")
        st.subheader("13.LangChain Certification")
        st.write("Key Topics: LangChain, Prompt Engineering, Model Fine-Tuning, Deployment Strategies")



    with st.expander("List of Deep Learning Certifications"):
        st.subheader("1. Deep Learning Specialization - DeepLearning.AI")
        st.write("Key Topics: CNNs, RNNs, Transformers, NLP, Computer Vision")
        st.subheader("2. TensorFlow Developer Certificate - Google")
        st.write("Key Topics: TensorFlow, Keras, Model Deployment, Transfer Learning")
        st.subheader("3. Advanced Computer Vision with TensorFlow")
        st.write("Key Topics: Object Detection, Image Segmentation, GANs")
        st.subheader("4. Natural Language Processing with Deep Learning - Stanford University")
        st.write("Key Topics: RNNs, LSTMs, Attention Mechanisms, Transformers")
        st.subheader("5. Stanford CS330: Deep Multi-Task and Meta Learning")
        st.write("Key Topics: Multi-Task Learning, Meta Learning, Optimization Techniques")
        st.subheader("6. Deep Learning for Computer Vision with Python - PyImageSearch")
        st.write("Key Topics: Image Classification, Object Detection, Image Segmentation")
        st.subheader("7. Fast.ai Practical Deep Learning for Coders")
        st.write("Key Topics: Practical Deep Learning, Transfer Learning, Fine-Tuning Models")



    with st.expander("List of Machine Learning Certifications"):
        st.subheader("1. edX MITx: Machine Learning with Python")
        st.write("Key Topics: Supervised Learning, Unsupervised Learning, Model Evaluation")
        st.subheader("2. Stanford CS229: Machine Learning")
        st.write("Key Topics: Supervised Learning, Unsupervised Learning, Neural Networks")
        st.subheader("3. Cloud Based Machine Learning - Google Cloud")
        st.write("Key Topics: Cloud ML, BigQuery ML, AutoML, Model Deployment")
        st.subheader("4. AWS Certified Machine Learning - Specialty")
        st.write("Key Topics: AWS ML Services, Model Deployment, Data Engineering")
        st.subheader("5. Azure Data Scientist Associate")
        st.write("Key Topics: Azure ML, Model Deployment, Data Preparation")
        st.subheader("7. IBM Machine Learning Professional Certificate")
        st.write("Key Topics: Supervised Learning, Unsupervised Learning, Model Evaluation")
        st.subheader("8. Google Machine Learning Crash Course")
        st.write("Key Topics: Supervised Learning, Unsupervised Learning, TensorFlow Basics")
        st.subheader("9. Microsoft Azure AI Fundamentals (AI-900)")
        st.write("Key Topics: Azure AI Services, Machine Learning Basics, AI Concepts")


    with st.expander("List of Statistics Certifications"):
        st.subheader("1. Coursera: Statistics with Python Specialization - University of Michigan")
        st.write("Key Topics: Descriptive Statistics, Inferential Statistics, Regression Analysis")
        st.subheader("2. edX: Data Science: Probability - Harvard University")
        st.write("Key Topics: Probability Theory, Random Variables, Distributions")
        st.subheader("3. Basian Statistics")
        st.write("Key Topics: Bayesian Inference, Prior and Posterior Distributions, Markov Chain Monte Carlo")
        st.subheader("4. IBM Advanced Data Science with Time Series")
        st.write("Key Topics: Time Series Analysis, Forecasting, ARIMA Models")
        st.subheader("5. Stanford's Introduction to Statistical Learning")
        st.write("Key Topics: Linear Regression, Classification, Resampling Methods")
        st.subheader("6. Statistics with R Specialization - Duke University")
        st.write("Key Topics: Descriptive Statistics, Inferential Statistics, Regression Analysis")
        st.subheader("7. MITx: Probability - The Science of Uncertainty and Data")
        st.write("Key Topics: Probability Theory, Random Variables, Distributions")

    with st.expander("List of AI and ML Engineer Certifications"):
        st.subheader("1. AI Engineer Professional Certificate - IBM")
        st.write("Key Topics: AI Concepts, Machine Learning, Deep Learning, Model Deployment")
        st.subheader("2. AI and Machine Learning for Coders - Google")
        st.write("Key Topics: AI Concepts, Machine Learning, Deep Learning, Model Deployment")
        st.subheader("3. AI Engineering with TensorFlow - Google Cloud")
        st.write("Key Topics: TensorFlow, Keras, Model Deployment, Transfer Learning")
        st.subheader("4. AI and Machine Learning for Business - Coursera")
        st.write("Key Topics: AI Concepts, Machine Learning, Business Applications")
        st.subheader("5. AI and Machine Learning for Coders - Fast.ai")
        st.write("Key Topics: Practical AI, Transfer Learning, Fine-Tuning Models") 
    
    with st.expander("List of MLOps Certifications"):
        st.subheader("1. MLOps Specialization - Coursera")
        st.write("Key Topics: MLOps Principles, Model Deployment, CI/CD for ML")
        st.subheader("2. MLOps with TensorFlow - Google Cloud")
        st.write("Key Topics: TensorFlow, Keras, Model Deployment, Transfer Learning")
        st.subheader("3. MLOps Engineer Professional Certificate - IBM")
        st.write("Key Topics: MLOps Principles, Model Deployment, CI/CD for ML")
        st.subheader("4. MLOps Fundamentals - Microsoft Azure")
        st.write("Key Topics: Azure ML, Model Deployment, Data Preparation")
        st.subheader("5. MLOps with AWS - Amazon Web Services")
        st.write("Key Topics: AWS ML Services, Model Deployment, Data Engineering")

    

            
    
    # with st.expander("List of Deep Learning Certifications", expanded=True):
    #     st.header("This content will be visible by default")
    #     st.button("Example button")
    
    # with st.expander("List of Deep Learning Certifications", expanded=True):
    #     st.header("This content will be visible by default")
    #     st.button("Example button")


with tab7:
    st.header("Teaching Experience")

    st.subheader("Teaching Philosophy")
    st.write(
        """
        My teaching philosophy centers around creating engaging and interactive learning experiences that empower students to develop a deep understanding of complex concepts. I strive to foster a collaborative environment where students feel comfortable asking questions, exploring new ideas, and applying their knowledge to real-world problems.
        """
    )

    st.subheader("Subject Areas")

    # Python
    with st.expander("Python"):
        st.write(
            """
            **Courses Taught:** Introduction to Python Programming, Data Analysis with Python, Machine Learning with Python

            **Technologies Used:** Python, Jupyter Notebook, Pandas, NumPy, Scikit-learn

            **Student Outcomes:** Students successfully completed projects involving data analysis, machine learning model development, and web application creation. Average student satisfaction score: 4.5/5.
            """
        )

    # Deep Learning
    with st.expander("Deep Learning"):
        st.write(
            """
            **Courses Taught:** Deep Learning Fundamentals, Convolutional Neural Networks, Recurrent Neural Networks

            **Technologies Used:** TensorFlow, Keras, PyTorch, CUDA

            **Student Outcomes:** Students built deep learning models for image classification, natural language processing, and time series forecasting. Project success rate: 90%.
            """
        )

    # NLP
    with st.expander("Natural Language Processing (NLP)"):
        st.write(
            """
            **Courses Taught:** Natural Language Processing with Python, Text Mining, Sentiment Analysis

            **Technologies Used:** NLTK, SpaCy, Gensim, Transformers

            **Student Outcomes:** Students developed NLP applications for text classification, sentiment analysis, and machine translation.
            """
        )

    # High-Performance Computation
    with st.expander("High-Performance Computation"):
        st.write(
            """
            **Courses Taught:** Introduction to High-Performance Computing, Parallel Programming with Python

            **Technologies Used:** NumPy, SciPy, Numba, Dask

            **Student Outcomes:** Students optimized Python code for high-performance computing environments.
            """
        )

    # AI Engineering
    with st.expander("Artificial Intelligence Engineering"):
        st.write(
            """
            **Courses Taught:** AI Engineering Fundamentals, Building AI-Powered Applications

            **Technologies Used:** Python, TensorFlow, Keras, Cloud Platforms (AWS, Azure, GCP)

            **Student Outcomes:** Students designed and deployed AI-powered applications using various cloud platforms.
            """
        )

    # MLOps
    with st.expander("Machine Learning Operations (MLOps)"):
        st.write(
            """
            **Courses Taught:** Introduction to MLOps, CI/CD for Machine Learning, Model Deployment and Monitoring

            **Technologies Used:** Docker, Kubernetes, MLflow, Jenkins, Prometheus

            **Student Outcomes:** Students built and deployed machine learning pipelines using MLOps principles.
            """
        )

    # R
    with st.expander("R"):
        st.write(
            """
            **Courses Taught:** Introduction to R Programming, Statistical Modeling with R

            **Technologies Used:** R, RStudio, ggplot2

            **Student Outcomes:** Students performed statistical analysis and data visualization using R.
            """
        )

    st.subheader("Teaching Institutes")

    # Great Learning
    st.markdown("#### Great Learning  <img src='https://img.icons8.com/color/30/000000/university.png'>", unsafe_allow_html=True)
    with st.expander("See Details"):
        st.write(
            """
            **Course Name:** Data Science and Machine Learning Bootcamp

            **Duration:** 6 Months

            **Role:** Instructor
            """
        )

    # UpGrad
    st.markdown("#### UpGrad  <img src='https://img.icons8.com/color/30/000000/university.png'>", unsafe_allow_html=True)
    with st.expander("See Details"):
        st.write(
            """
            **Course Name:** Advanced Machine Learning and AI Program

            **Duration:** 9 Months

            **Role:** Mentor
            """
        )

    # Jigsaw Academy
    st.markdown("#### Jigsaw Academy  <img src='https://img.icons8.com/color/30/000000/university.png'>", unsafe_allow_html=True)
    with st.expander("See Details"):
        st.write(
            """
            **Course Name:** Data Science Specialization

            **Duration:** 12 Months

            **Role:** Guest Lecturer
            """
        )

    # IIT Roorkee
    st.markdown("#### IIT Roorkee  <img src='https://img.icons8.com/color/30/000000/university.png'>", unsafe_allow_html=True)
    with st.expander("See Details"):
        st.write(
            """
            **Course Name:** Deep Learning Workshop

            **Duration:** 2 Days

            **Role:** Workshop Facilitator
            """
        )

    # IIM Indore (IIM I)
    st.markdown("#### IIM Indore (IIM I)  <img src='https://img.icons8.com/color/30/000000/university.png'>", unsafe_allow_html=True)
    with st.expander("See Details"):
        st.write(
            """
            **Course Name:** Business Analytics Program

            **Duration:** 1 Week

            **Role:** Visiting Faculty
            """
        )

    # IIM Ahmedabad (IIM A)
    st.markdown("#### IIM Ahmedabad (IIM A)  <img src='https://img.icons8.com/color/30/000000/university.png'>", unsafe_allow_html=True)
    with st.expander("See Details"):
        st.write(
            """
            **Course Name:** AI for Business Leaders

            **Duration:** 3 Days

            **Role:** Guest Speaker
            """
        )

    st.subheader("Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Students Taught", value=9876)
    with col2:
        st.metric(label="Average Satisfaction Score", value="4.6/5")
    with col3:
        st.metric(label="Project Success Rate", value="85%")

    st.subheader("Call to Action")
    st.write(
        """
        For a detailed teaching portfolio or further information, please contact me.
        """
    )

    # Request Course Enrollment Button
    st.subheader("Request Course Enrollment")
    user_name = st.text_input("Your Name", key="course_enrollment_name")
    user_email = st.text_input("Your Email", key="course_enrollment_email")
    course_interest = st.selectbox("Interested in Courses related to:", ["Data Science", "AI", "Machine Learning", "Deep Learning", "MLOps", "DevOps"], key="course_enrollment_interest")
    user_message = st.text_area("Optional Message", key="course_enrollment_message")

def display_metric(label, value, column):
    """Displays a metric in the specified column."""
    with column:
        st.metric(label=label, value=value)

def send_email(subject, message, receiver_email):
    """Sends an email using the provided subject and message."""
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "kiran90429@gmail.com"  # Admin's email
    password = input("Type your password and press enter:")

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

        st.write(f"**Contact Person:** {contact_person}")
        st.write(f"**Phone Number:** {phone_number}")
        st.write(f"**Preferred Appointment Dates/Times:** {preferred_dates}")

    st.subheader("Training Metrics")

    # Dummy Data (Replace with actual data source)
    total_companies_trained = 150
    companies_trained_this_month = 15
    companies_trained_this_quarter = 40
    month_over_month_trend = 0.10  # 10% increase
    quarter_over_quarter_trend = -0.05  # 5% decrease
    last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Companies Trained", value=total_companies_trained)
        st.metric(label="Companies Trained This Month", value=companies_trained_this_month, delta=f"{month_over_month_trend:.2%}", delta_color="normal")
    with col2:
        st.metric(label="Companies Trained This Quarter", value=companies_trained_this_quarter, delta=f"{quarter_over_quarter_trend:.2%}", delta_color="inverse")

    # Data Visualization (Dummy Data)
    import pandas as pd
    import numpy as np
    chart_data = pd.DataFrame(
        np.random.randn(12, 1),
        columns=['Companies Trained'],
        index=pd.date_range(start='2024-07-01', periods=12, freq='MS')
    )
    st.line_chart(chart_data)

    st.write(f"Data Source: Dummy Data")
    st.write(f"Last Updated: {last_updated}")

    st.subheader("Data Science Course Catalog")

    # Course 1
    with st.container():
        st.subheader("Data Science with Python")

with tab8:
    st.header("Consulting Engagements")

    # Engagement 1
    with st.container():
        st.subheader("HotStar - Predictive Maintenance System")
        st.write("**Description:** Developed a predictive maintenance system for hotstar Corp's manufacturing equipment using machine learning. The system predicts equipment failures, allowing for proactive maintenance and reducing downtime.")
        st.write("**Engagement Dates:** 2024-01-15 to 2024-07-31")
        st.write("**Role:** Lead Data Scientist")
        st.write("**Technologies Used:** Python, Scikit-learn, TensorFlow, AWS")
        st.write("**Team Members:** Satya from US")
        st.write("**Key Deliverables:** Predictive model, maintenance schedule, dashboard")
        st.write("Client Testimonial: 'The predictive maintenance system has significantly reduced our equipment downtime and saved us thousands of dollars.'")
        st.write("**Project Impact:** Reduced equipment downtime by 15%, saving $50,000 in maintenance costs.")
        st.markdown("---")

    # Engagement 2
    with st.container():
        st.subheader("Walmart - Customer Segmentation and Targeting")
        st.write("**Description:** Implemented a customer segmentation and targeting strategy for Beta Industries using data analytics and machine learning. The strategy identified key customer segments and developed targeted marketing campaigns to increase sales and customer loyalty.")
        st.write("**Engagement Dates:** 2023-09-01 to 2024-03-31")
        st.write("**Role:** Data Science Consultant")
        st.write("**Technologies Used:** R, SQL, Tableau, Google Analytics")
        st.write("**Team Members:** David Lee, Sarah Chen")
        st.write("**Key Deliverables:** Customer segmentation model, marketing campaign plan, performance reports")
        st.write("Client Testimonial: 'The customer segmentation and targeting strategy has significantly improved our marketing ROI.'")
        st.write("**Project Impact:** Increased sales by 10% and customer loyalty by 5%.")
        st.markdown("---")

    # Engagement 3
    with st.container():
        st.subheader("pfizer Healthcare - Medical Image Analysis")
        st.write("**Description:** Developed a computer vision system for medical image analysis to assist radiologists in detecting diseases. The system analyzes X-rays and MRIs to identify potential anomalies, improving diagnostic accuracy and reducing the workload of radiologists.")
        st.write("**Engagement Dates:** 2024-11-01 to 2025-02-31")

    # Request Appointment Button

with tab9:
    st.header("Corporate Training")

    st.subheader("Training Metrics")

    # Dummy Data (Replace with actual data source)
    total_companies_trained = 9
    companies_trained_this_month = 1
    companies_trained_this_quarter = 2
    month_over_month_trend = 0.10  # 10% increase
    quarter_over_quarter_trend = -0.05  # 5% decrease
    last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Companies Trained", value=total_companies_trained)
        st.metric(label="Companies Trained This Month", value=companies_trained_this_month, delta=f"{month_over_month_trend:.2%}", delta_color="normal")
    with col2:
        st.metric(label="Companies Trained This Quarter", value=companies_trained_this_quarter, delta=f"{quarter_over_quarter_trend:.2%}", delta_color="inverse")

    # Data Visualization (Dummy Data)
    import pandas as pd
    import numpy as np
    chart_data = pd.DataFrame(
        np.random.randn(12, 1),
        columns=['Companies Trained'],
        index=pd.date_range(start='2024-07-01', periods=12, freq='MS')
    )
    st.line_chart(chart_data)

    # st.write(f"Data Source: Dummy Data")
    # st.write(f"Last Updated: {last_updated}")

    st.subheader("Courses trained along with companies details.")

    # Course 1
    with st.container():
        st.subheader("Data Science with Python")
        st.write("**Description:** A comprehensive introduction to data science using Python.")
        st.write("**Training Hours:** 40")
        st.write("**Company:** SG Analytics")
        st.markdown("---")

    # Course 2
    with st.container():
        st.subheader("Data Science with R")
        st.write("**Description:** A comprehensive introduction to data science using R.")
        st.write("**Training Hours:** 40")
        st.write("**Company:** UNext/Deloitte")
        st.markdown("---")

    # Course 3
    with st.container():
        st.subheader("Statistics for Data Science ")
        st.write("**Description:** A deep dive into statistical concepts for data science.")
        st.write("**Training Hours:** 30")
        st.write("**Company:** Cognizant under Jigsaw Engagement")
        st.markdown("---")

    # Course 4
    with st.container():
        st.subheader("Deep Learning Fundamentals For SG Analytics")
        st.write("**Description:** An introduction to deep learning concepts and techniques.")
        st.write("**Training Hours:** 50")
        st.write("**Company:** SG Analytics")
        st.markdown("---")

    # Course 5
    with st.container():
        st.subheader("Generative AI")
        st.write("**Description:** An introduction to Generative AI models and applications.")
        st.write("**Training Hours:** 60")
        st.write("**Company:** SG Analytics")
        st.markdown("---")

with tab3:
    st.header("Academic Journey")
    from streamlit_extras.metric_cards import style_metric_cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="All Competitive Examinations in 2010", value="Top 10%")
    col2.metric(label="Number of Master Degrees", value="3+")
    col3.metric(label="Total Patents Filed", value="7+")
    col4.metric(label="Total Research Papers Published", value="20+")
    style_metric_cards()
    with st.container(border=True):
        st.subheader("IIT Madras, Chennai, IN")
        col1,col2,col3, col4 = st.columns(4)
        with col1:
            
            st.write("Degree: B. Tech")
            st.write("Major: Engineering Disciplines")
            st.write("Minor: Data Analytics")
            st.write("Graduation Date: Jun 2014")
            st.write("GPA: 7.1/10")
        try:
            col4.image("iit_madras_logo.png", width=200)
        except FileNotFoundError:
            col4.write("Logo not found")

        st.markdown(
            """
            <style>
            @media (max-width: 768px) {
                .container3 {
                    width: 100%; /* Full width on smaller screens */
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        
    with st.container(border=True):
        st.subheader("University of Chicago Graham School, Chicago, US")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            
            st.write("Degree: Post Graduate Program")
            st.write("Major: Data Science and Machine Learning")
            st.write("Graduation Date: May 2019")
            st.write("GPA: 3.5/4.0")
        try:
            col4.image("university_of_chicago.png", width=200)
        except FileNotFoundError:
            col4.write("Logo not found")

        st.markdown(
            """
            <style>
            @media (max-width: 768px) {
                .container3 {
                    width: 100%; /* Full width on smaller screens */
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.subheader("Birla Institute of Technology and Science, Pilani, IN")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            
            st.write("Degree: M. Tech")
            st.write("Major: Software Systems")
            st.write("Specialization: Data Sciences and Cloud Computing")
            st.write("Graduation Date: Jun 2021")
            st.write("GPA: 7.5/10")

        try:
            col4.image("BITS_Pilani-Logo.png", width=200)
        except FileNotFoundError:
            col4.write("Logo not found")

        st.markdown(
            """
            <style>
            @media (max-width: 768px) {
                .container3 {
                    width: 100%; /* Full width on smaller screens */
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )




    st.header("Academic Projects")
    container4 = st.container(border=True)
    container4.write("""
    * Developed a machine learning project for Predicting prepayments and defaults for privatized banks.
    * Wake word detection model for the custom word HIFREEDA
    * Built a CNN based image classifier, which will be able to distinguish between different handwritten Devanagari characters.
    * Distinguishing between Hillary Clinton and Trump tweets.
    * Developed a streaming analytics application to analyze movie recommendations.
    * Music Generation using Deep Learning.
    * Built and implement and machine learning NLP based project which computes sentiment for an incoming review in amazon online website.
    """)

    st.header("Patents and Research Papers")
    # Timeline Data
    timeline_data = [
        {"Name": "Demand Sensing and Forecasting US 17/110,992",
            "link": "https://drive.google.com/file/d/1nvOsIO00G8Dwt5cNJjWemTGwtgbVVF9J/view?usp=sharing"},
        {"Name": "Psuedo Random Forest",
            "link": "https://drive.google.com/file/d/1w7kwxIumpsqam3IcCTGeLmVAEUJI1m9Z/view?usp=sharing"},
        {"Name": "Noise Detection and Removal",
            "link": "https://drive.google.com/file/d/1h2EtSHWM1qaScPZ9eBScVoW39nEFi3j8/view?usp=sharing"},
        {"Name": "Generative Sampling Technique",
            "link": "https://drive.google.com/file/d/199Vm0aMKafgu_tzBE5hnt0OhcpehH4Ym/view?usp=sharing"},
        {"Name": "Automated Extraction of Entities from claim documents",
            "link": "https://drive.google.com/file/d/1w7kwxIumpsqam3IcCTGeLmVAEUJI1m9Z/view?usp=sharing"},
        {"Name": "Semi Supervised Gyan",
            "link": "https://drive.google.com/file/d/1h2EtSHWM1qaScPZ9eBScVoW39nEFi3j8/view?usp=sharing"}
    ]
    container5 = st.container(border=True)
    for item in timeline_data:
        container5.write(f'[{item["Name"]}]({item["link"]})')

with tab8:
    st.header("Consulting")
    st.subheader("Consulting Services Offered")
    

with tab1:
    st.header("Kiran Kumar Kandula")
    cols = st.columns(3)

    with cols[0]:
        ui.metric_card(title="Total Projects Deployed", content="40+", description="",key="card1")
    with cols[1]:
        ui.metric_card(title="Patents Published", content="7+", description="", key="card2")
    with cols[2]:
        ui.metric_card(title="Total Working Experience", content="10+", description="", key="card3")

    # st.write(ui.metric_card)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Principal Data Scientist")
        with col3:
            st.subheader("Micron Technology, Inc.")
        
 

        st.write("""
        Data Scientist and Full Stack Developer with 10+ years of experience. Expert in Data Science, Machine Learning, Big Data, Computer Vision, Deep Learning, Python, R, Java, .Net, SQL, Cloud Solutions, LLM, and more.

        **Skills:** Data Analysis, Feature Engineering, Big Data, Machine Learning, Deep Learning, Python, R, Java, .Net, SQL, Cloud Solutions, LLM, and more.

        **Experience:** Principal Data Scientist at Micron Technology, Microsoft Technology Consultant at PwC India, Data Scientist at GEP Solutions, and more.

        **Education:** B.Tech from IIT Madras, Post Graduate Program from University of Chicago Graham School, M.Tech from Birla Institute of Technology and Science, Pilani.
        """)
    try:
        with open("about_me.txt", "r") as f:
            about_me_content = f.read()

        st.markdown("""
            <style>
            .highlighted-yellow {
                background-color: #FFFF00; /* Yellow */
                animation: pulse 2s infinite alternate;
                padding: 2px;
                border-radius: 5px;
            }
            .highlighted-green {
                background-color: #90EE90; /* LightGreen */
                animation: pulse 2s infinite alternate;
                padding: 2px;
                border-radius: 5px;
            }
            .highlighted-blue {
                background-color: #ADD8E6; /* LightBlue */
                animation: pulse 2s infinite alternate;
                padding: 2px;
                border-radius: 5px;
            }
            @keyframes pulse {
                from { transform: scale(1); }
                to { transform: scale(1.05); }
            }
            </style>
        """, unsafe_allow_html=True)

        about_me_content = about_me_content.replace("full-stack data scientist", "<span class='highlighted-yellow'>full-stack data scientist</span>")
        about_me_content = about_me_content.replace("machine learning lifecycle", "<span class='highlighted-green'>machine learning lifecycle</span>")
        about_me_content = about_me_content.replace("data engineering", "<span class='highlighted-blue'>data engineering</span>")
        about_me_content = about_me_content.replace("cloud platforms", "<span class='highlighted-yellow'>cloud platforms</span>")
        about_me_content = about_me_content.replace("machine learning skillset", "<span class='highlighted-green'>machine learning skillset</span>")
        about_me_content = about_me_content.replace("front-end development skills", "<span class='highlighted-blue'>front-end development skills</span>")

        st.markdown(f"<div style='font-size: 16px;'>{about_me_content}</div>", unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Could not load about_me.txt. Please ensure the file exists.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

with tab2:
    st.header("Professional Journey")
    cols = st.columns(4)

    with cols[0]:
        ui.metric_card(title="Total Projects Deployed", content="40+", description="",key="card4")
    with cols[1]:
        ui.metric_card(title="Patents Published", content="7+", description="", key="card5")
    with cols[2]:
        ui.metric_card(title="Total Working Experience", content="10+", description="", key="card6")

    with cols[3]:
        ui.metric_card(title="Client Engagements", content="30+", description="", key="card7")

    # Timeline Data
    timeline_data = [
        {"Name": "Micron Technology, Inc.",
            "Position": "Principal Data Scientist",
            "start": datetime(2021, 10, 1),
            "end": datetime(2025, 5, 30),
            "description": "Quantum Machine Learning suite for variation identification. Code Assistant Tool. Reinforcement learning based Engine for Tool Trouble shooting. Built advanced statistics/data-science models.",
            "logo": "BITS_Pilani-Logo.png"},
        {"Name": "GEP SOLUTIONS",
            "Position": "Data Scientist",
            "start": datetime(2019, 8, 1),
            "end": datetime(2025, 5, 30),
            "description": "Built NLP based supplier intent identifier. Built a text summary extractor. Built promotional modelling for demand planners. Built a large-scale Demand Planning and Demand Sensing pipeline.",
            "logo": "university_of_chicago.png"},
        {"Name": "HIGH RADIUS PVT LIMITED",
            "Position": "Data Scientist",
            "start": datetime(2017, 7, 1),
            "end": datetime(2025, 5, 30),
            "description": "Proactively built and automated the claim extraction process. Implemented GANs, Semi Supervised GANs. Customer Segmentation model. Implemented “HiFreeda” Wake word detection model.",
            "logo": "University-of-Chicago-Seal.png"},
        {"Name": "BITMIN INFO SYSTEMS",
            "Position": "Data Scientist and Full Stack Developer",
            "start": datetime(2016, 4, 1),
            "end": datetime(2025, 5, 30),
            "description": "Developed apps, worked on the back-end and front-end. Created Machine Learning-based tools. Built a webservice in .net which extracts the payments from quick books.",
            "logo": "iit_madras_logo.png"},
        {"Name": "PIXENTIA SOLUTIONS",
            "Position": "Software Engineer-I",
            "start": datetime(2015, 2, 1),
            "end": datetime(2025, 5, 30),
            "description": "Collaborated with team members to create applications system analysis based upon client requirements. Created tools like resume parser, Developed digital badge online representation of a skill.",
            "logo": "bitspilani_logo.png"},
        {"Name": "GYAN DATA Pvt. LTD",
            "Position": "Software Engineer Trainee",
            "start": datetime(2012, 11, 1),
            "end": datetime(2025, 5, 30),
            "description": "Developed and hosted apps like remote scilab for computations in chemical engineering labs remotely, worked on enterprise level applications like sandman with machine learning as core.",
            "logo": "BITS_Pilani-Logo.png"},
    ]

    with st.container(border=True):
        timeline_data.reverse()
        # Create Figure
        fig = go.Figure()
        for i, item in enumerate(timeline_data):
            fig.add_trace(go.Scatter(
                x=[item["start"], item["end"]],
                y=[item["Name"], item["Name"]],
                mode="lines+markers",
                marker=dict(size=12),
                line=dict(width=4),
                name=item["Name"],
                text=item["Position"],
                hovertemplate=f"<b>{item['Name']}<br><img src='{item['logo']}' width='50' alt='{item['Name']} logo'><br>{item['Position']}<br>{item['start'].strftime('%Y-%m-%d')} - {item['end'].strftime('%Y-%m-%d')}<br>{item['description']}<extra></extra>"
            ))
        fig.update_layout(
            title="Professional Journey Timeline",
            xaxis_title="Time",
            yaxis=dict(
                showgrid=False,
                showline=False,
                zeroline=False,
            ),
            hovermode="closest"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Data Table
    import pandas as pd
    import pandas as pd

    data = {
        "Company Name": ["Micron Technology, Inc.", "GEP SOLUTIONS", "HIGH RADIUS PVT LIMITED", "BITMIN INFO SYSTEMS", "PIXENTIA SOLUTIONS", "GYAN DATA Pvt. LTD"],
        "Responsibilities": [
            """
            Quantum Machine Learning (QML Project)\n
            Code Assistant Tool Development\n
            Reinforcement Learning Applications
            """,
            """
            Natural Language Processing (NLP)\n
            Text Summary Extractor Development\n
            Promotional Modeling<br>
            Demand Planning
            """,
            """
            Claim Extraction\n
            Generative Adversarial Networks (GANs)\n
            Customer Segmentation<br>
            Wake word detection
            """,
            """
            App Development\n
            Machine Learning Tools Development\n
            Webservice Development
            """,
            """
            Application System Analysis\n
            Resume Parser Development\n
            Digital Badge Development
            """,
            """
            Enterprise-Level Application Development
            """
        ],
        "Clients": ["Micron FABs( Manufacturing)", "NEXXE (GEP product), Adidas, Nike, MC Donalds", "P&G", "Highradius, Hunamakia, CeeChat", "Sumtotalsystems", "Sandman"]
    }

    df = pd.DataFrame(data)
    st.data_editor(df, height=400, column_config={
        "Company Name": st.column_config.Column(width="large"),
        "Responsibilities": st.column_config.Column(width="large"),
        "Clients": st.column_config.Column(width="medium"),
    },
    )

