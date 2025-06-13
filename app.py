import streamlit as st
import time  # Import the time module

if "page_configured" not in st.session_state:
    st.set_page_config(layout="wide")
    st.session_state["page_configured"] = True

# if "app_loaded" not in st.session_state:
#     st.session_state["app_loaded"] = False
if "app_loaded" not in st.session_state:
    st.session_state["app_loaded"] = False

if not st.session_state["app_loaded"]:
    with st.spinner("Loading..."):
        st.markdown("""
            <div style="border: 4px solid rgba(0, 0, 0, 0.1); border-left-color: #007bff; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite;"></div>
            <style>
            @keyframes spin {
              0% {
                transform: rotate(0deg);
              }
              100% {
                transform: rotate(360deg);
              }
            }
            </style>
        """, unsafe_allow_html=True)
        # Simulate loading time (replace with actual loading logic)
        time.sleep(2)
        st.session_state["app_loaded"] = True
        st.rerun()

# Add a spinner
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



def display_metric(label, value, column):
    """Displays a metric in the specified column."""
    with column:
        st.metric(label=label, value=value)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["About Me", "Professional Journey", "Academic Journey", "Certifications", "Projects", "Expertise", "Teaching Experience", "Consulting", "Corporate Training", "Academic Engagements"])


project_data = {
    "MICRON": [
        "Published more than 10+ internal papers (adjudged in top 2 globally), Filed 5+ invention disclosures and won multiple awards like IDEA of the Quarter, Cultural Championships, and Best Performer awards in the team.",
        "Quantum Machine Learning suite for variation identification in the semiconductor manufacturing process. <span style=\"color: green;\">**↑ Improved variation identification accuracy by 15% ↑**</span>.",
        "Code Assistant Tool (Intelligence Tool) for building E2End Data Science Applications. <span style=\"color: green;\">**↑ Reduced development time by 20% ↑**</span>.",
        "Reinforcement learning-based engine for tool troubleshooting in the semiconductor fabrication process. <span style=\"color: green;\">**↑ Decreased troubleshooting time by 25% ↑**</span>.",
        "Built advanced statistics/data science models for identifying variability, which in turn enhances productivity or yield gain. Developed a computer vision model to identify defect patterns. <span style=\"color: green;\">**↑ Increased yield by 10% ↑**</span>.",
        "Built a highly performant model suite that consumes less memory and runs ultra-fast. CPU/GPU acceleration tools increased processing speed by 50-60% and reduced the cost of computation by 50%. <span style=\"color: green;\">**↑ Increased processing speed by 50-60% ↑**</span> and <span style=\"color: red;\">**↓ Reduced the cost of computation by 50% ↓**</span>",
        "Planned and executed the CDS (citizen data science model) in multiple Micron fabs, encouraging all manufacturing engineers to develop problem statements and solve them using the CDS models, which in turn increased the efficiency of the fabs by 20%. <span style=\"color: green;\">**↑ Increased the efficiency of fabs by 20% ↑**</span>"
    ],
    "GEP SOLUTIONS": [
        "Built an NLP-based supplier intent identifier and sentiment detector from news feed data. <span style=\"color: green;\">**↑ Achieved 90% accuracy in intent identification ↑**</span>.",
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
        "Built a web service in .NET that extracts payments from QuickBooks. Also worked on building a proximity distance app in Android."
    ],
    "PIXENTIA SOLUTIONS": [
        "Collaborated with team members to create application system analysis based on client requirements. Created tools like a resume parser and developed a digital badge for online representation of a skill. <span style=\"color: green;\">**↑ Awarded 'Innovation Badge' for developing the digital badge system ↑**</span>."
    ],
    "GYAN DATA Pvt. LTD": [
        "Developed and hosted apps like remote scilab for computations in chemical engineering labs remotely, worked on enterprise level applications like sandman with machine learning as core."
    ],
    "ACADEMIC PROJECTS": [
        "Developed a machine learning project for Predicting prepayments and defaults for privatized banks. <span style=\"color: green;\">**↑ Achieved 85% accuracy in predicting defaults ↑**</span>.",
        "Developed a wake word detection model for the custom word 'HIFREEDA'.",
        "Built a CNN-based image classifier that can distinguish between different handwritten Devanagari characters. <span style=\"color: green;\">**↑ Achieved 92% classification accuracy ↑**</span>.",
        "Distinguishing between Hillary Clinton and Trump tweets.",
        "Developed a streaming analytics application to analyze movie recommendations.",
        "Music Generation using Deep Learning.",
        "Built and implemented a machine learning NLP-based project that computes sentiment for incoming reviews on the Amazon website."
    ],
    "PERSONAL PROJECTS": [
        "Designed an algorithm called Pseudo Random Forest that works better than the Random Forest algorithm. <span style=\"color: green;\">**↑ Improved accuracy by 10% compared to Random Forest ↑**</span>.",
        "Implemented and invented a new technique for oversampling.",
        "Designed TIC-TOC game machine with Reinforcement Learning. <span style=\"color: green;\">**↑ Achieved a winning rate of 90% against random players ↑**</span>.",
        "Persona based custom feed recommendation engine.",
        "Chat bot with RNN."
    ]
}

with tab5:
# Display projects
    st.header("Projects")
# Iterate through companies
    # Iterate through each company and their projects
    for company, projects in project_data.items():
# Format project list as HTML
        st.subheader(company)
        # Combine all projects into a single HTML string for efficient rendering
        project_html = "".join(f"<li>{project}</li>" for project in projects)
        st.write(f"<ul>{project_html}</ul>", unsafe_allow_html=True)

with tab6:
    st.header("Expertise In")
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


    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>Hands-On Generative AI Stack</h3>", unsafe_allow_html=True)
        
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

                st.image("Python-Logo.png", width=150)
            with col2:

                numpy_image_path = "numpy.png"
                st.image(numpy_image_path, width=150)
            with col3:

                pandas_image_path = "pandas_png2.jpg"
                st.image(pandas_image_path, width=150)
            with col4:
                keras_image_path = "keras_logo.png"
                st.image(keras_image_path, width=150)
            with col5:

                tensorflow_image_path = "tf.png"
                st.image(tensorflow_image_path, width=150)
            with col6:

                scikit_learn_image_path = "scikit-learn.png"
                st.image(scikit_learn_image_path, width=150)
            with col7:

                statsmodels_image_path = "statsmodels.png"
                st.image(statsmodels_image_path, width=150)
            with col8:

                plotly_image_path = "causalml.png"
                st.image(plotly_image_path, width=150)
            with col9:

                pytorch_image_path = "pytorch.png"
                st.image(pytorch_image_path, width=150)
            with col10:
                st.image('pymilo.jpg', width=150)


        with st.container(border=True):
            st.markdown("<div style='text-align: center;'><h4>Deployment of Models with Python</h4></div>", unsafe_allow_html=True)
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
            with col1:

                st.image("flask_python.png", width=300)
                # st.write("Flask")
            with col2:
                st.image("fastapi.png", width=300)
            with col3:
                st.image('asyncio.png', width=300)
            with col4:
                st.image("rio.jpg", width=300)
            with col5:

                st.image("streamlit.png", width=300)
            with col6:
                st.image("nicegui.png", width=300)
            with col7:
                st.image("bottle.jpg", width=300)
            with col8:
                st.image("falcon.png", width=300)
            with col9:
                st.image('dash.png', width=300)

            with col10:
                st.image('taipy.png', width=300)


        with st.container(border=True):
            st.markdown("<div style='text-align: center;'><h4>DataScience with R</h4></div>", unsafe_allow_html=True)
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)
            with col1:

                
                st.image("R_logo.png", width=300)
            with col2:
                st.image("mlr3.png", width=300)
            with col3:
                st.image('causalimpact.png', width=500)
            with col4:
                st.image("torch.png", width=300)
            with col5:
                st.image("golem.png", width=300)
            with col6:
                st.image("rcaret.png", width=300)
            with col7:
                st.image("e1071.png", width=300)
            with col8:
                st.image('purrr.jpg', width=300)
               

            with col9:
                st.image('tidyquant.png', width=300)

            with col10:
                st.image('rlang.png', width=300)
            
            with col11:
                st.image('fastai.png', width=300)

        with st.container(border=True):
            st.markdown("<div style='text-align: center;'><h4>Deployment of Models in R</h4></div>", unsafe_allow_html=True)
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
            with col1:

                
                st.image("plumber.png", width=300)
            with col2:
                st.image("httr2.png", width=300)
            with col3:
                st.image('restfulr.png', width=300)
            with col4:
                st.image("vetiver.png", width=300)
            with col5:
                st.image("shiny.jpg", width=300)
            with col6:
                st.image("rhino.png", width=300)
            with col7:
                st.image("rsconnect.png", width=300)
            with col8:
                st.image('reticulate.png', width=300)

            with col9:
                st.image('keras.png', width=300)

            with col10:
                st.image('book.png', width=300)

    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>Expertised in Machine Learning Operations and Engineering Tools</h4>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = st.columns(13)
        with col1:
            st.image("mlflow.png", width=300)
        with col2:
            st.image("kubeflow.png", width=300)
        with col3:
            st.image("airflow.png", width=300)
        with col4:
            st.image("dagster.jpg", width=300)
        with col5:
            st.image("prefect.png", width=300)
        with col6:
            st.image("dvc.png", width=300)
        with col7:
            st.image("seldon.png", width=300)
        with col8:
            st.image("kserve.png", width=300)
        with col9:
            st.image("tfx.png", width=500)
        with col10:
            st.image("metaflow.png", width=300)
        with col11:
            st.image("polyaxon.png", width=300)
        with col12:
            st.image('wb.png', width=300)
        with col13:
            st.image('kedro.png', width=300)

    with st.container(border=True):
        st.markdown("<h4 style='text-align: center;'>Full Stack Engineering and Development</h4>", unsafe_allow_html=True)
        col1, col2, col3,col4  = st.columns(4)
        with col2:
            st.image("mean_stack.png", width=400)
        
        with col3:
            st.image("mern_stack.png", width=400)


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

with tab10:
    st.header("Academic Engagements")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Total Hours Spent Teaching", value=2500, delta=50)
    with col2:
        st.metric(label="Number of Colleges Engaged", value=15, delta=5)

    style_metric_cards(background_color="#fff", border_size_px=1, border_color="#ccc", border_radius_px=5)


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
            # st.write("Description of IIT Hyderabad data.")
            st.write("#### IIIT Delhi")
            st.image("indraprastha-institute-of-information-technology-iiit-delhi-logo.png", width=150)
            # st.write("Description of IIIT Delhi data.")
            st.write("#### Guru Nanak Educational Society")
            st.image("gurunanak.jpg", width=200)
            # st.write("Description of Guru Nanak Educational Society data.")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='container'>", unsafe_allow_html=True)
            st.write("#### University of Texas at Austin")
            st.image("University_of_Texas_at_Austin_seal.svg.png", width=150)
            # st.write("Description of Princeton University data.")
            st.write("#### Vignan Institute of Technology")
            st.image("vignan.jpg", width=150)
            # st.write("Description of Vignan Institute of Technology data.")
            st.write("#### Manipal University Bangalore")
            st.image("manipal.jpg", width=170)

            st.markdown("</div>", unsafe_allow_html=True)

        with col5:
            st.markdown("<div class='container'>", unsafe_allow_html=True)
            st.write("#### CMR College of Engineering")
            st.image("cmr.jpg", width=150)
            # st.write("Description of CMR College of Engineering data.")
            st.write("#### Geetanjali University")
            st.image("geetanjali.jpg", width=150)
            st.write("#### KIIT University")
            st.image("KIIT-Logo.png", width=200)
            # st.write("Description of KIIT University data.")
            st.markdown("</div>", unsafe_allow_html=True)



# Display certificates
with tab4:
# Display certificates
    st.header("Certificates")
# Load skills data

    roles = load_skills_data()
# Initialize certificate counters
# Calculate certificate counts

    # Calculate certificate counts based on skills
    deep_learning_certs = 0
    machine_learning_certs = 0
    statistics_certs = 0
    generative_ai_certs = 0
# Iterate through roles and count certificates
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
# Display metrics in columns
        elif role["name"] == "MLOps Engineer":
# Display certificate counts in metrics
            mlops_certs = len(role["skills"])

    col1, col2, col3 = st.columns(3)
    with col1:
        display_metric("Deep Learning", deep_learning_certs, col1)
# Display lists of certifications
        display_metric("Machine Learning", machine_learning_certs, col1)
    with col2:
        display_metric("Statistics", statistics_certs, col2)
        display_metric("AI and ML Engineer", ai_engineering_certs, col2)
# Display certificate lists in expanders
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
# Helper function to display certification list with expander
        st.subheader("5. Azure Data Scientist Associate")
        st.write("Key Topics: Azure ML, Model Deployment, Data Preparation")
        st.subheader("7. IBM Machine Learning Professional Certificate")
        st.write("Key Topics: Supervised Learning, Unsupervised Learning, Model Evaluation")
        st.subheader("8. Google Machine Learning Crash Course")
        st.write("Key Topics: Supervised Learning, Unsupervised Learning, TensorFlow Basics")
# Helper function to display certification list
        st.subheader("9. Microsoft Azure AI Fundamentals (AI-900)")
        st.write("Key Topics: Azure AI Services, Machine Learning Basics, AI Concepts")


    def display_certification_list(title, certifications):
        with st.expander(title):
            for i, cert in enumerate(certifications):
                st.subheader(f"{i+1}. {cert['name']}")
                st.write(f"Key Topics: {cert['topics']}")

    statistics_certifications = [
        {"name": "Coursera: Statistics with Python Specialization - University of Michigan", "topics": "Descriptive Statistics, Inferential Statistics, Regression Analysis"},
        {"name": "edX: Data Science: Probability - Harvard University", "topics": "Probability Theory, Random Variables, Distributions"},
        {"name": "Bayesian Statistics", "topics": "Bayesian Inference, Prior and Posterior Distributions, Markov Chain Monte Carlo"},
        {"name": "IBM Advanced Data Science with Time Series", "topics": "Time Series Analysis, Forecasting, ARIMA Models"},
        {"name": "Stanford's Introduction to Statistical Learning", "topics": "Linear Regression, Classification, Resampling Methods"},
        {"name": "Statistics with R Specialization - Duke University", "topics": "Descriptive Statistics, Inferential Statistics, Regression Analysis"},
        {"name": "MITx: Probability - The Science of Uncertainty and Data", "topics": "Probability Theory, Random Variables, Distributions"},
    ]

    ai_ml_certifications = [
        {"name": "AI Engineer Professional Certificate - IBM", "topics": "AI Concepts, Machine Learning, Deep Learning, Model Deployment"},
        {"name": "AI and Machine Learning for Coders - Google", "topics": "AI Concepts, Machine Learning, Deep Learning, Model Deployment"},
        {"name": "AI Engineering with TensorFlow - Google Cloud", "topics": "TensorFlow, Keras, Model Deployment, Transfer Learning"},
        {"name": "AI and Machine Learning for Business - Coursera", "topics": "AI Concepts, Machine Learning, Business Applications"},
        {"name": "AI and Machine Learning for Coders - Fast.ai", "topics": "Practical AI, Transfer Learning, Fine-Tuning Models"},
    ]

    mlops_certifications = [
        {"name": "MLOps Specialization - Coursera", "topics": "MLOps Principles, Model Deployment, CI/CD for ML"},
        {"name": "MLOps with TensorFlow - Google Cloud", "topics": "TensorFlow, Keras, Model Deployment, Transfer Learning"},
        {"name": "MLOps Engineer Professional Certificate - IBM", "topics": "MLOps Principles, Model Deployment, CI/CD for ML"},
        {"name": "MLOps Fundamentals - Microsoft Azure", "topics": "Azure ML, Model Deployment, Data Preparation"},
        {"name": "MLOps with AWS - Amazon Web Services", "topics": "AWS ML Services, Model Deployment, Data Engineering"},
    ]

    display_certification_list("List of Statistics Certifications", statistics_certifications)
    display_certification_list("List of AI and ML Engineer Certifications", ai_ml_certifications)
    display_certification_list("List of MLOps Certifications", mlops_certifications)


with tab7:
# Display teaching experience information
    st.header("Teaching Experience")

    st.subheader("Teaching Philosophy")
    st.write(
        """
        My teaching philosophy centers around creating engaging and interactive learning experiences that empower students to develop a deep understanding of complex concepts. I strive to foster a collaborative environment where students feel comfortable asking questions, exploring new ideas, and applying their knowledge to real-world problems.
        """
    )

    st.subheader("Subject Areas")
# Python courses

    # Python
    with st.expander("Python"):
        st.write(
            """
            **Courses Taught:** Introduction to Python Programming, Data Analysis with Python, Machine Learning with Python

            **Technologies Used:** Python, Jupyter Notebook, Pandas, NumPy, Scikit-learn

            **Student Outcomes:** Students successfully completed projects involving data analysis, machine learning model development, and web application creation. Average student satisfaction score: 4.5/5.
            """
# Deep Learning courses
        )

    # Deep Learning
    with st.expander("Deep Learning"):
        st.write(
            """
            **Courses Taught:** Deep Learning Fundamentals, Convolutional Neural Networks, Recurrent Neural Networks

            **Technologies Used:** TensorFlow, Keras, PyTorch, CUDA

            **Student Outcomes:** Students built deep learning models for image classification, natural language processing, and time series forecasting. Project success rate: 90%.
# NLP courses
            """
        )

    # NLP
    with st.expander("Natural Language Processing (NLP)"):
        st.write(
            """
            **Courses Taught:** Natural Language Processing with Python, Text Mining, Sentiment Analysis

# Request Course Enrollment Button
    if st.button("Request Course Enrollment"):
        st.write("Thank you for your interest! Please contact me for enrollment details.")
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

            **Role:** Workshop facilitator
            """
        )

    # IIM Indore (IIM I)
    st.markdown("#### IIM Indore (IIM I)  <img src='https://img.icons8.com/color/30/000000/university.png'>", unsafe_allow_html=True)
    with st.expander("See Details"):
        st.write(
            """
            **Course Name:** Business Analytics Program

            **Duration:** 1 Week

            **Role:** Visiting faculty
            """
        )

    # IIM Ahmedabad (IIM A)
    st.markdown("#### IIM Ahmedabad (IIM A)  <img src='https://img.icons8.com/color/30/000000/university.png'>", unsafe_allow_html=True)
    with st.expander("See Details"):
        st.write(
            """
            **Course Name:** AI for Business Leaders

            **Duration:** 3 Days

            **Role:** Guest speaker
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
# Display past consulting engagements

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

# Display email input fields and send button
    # Request Appointment Button
# Email input fields
    with st.container(border=True):
        st.subheader("Need my consultation! please feel free to reach out to me!")
        col1, col2 = st.columns(2)
        with col1:
            subject = st.text_input("Subject", placeholder="Enter the subject here")
        with col2:
            receiver_email = st.text_input("Receiver Email", placeholder="Enter receiver email here")

    # with st.container():
        message = st.text_area("Message", placeholder="Enter your message here")

    # Button to send email
    # with st.container():
       
        if st.button("Send Email"):
            send_email(subject, message, receiver_email)
            st.success("Email sent successfully!")

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

 

    st.subheader("Courses trained along with companies details.")

# Course 1: Data Science with Python
    # Course 1
    with st.container():
        st.subheader("Data Science with Python")
        st.write("**Description:** A comprehensive introduction to data science using Python.")
        st.write("**Training Hours:** 40")
        st.write("**Company:** SG Analytics")
        st.markdown("---")
# Course 2: Data Science with R

    # Course 2
    with st.container():
        st.subheader("Data Science with R")
        st.write("**Description:** A comprehensive introduction to data science using R.")
        st.write("**Training Hours:** 40")
        st.write("**Company:** UNext/Deloitte")
# Course 3: Statistics for Data Science
        st.markdown("---")

    # Course 3
    with st.container():
        st.subheader("Statistics for Data Science ")
        st.write("**Description:** A deep dive into statistical concepts for data science.")
        st.write("**Training Hours:** 30")
        st.write("**Company:** Cognizant under Jigsaw Engagement")
# Course 4: Deep Learning Fundamentals
        st.markdown("---")

    # Course 4
    with st.container():
        st.subheader("Deep Learning Fundamentals For SG Analytics")
        st.write("**Description:** An introduction to deep learning concepts and techniques.")
        st.write("**Training Hours:** 50")
# Course 5: Generative AI
        st.write("**Company:** SG Analytics")
        st.markdown("---")

    # Course 5
    with st.container():
        st.subheader("Generative AI")
# Display academic journey information
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
        {"Name": "Pseudo Random Forest",
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
        Data Scientist and Full Stack Developer with 10+ years of experience.

        **Skills:** Data Analysis, Feature Engineering, Big Data, Machine Learning, Deep Learning, Python, R, Java, .Net, SQL, Cloud Solutions, LLM, Front-End Frameworks (React, Angular, Vue), Frameworks (Langchain, LlamaIndex, Haystack), Databases (Weaviate, Pinecone, Chroma), Foundation Models (GPT, Claude, Gemini), Cloud Providers (AWS, Azure, GCP), and more.

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

# Display professional journey information
with tab2:
# Display metrics
    st.header("Professional Journey")
    cols = st.columns(4)

    with cols[0]:
        ui.metric_card(title="Total Projects Deployed", content="40+", description="",key="card4")
    with cols[1]:
        ui.metric_card(title="Patents Published", content="7+", description="", key="card5")
    with cols[2]:
        ui.metric_card(title="Total Working Experience", content="10+", description="", key="card6")

    with cols[3]:
# Timeline Data
        ui.metric_card(title="Client Engagements", content="30+", description="", key="card7")

    # Timeline Data
    timeline_data = [
        {"Name": "Micron Technology, Inc.",
            "Position": "Principal Data Scientist",
            "start": datetime(2021, 10, 14),
            "end": datetime(2025, 6, 30),
            "description": "Quantum Machine Learning suite for variation identification. Code Assistant Tool. Reinforcement learning based Engine for Tool Troubleshooting. Built advanced statistics/data-science models.",
            "logo": "BITS_Pilani-Logo.png"},
        {"Name": "GEP SOLUTIONS",
            "Position": "Data Scientist",
            "start": datetime(2019, 8, 9),
            "end": datetime(2021, 10, 13),
            "description": "Built NLP based supplier intent identifier. Built a text summary extractor. Built promotional modelling for demand planners. Built a large-scale Demand Planning and Demand Sensing pipeline.",
            "logo": "university_of_chicago.png"},
        {"Name": "HIGH RADIUS PVT LIMITED",
            "Position": "Data Scientist",
            "start": datetime(2017, 7, 1),
            "end": datetime(2019, 8, 8),
            "description": "Proactively built and automated the claim extraction process. Implemented GANs, Semi Supervised GANs. Customer Segmentation model. Implemented “HiFreeda” Wake word detection model.",
            "logo": "University-of-Chicago-Seal.png"},
        {"Name": "BITMIN INFO SYSTEMS",
            "Position": "Data Scientist and Full Stack Developer",
            "start": datetime(2016, 4, 1),
            "end": datetime(2017, 6, 30),
            "description": "Developed apps, worked on the back-end and front-end. Created Machine Learning-based tools. Built a web service in .NET that extracts payments from QuickBooks.",
            "logo": "iit_madras_logo.png"},
        {"Name": "PIXENTIA SOLUTIONS",
            "Position": "Software Engineer-I",
            "start": datetime(2015, 2, 1),
            "end": datetime(2016, 3, 31),
            "description": "Collaborated with team members to create applications system analysis based upon client requirements. Created tools like resume parser, Developed digital badge online representation of a skill.",
            "logo": "bitspilani_logo.png"},
        {"Name": "GYAN DATA Pvt. LTD",
            "Position": "Software Engineer Trainee",
            "start": datetime(2012, 11, 1),
            "end": datetime(2014, 5, 30),
            "description": "Developed and hosted apps like remote scilab for computations in chemical engineering labs remotely, worked on enterprise level applications like sandman with machine learning as core.",
# Create timeline chart
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

