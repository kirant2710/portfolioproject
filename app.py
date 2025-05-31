import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import json

st.set_page_config(layout="wide")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["About Me", "Professional Journey", "Academic Journey","Certificates","Projects", "Skills" , "Teching Experience", "Corporate Training and Consulting"])

project_data = {
    "MICRON": [
        "Quantum Machine Learning suite for variation identification in the semiconductor manufacturing process.",
        "Code Assistant Tool (Intelligence Tool) for building E2End Data Science Applications.",
        "Reinforcement learning based Engine for Tool Trouble shooting in Semiconductor Fabrication Process.",
        "Built advanced statistics/data-science models for identifying variability which in turn enhances the productivity or yield gain. Computer Vision Gyan based model identifying defect patterns."
    ],
    "GEP SOLUTIONS": [
        "Built NLP based supplier intent identifier, sentiment detector from news feed data.",
        "Built a text summary extractor from news feeds and duplicate news detection.",
        "Built promotional modelling for demand planners which is included in NEXXE (GEP product).",
        "Built a large-scale Demand Planning and Demand Sensing pipeline for the Sales data which runs with different time series models including hierarchical time series-based models.",
        "Built an optimized hierarchical time series model which runs on py spark with 2000 items in 5 min span.",
        "Built a spend forecasting pipeline with different time series algorithms.",
        "Built a Hybrid Time Series Model for Forecasting Commodities and Sales Data."
    ],
    "HIGH RADIUS PVT LIMITED": [
        "Proactively built and automated the claim extraction process from pdf templates by using deep learning and NLP.",
        "Implemented GANs, Semi Supervised GANs for check/remittance identification/classification instead of labeling for each data point.",
        "Customer Segmentation model for high radius Cash App customers.",
        "Implemented “HiFreeda” Wake word detection model for high radius chat bot FREEDA.",
        "Implemented Noise Detection GANs for the noise removal from images.",
        "Built and implemented deduction validity predictor for P&G.",
        "Implemented auto regex generator which gives regex by using pattern in the data.",
        "Implemented and created many feature engineering techniques, designed new algorithms for Oversampling analysis for imbalanced data and ML (Pseudo Random Forest which works better than RF in ML)."
    ],
    "BITMIN INFO SYSTEMS": [
        "Developed apps, worked on the back-end and front-end enabling the existence of features and application development.",
        "Created Machine Learning-based tools for the company, such as Employee Iteration Rate prediction, Payment Date Prediction, Fraud Detection Analytics.",
        "Built a webservice in .net which extracts the payments from quick books, also worked on building a proximity distance app (ceechat) in android."
    ],
    "PIXENTIA SOLUTIONS": [
        "Collaborated with team members to create applications system analysis based upon client requirements. Created tools like resume parser, Developed digital badge online representation of a skill."
    ],
    "GYAN DATA Pvt. LTD": [
        "Developed and hosted apps like remote scilab for computations in chemical engineering labs remotely, worked on enterprise level applications like sandman with machine learning as core."
    ],
    "ACADEMIC PROJECTS": [
        "Developed a machine learning project for Predicting prepayments and defaults for privatized banks.",
        "Wake word detection model for the custom word HIFREEDA",
        "Build a CNN based image classifier, which will be able to distinguish between different handwritten Devanagari characters.",
        "Distinguishing between Hillary Clinton and Trump tweets.",
        "Developed a streaming analytics application to analyze movie recommendations.",
        "Music Generation using Deep Learning.",
        "Built and implement and machine learning NLP based project which computes sentiment for an incoming review in amazon online website."
    ],
    "PERSONAL PROJECTS": [
        "Designed an Algorithm called as Pseudo Random Forest Which works better than Random Forest algorithm.",
        "Implemented and invented a new technique for oversampling.",
        "Designed TIC-TOC game machine with Reinforcement Learning.",
        "Persona based custom feed recommendation engine.",
        "Chat bot with RNN."
    ]
}

with tab5:
    st.header("Projects")
    for company, projects in project_data.items():
        st.subheader(company)
        for project in projects:
            st.write(f"* {project}")

with tab6:
    st.header("Skills")

    # Load skills data from JSON file
    try:
        with open("skills_data.json", "r") as f:
            skills_data = json.load(f)
        roles = skills_data["roles"]
    except FileNotFoundError:
        st.error("Could not load skills data. Please ensure skills_data.json exists.")
        roles = []
    except json.JSONDecodeError:
        st.error("Could not decode skills data. Please ensure skills_data.json is valid JSON.")
        roles = []

    # Add custom CSS for styling
    st.markdown(
        """
        <style>
        .skills-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-gap: 10px;
        }
        .skill-item {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .skill-item img {
            width: 20px;
            height: 20px;
            vertical-align: middle;
            margin-right: 5px;
        }
        @media (max-width: 768px) {
            .skills-grid {
                grid-template-columns: repeat(1, 1fr);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.write("<div class='skills-grid'>", unsafe_allow_html=True)  # Start the grid

    skill_list = roles  # Use roles directly from loaded data
    grid_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]  # Define grid positions

    # Populate the grid with skills
    for i in range(min(len(skill_list), len(grid_positions))):  # Limit to available positions
        try:
            role = skill_list[i]
            row, col = grid_positions[i]
            st.write(f"<div class='skill-item'>", unsafe_allow_html=True)  # Start a skill item
            st.subheader(role["name"])
            for skill in role["skills"]:
                icon_html = f'<img src="{skill["icon"]}" width="20" alt="{skill.get("alt", "Skill Icon")}">'
                st.markdown(icon_html + f' **{skill["name"]}:** {skill["description"]}', unsafe_allow_html=True)
            st.write("</div>", unsafe_allow_html=True)  # Close the skill item
        except Exception as e:
            st.error(f"Error rendering skill item: {e}")

    st.write("</div>", unsafe_allow_html=True)  # Close the grid

with tab4:
    st.header("Certificates")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.metric(label="Active Certificates", value=120)
    with col2:
        st.metric(label="Certificates Pending Approval", value=15, delta="2%")
    with col3:
        st.metric(label="Percentage of Employees Certified", value="85%", delta="5%")
    with col4:
        st.metric(label="Cost Savings from Certifications", value="$150,000", delta="$10,000")

with tab7:
    st.header("Teaching Experience")
    st.write("This section is under construction.")

with tab8:
    st.header("Corporate Training and Consulting")
    st.write("This section is under construction.")

with tab3:
    st.header("Academic Journey")

    st.subheader("IIT Madras, Chennai, IN")
    st.write("Degree: B. Tech")
    st.write("Major: Chemical Engineering")
    st.write("Minor: Data Analytics")
    st.write("Graduation Date: Jun 2014")
    st.write("GPA: 7.1/10")

    st.subheader("University of Chicago Graham School, Chicago, US")
    st.write("Degree: Post Graduate Program")
    st.write("Major: Data Science and Machine Learning")
    st.write("Graduation Date: May 2019")
    st.write("GPA: 3.5/4.0")

    st.subheader("Birla Institute of Technology and Science, Pilani")
    st.write("Degree: M. Tech")
    st.write("Major: Software Systems")
    st.write("Specialization: Data Sciences and Cloud Computing")
    st.write("Graduation Date: Jun 2021")
    st.write("GPA: 7.5/10")

    st.header("Academic Projects")
    st.write("""
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

    for item in timeline_data:
        st.write(f'[{item["Name"]}]({item["link"]})')

with tab1:
    st.header("Kiran Kumar Kandula")
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

with tab2:
    st.header("Professional Journey")

    # Timeline Data
    timeline_data = [
        {"Name": "Micron Technology, Inc.",
            "Position": "Principal Data Scientist",
            "start": datetime(2021, 10, 1),
            "end": datetime(2025, 5, 30),
            "description": "Quantum Machine Learning suite for variation identification. Code Assistant Tool. Reinforcement learning based Engine for Tool Trouble shooting. Built advanced statistics/data-science models."},
        {"Name": "GEP SOLUTIONS",
            "Position": "Data Scientist",
            "start": datetime(2019, 8, 1),
            "end": datetime(2025, 5, 30),
            "description": "Built NLP based supplier intent identifier. Built a text summary extractor. Built promotional modelling for demand planners. Built a large-scale Demand Planning and Demand Sensing pipeline."},
        {"Name": "HIGH RADIUS PVT LIMITED",
            "Position": "Data Scientist",
            "start": datetime(2017, 7, 1),
            "end": datetime(2025, 5, 30),
            "description": "Proactively built and automated the claim extraction process. Implemented GANs, Semi Supervised GANs. Customer Segmentation model. Implemented “HiFreeda” Wake word detection model."},
        {"Name": "BITMIN INFO SYSTEMS",
            "Position": "Data Scientist and Full Stack Developer",
            "start": datetime(2016, 4, 1),
            "end": datetime(2025, 5, 30),
            "description": "Developed apps, worked on the back-end and front-end. Created Machine Learning-based tools. Built a webservice in .net which extracts the payments from quick books."},
        {"Name": "PIXENTIA SOLUTIONS",
            "Position": "Software Engineer-I",
            "start": datetime(2015, 2, 1),
            "end": datetime(2025, 5, 30),
            "description": "Collaborated with team members to create applications system analysis based upon client requirements. Created tools like resume parser, Developed digital badge online representation of a skill."},
        {"Name": "GYAN DATA Pvt. LTD",
            "Position": "Software Engineer Trainee",
            "start": datetime(2012, 11, 1),
            "end": datetime(2025, 5, 30),
            "description": "Developed and hosted apps like remote scilab for computations in chemical engineering labs remotely, worked on enterprise level applications like sandman with machine learning as core."},
    ]

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
            hovertemplate=f"<b>{item['Name']}</b><br>{item['Position']}<br>{item['start'].strftime('%Y-%m-%d')} - {item['end'].strftime('%Y-%m-%d')}<br>{item['description']}<extra></extra>"
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
