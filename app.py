import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide",
                   page_title='JobIntel',
                   page_icon='💼',
                   )

# data analysis process
job = pd.read_csv(r"D:\Data Science Works\Data\eda_data.csv")

job.drop(columns=['Unnamed: 0', 'Job Description', 'Company Name', 'age', 'desc_len', 'num_comp', 'python_yn', 'R_yn',
                  'max_salary', 'job_simp', 'job_state', 'min_salary', 'spark', 'aws', 'excel', 'seniority',
                  'same_state', 'Competitors', 'hourly', 'employer_provided'], inplace=True)

job['Type of ownership'] = job['Type of ownership'].str.replace('Company ', '').str.replace('-', '').str.strip()

job = job[job['Type of ownership'] != '1']

job = job.reset_index(drop=True)

job = job[job['Founded'] != -1]

job = job.reset_index(drop=True)

job['Job Title'] = job['Job Title'].str.upper()

job['Revenue'] = job['Revenue'].str.replace('Unknown / Non-Applicable', 'Undisclosed')

job.rename(
    columns={'Salary Estimate': 'Salary Est.(By Glassdoor)', 'Size': 'Size of Employees', 'Type of ownership': 'Type',
             'avg_salary': 'AVG. Salary in USD', 'company_txt': 'Company Name'}, inplace=True)

job['Company Name'] = job['Company Name'].str.replace('<intent>', 'Intent Company Group').str.strip()

job['Company Name'].sort_values(ascending=True).unique()

ds_keywords = ['DATA SCIENTIST, RICE UNIVERSITY', 'DATA SCIENTIST', 'HEALTHCARE DATA SCIENTIST',
               'RESEARCH SCIENTIST', 'STAFF DATA SCIENTIST - TECHNOLOGY', 'SCIENTIST I/II, BIOLOGY',
               'CUSTOMER DATA SCIENTIST', 'ASSOCIATE SCIENTIST', 'SCIENTIST 2, QC VIRAL VECTOR',
               'DATA SCIENTIST - ALGORITHMS & INFERENCE', 'SCIENTIST',
               'PRICIPAL SCIENTIST MOLECULAR AND CELLULAR BIOLOGIST', 'SCIENTIST/SENIOR SCIENTIST, AUTOIMMUNE',
               'DATA SCIENTIST (ACTUARY, FSA OR ASA)', 'STAFF DATA SCIENTIST',
               'SENIOR SCIENTIST - NEUROSCIENCE', 'MEDICAL LAB SCIENTIST',
               'RISK AND ANALYTICS IT, DATA SCIENTIST', 'PRINCIPAL SCIENTIST, HEMATOLOGY',
               'SCIENTIST, ANALYTICAL DEVELOPMENT', 'LEAD DATA SCIENTIST', 'SPECTRAL SCIENTIST/ENGINEER',
               'DIRECTOR, DATA SCIENCE',
               'SR. DATA SCIENTIST', 'COLLEGE HIRE - DATA SCIENTIST - OPEN TO DECEMBER 2019 GRADUATES',
               'SENIOR RESEARCH SCIENTIST - EMBEDDED SYSTEM DEVELOPMENT FOR DEVOPS',
               'DATA SCIENTIST - BIOINFORMATICS', 'SR. SCIENTIST - DIGITAL & IMAGE ANALYSIS/COMPUTATIONAL PATHOLOGY',
               'PRINCIPAL SCIENTIST - IMMUNOLOGIST', 'PRINCIPAL DATA SCIENTIST (COMPUTATIONAL CHEMISTRY)',
               'PRINCIPAL SCIENTIST, CHEMISTRY & IMMUNOLOGY', 'MEDICAL LABORATORY SCIENTIST',
               'RESEARCH SCIENTIST - BIOLOGICAL SAFETY', 'PRINCIPAL DATA SCIENTIST WITH OVER 10 YEARS EXPERIENCE',
               'MED TECH/LAB SCIENTIST - LABORATORY', 'PL ACTUARIAL-LEAD DATA SCIENTIST',
               'SR. SCIENTIST, QUANTITATIVE TRANSLATIONAL SCIENCES',
               'MEDICAL TECHNOLOGIST / CLINICAL LABORATORY SCIENTIST',
               'ASSOCIATE DATA SCIENTIST/COMPUTER SCIENTIST',
               'ASSOCIATE SCIENTIST/SCIENTIST, PROCESS ANALYTICAL TECHNOLOGY - SMALL MOLECULE ANALYTICAL CHEMISTRY',
               'SENIOR DATA SCIENTIST: CAUSAL & PREDICTIVE ANALYTICS AI INNOVATION LAB',
               'PRINCIPAL DATA ENGINEER, DATA PLATFORM & INSIGHTS',
               'SR. DATA SCIENTIST II', 'VP, DATA SCIENCE', 'MED TECH/LAB SCIENTIST- SOUTH COASTAL LAB',
               'FOOD SCIENTIST - DEVELOPER', 'SENIOR RESEARCH STATISTICIAN- DATA SCIENTIST',
               'STAFF SCIENTIST-DOWNSTREAM PROCESS DEVELOPMENT', 'DATA SCIENTIST, SENIOR',
               'CUSTOMER DATA SCIENTIST/SALES ENGINEER', 'DATA SCIENTIST, OFFICE OF DATA SCIENCE',
               'DATA SCIENCE ANALYST', 'INSURANCE DATA SCIENTIST',
               'SENIOR SCIENTIST (NEUROSCIENCE)', 'PROJECT SCIENTIST', 'SENIOR RISK DATA SCIENTIST',
               'DATA SCIENTIST - RESEARCH', 'DATA SCIENTIST SR', 'R&D SR DATA SCIENTIST',
               'AG DATA SCIENTIST', 'STAFF SCIENTIST', 'DATA SCIENTIST II',
               'CUSTOMER DATA SCIENTIST/SALES ENGINEER (BAY', 'SR. SCIENTIST II',
               'DATA SCIENTIST - HEALTH DATA ANALYTICS', 'SENIOR DATA SCIENTIST',
               'CLINICAL LABORATORY SCIENTIST', 'SENIOR INSURANCE DATA SCIENTIST',
               'SENIOR DATA SCIENTIST - R&D ONCOLOGY', 'SENIOR DATA SCIENCE SYSTEMS ENGINEER',
               'ENVIRONMENTAL ENGINEER/SCIENTIST', 'ASSOCIATE SCIENTIST, LC/MS BIOLOGICS',
               'SR. SCIENTIST METHOD DEVELOPMENT', 'PROJECT SCIENTIST - AUTON LAB, ROBOTICS INSTITUTE',
               'SENIOR SCIENTIST - REGULATORY SUBMISSIONS', 'SENIOR MANAGER, EPIDEMIOLOGIC DATA SCIENTIST',
               'SCIENTIST - BIOMARKER AND FLOW CYTOMETRY', 'ASSOCIATE ENVIRONMENTAL SCIENTIST - WILDLIFE BIOLOGIST',
               'ASSOCIATE, DATA SCIENCE, INTERNAL AUDIT', 'SENIOR LIDAR DATA SCIENTIST',
               'DIRECTOR II, DATA SCIENCE - GRM ACTUARIAL', 'SCIENTIST, PHARMACOMETRICS',
               'STAFF SCIENTIST- UPSTREAM PD',
               'SCIENTIST MANUFACTURING PHARMA - KENTUCKY BIOPROCESSING',
               'SCIENTIST, MOLECULAR/CELLULAR BIOLOGIST',
               'PRODUCT ENGINEER – SPATIAL DATA SCIENCE AND STATISTICAL ANALYSIS',
               'MEDICAL LAB SCIENTIST - MLT', 'SR SOFTWARE ENGINEER (DATA SCIENTIST)',
               'ASSOCIATE DIRECTOR/DIRECTOR, SAFETY SCIENTIST',
               'SCIENTIST - ANALYTICAL SERVICES', 'COMPUTATIONAL CHEMIST/DATA SCIENTIST',
               'SCIENTIST MANUFACTURING - KENTUCKY BIOPROCESSING',
               'DATA SCIENCE PROJECT MANAGER', 'MANAGER OF DATA SCIENCE',
               'SOFTWARE ENGINEER (DATA SCIENTIST/SOFTWARE ENGINEER) - SISW - MG',
               'ASSOCIATE RESEARCH SCIENTIST I (PROTEIN EXPRESSION AND PRODUCTION)',
               'RESEARCH SCIENTIST / PRINCIPAL RESEARCH SCIENTIST - MULTIPHYSICAL SYSTEMS',
               'RESEARCH SCIENTIST, MACHINE LEARNING DEPARTMENT',
               'DIRECTOR, PRECISION MEDICINE CLINICAL BIOMARKER SCIENTIST',
               'SENIOR FORMULATIONS SCIENTIST II', 'DATA SCIENCE MANAGER',
               'SOFTWARE ENGINEER STAFF SCIENTIST: HUMAN LANGUAGE TECHNOLOGIES',
               'SR SCIENTIST, IMMUNO-ONCOLOGY - ONCOLOGY',
               'RESEARCH SCIENTIST OR SENIOR RESEARCH SCIENTIST - COMPUTER VISION',
               'RESEARCH SCIENTIST – SECURITY AND PRIVACY', 'SENIOR OPERATIONS DATA ANALYST, CALL CENTER OPERATIONS',
               'DIRECTOR II, DATA SCIENCE - GRS PREDICTIVE ANALYTICS',
               'MANAGER, SAFETY SCIENTIST, MEDICAL SAFETY & RISK MANAGEMENT',
               'DATA SCIENTIST MANAGER',
               'RESEARCH COMPUTER SCIENTIST - RESEARCH ENGINEER - SR. COMPUTER SCIENTIST - SOFTWARE DEVELOPMENT',
               'SENIOR DATA SCIENTIST - ALGORITHMS', 'SCIENTIST, IMMUNO-ONCOLOGY',
               'DATA SCIENTIST IN TRANSLATIONAL MEDICINE', 'CLINICAL SCIENTIST, CLINICAL DEVELOPMENT',
               'SENIOR DATA SCIENTIST – VISUALIZATION, NOVARTIS AI INNOVATION LAB', 'SENIOR DATA SCIENTIST ONCOLOGY',
               'PRINCIPAL DATA SCIENTIST',
               'DATA SCIENTIST - ALPHA INSIGHTS', 'DATA SCIENCE ENGINEER - MOBILE',
               'ASSOCIATE SCIENTIST / SR. ASSOCIATE SCIENTIST, ANTIBODY DISCOVERY',
               'SCIENTIST - CVRM METABOLISM - IN VIVO PHARMACOLOGY',
               'DATA SCIENTIST (WAREHOUSE AUTOMATION)', 'JR. DATA SCIENTIST',
               'SR EXPERT DATA SCIENCE,ADVANCED VISUAL ANALYTICS (ASSOCIATE LEVEL)',
               'ASSOCIATE PRINCIPAL SCIENTIST, PHARMACOGENOMICS', 'ASSISTANT DIRECTOR/DIRECTOR, OFFICE OF DATA SCIENCE',
               'DATA SCIENTIST - SYSTEMS ENGINEERING', 'SENIOR SCIENTIST - BIOSTATISTICIAN',
               'SENIOR SCIENTIST - TOXICOLOGIST - PRODUCT INTEGRITY (STEWARDSHIP)', 'PRODUCT ENGINEER – DATA SCIENCE',
               'SENIOR DATA SCIENTIST / MACHINE LEARNING', 'CLINICAL DATA SCIENTIST',
               'DATA SCIENTIST - QUANTITATIVE', 'DIGITAL HEALTH DATA SCIENTIST', ]
da_keywords = ['FOUNDATIONAL COMMUNITY SUPPORTS DATA ANALYST', 'QUALITY CONTROL SCIENTIST III- ANALYTICAL DEVELOPMENT',
               'SENIOR HEALTH DATA ANALYST, STAR RATINGS', 'SENIOR QUANTITATIVE ANALYST',
               'SENIOR DATA ANALYST/SCIENTIST', 'DATA MODELER (ANALYTICAL SYSTEMS)', 'SUPPLY CHAIN DATA ANALYST',
               'IT ASSOCIATE DATA ANALYST', 'BUSINESS DATA ANALYST',
               'DATA SCIENTIST - SALES', 'DATA ANALYST CHEMIST - QUALITY SYSTEM CONTRACTOR',
               'PRINCIPAL, DATA SCIENCE - ADVANCED ANALYTICS', 'LEAD DATA ANALYST', 'PRODUCTS DATA ANALYST II',
               'DATA ANALYST, PERFORMANCE PARTNERSHIP', 'SYSTEMS ENGINEER II - DATA ANALYST',
               'SENIOR RESEARCH ANALYTICAL SCIENTIST-NON-TARGETED ANALYSIS',
               'DIRECTOR DATA SCIENCE', 'SR DATA ANALYST - IT', 'DATA ANALYST SENIOR',
               'ANALYTICS - BUSINESS ASSURANCE DATA ANALYST',
               'SALESFORCE ANALYTICS CONSULTANT''MARKETING DATA ANALYST, MAY 2020 UNDERGRAD',
               'CORPORATE RISK DATA ANALYST (SQL BASED) - MILWAUKEE OR', 'REVENUE ANALYTICS MANAGER',
               'DATA ANALYTICS PROJECT MANAGER', 'ASSOCIATE DATA ANALYST- GRADUATE DEVELOPMENT PROGRAM',
               'CONSULTANT - ANALYTICS CONSULTING', 'MARKET DATA ANALYST', 'CLINICAL DATA ANALYST',
               'ASSOCIATE SCIENTIST/SCIENTIST, PROCESS ANALYTICAL TECHNOLOGY - SMALL MOLECULE ANALYTICAL CHEMISTRY',
               'CONSULTANT– DATA ANALYTICS GROUP', 'INFORMATION SECURITY DATA ANALYST',
               'INSURANCE FINANCIAL DATA ANALYST', 'SURVEY DATA ANALYST', 'LEAD HEALTH DATA ANALYST - FRONT END',
               'SR. DATA ANALYST', 'PROGRAM/DATA ANALYST', 'JUNIOR DATA ANALYST', 'SYSTEM AND DATA ANALYST',
               'DATA & ANALYTICS CONSULTANT (NYC)', 'BUSINESS INTELLIGENCE ANALYST / DEVELOPER',
               'ANALYTICS MANAGER - DATA MART', 'MARKETING DATA ANALYST', 'SOFTWARE ENGINEER - DATA VISUALIZATION',
               'ANALYTICS MANAGER', 'EXCEL / VBA / SQL DATA ANALYST', 'RADAR DATA ANALYST',
               'DIGITAL MARKETING & ECOMMERCE DATA ANALYST', 'BI & PLATFORM ANALYTICS MANAGER',
               'SR. DATA SCIENTIST', 'BUSINESS DATA ANALYST, SQL', 'ENTIST - ANALYTICS, PERSONALIZED HEALTHCARE (PHC)',
               'DATA ANALYST - ASSET MANAGEMENT', 'FINANCIAL DATA ANALYST', 'DATA ANALYTICS MANAGER',
               'SENIOR DATA ANALYST', 'DATA ANALYST / SCIENTIST''E-COMMERCE DATA ANALYST', 'ANALYTICS CONSULTANT',
               'JR. BUSINESS DATA ANALYST', 'R&D DATA ANALYSIS SCIENTIST', 'WEB DATA ANALYST', 'DATA ANALYST',
               'ASSOCIATE DATA ANALYST', ]
de_keywords = ['DATA ARCHITECT / DATA MODELER', 'SR. DATA ENGINEER | BIG DATA SAAS PIPELINE',
               'SENIOR ENGINEER, DATA MANAGEMENT ENGINEERING', 'DATA ENGINEERING ANALYST', 'DATA ENGINEER - ETL',
               'DATA ENGINEER - CONSULTANT (CHARLOTTE BASED)', 'SOFTWARE DATA ENGINEER - COLLEGE',
               'LEAD DATA ENGINEER (PYTHON)', 'LEAD BIG DATA ENGINEER',
               'TECHNOLOGY-MINDED, DATA PROFESSIONAL OPPORTUNITIES', 'SR DATA ENGINEER (SR BI DEVELOPER)',
               'DATA MODELER - DATA SOLUTIONS ENGINEER', 'ENTERPRISE ARCHITECT, DATA', 'IT - DATA ENGINEER II',
               'STAFF DATA ENGINEER', 'ASSOCIATE DATA ENGINEER',
               'STAFF BI AND DATA ENGINEER', 'BIG DATA ENGINEER',
               'DATA ENGINEER, DATA ENGINEERING AND ARTIFICAL INTELLIGENCE', 'DATA ENGINEER I - AZURE',
               'SQL DATA ENGINEER', 'BIG DATA ENGINEER - CHICAGO - FUTURE OPPORTUNITY', 'SR. DATA ENGINEER',
               'SR. DATA ENGINEER - CONTRACT-TO-HIRE (JAVA)', 'LEAD DATA ENGINEER', 'MONGODB DATA ENGINEER II',
               'SENIOR DATA SCIENTIST STATISTICS',
               'SENIOR SPARK ENGINEER (DATA SCIENCE)', 'SENIOR DATA ENGINEER', 'DATA MODELER', 'DATA ENGINEER I',
               'DATA MANAGEMENT SPECIALIST', 'DATA ENGINEER', ]
ml_keywords = ['MACHINE LEARNING ENGINEER (NLP)', 'PRINCIPAL MACHINE LEARNING SCIENTIST',
               'ASSOCIATE MACHINE LEARNING ENGINEER / DATA SCIENTIST MAY 2020 UNDERGRAD',
               'SENIOR DATA SCIENTIST 4 ARTIFICIAL INTELLIGENCE', 'SENIOR DATA SCIENTIST ARTIFICIAL INTELLIGENCE',
               'MANAGING DATA SCIENTIST/ML ENGINEER',
               'ASSOCIATE DIRECTOR, PLATFORM AND DEVOPS- DATA ENGINEERING AND ARITIFICAL INTELLIGENCE',
               'MACHINE LEARNING ENGINEER - REGULATORY', 'DIRECTOR - DATA, PRIVACY AND AI GOVERNANCE',
               'SENIOR RESEARCH SCIENTIST-MACHINE LEARNING', 'STAFF MACHINE LEARNING ENGINEER',
               'SENIOR DATA & MACHINE LEARNING SCIENTIST', 'MACHINE LEARNING RESEARCH SCIENTIST',
               'MACHINE LEARNING ENGINEER', 'DATA SCIENTIST/ML ENGINEER',
               'SENIOR MACHINE LEARNING (ML) ENGINEER / DATA SCIENTIST - CYBER SECURITY ANALYTICS',
               'DATA SCIENTIST IN ARTIFICIAL INTELLIGENCE EARLY CAREER', 'SENIOR DATA SCIENTIST / MACHINE LEARNING',
               'DATA SCIENTIST / MACHINE LEARNING EXPERT', ]

ds_keywords = [item.lower() for item in ds_keywords]
da_keywords = [item.lower() for item in da_keywords]
de_keywords = [item.lower() for item in de_keywords]
ml_keywords = [item.lower() for item in ml_keywords]


def categorize_job(job_title):
    job_title = job_title.lower()  # Convert to lowercase for case-insensitive matching
    if any(keyword in job_title for keyword in ds_keywords):
        return 'Data Science'
    elif any(keyword in job_title for keyword in da_keywords):
        return 'Data Analytics'
    elif any(keyword in job_title for keyword in de_keywords):
        return 'Data Engineering'
    else:
        return 'Machine Learning'


job['Job Category'] = job['Job Title'].apply(categorize_job)

st.markdown("<h1 style='font-size: 85px; text-align: center; color: green;'>JobIntel</h1>", unsafe_allow_html=True)
st.markdown("""
# Welcome to JobIntel: Your Personalized Job Explorer for Data Careers

### **JobIntel** is a tool designed to provide insights into top-paying roles across different data-driven 
professions such as **Data Science**, **Data Analytics**, **Data Engineering** and **Machine Learning**.

#### With JobIntel, you can explore:
- **Top-paying companies** across different job categories
- **Sector-specific companies** and insights
- Information like **company rating**, **revenue**, and **average salaries**
""")

company = job['Company Name'].sort_values(ascending=True).unique()

st.sidebar.header("Explore Data Careers")
radio = st.sidebar.radio("Select Stream", ['Data Scientist', 'Data Analyst', 'Data Engineer',
                                           'Machine Learning Engineer'])

job.drop_duplicates(keep='first', inplace=True)

job = job.drop(columns={'Salary Est.(By Glassdoor)', 'Founded', 'Industry', 'Size of Employees', 'Job Title'})

job.reset_index()
job.index = np.arange(1, len(job) + 1)

new = ['Company Name', 'Job Category', 'Sector', 'Type', 'Rating', 'Revenue', 'AVG. Salary in USD', 'Location',
       'Headquarters']
job = job.reindex(columns=new)

job['Rating'] = job['Rating'].replace(-1., '3.8')
job['Rating'] = job['Rating'].replace(4., '4.0')
job['Rating'] = job['Rating'].replace(5., '5.0')
job['Rating'] = job['Rating'].replace(3., '3.0')

if radio == 'Data Scientist':
    job['Rating'] = job['Rating'].replace('-1', '3.8')
    ds_5 = job[job['Job Category'] == 'Data Science'].sort_values(by='AVG. Salary in USD', ascending=False).head()
    ds_5.index = np.arange(1, len(ds_5) + 1)
    st.header("Top 5 Job Salary in Data Science")
    st.dataframe(ds_5)
elif radio == 'Data Analyst':
    da_5 = job[job['Job Category'] == 'Data Analytics'].sort_values(by='AVG. Salary in USD', ascending=False).head()
    da_5.index = np.arange(1, len(da_5) + 1)
    st.header("Top 5 Job Salary in Data Analytics")
    st.dataframe(da_5)
elif radio == 'Data Engineer':
    de_5 = job[job['Job Category'] == 'Data Engineering'].sort_values(by='AVG. Salary in USD', ascending=False).head()
    de_5.index = np.arange(1, len(de_5) + 1)
    st.header("Top 5 Job Salary in Data Engineering")
    st.dataframe(de_5)
else:
    ml_5 = job[job['Job Category'] == 'Machine Learning'].sort_values(by='AVG. Salary in USD', ascending=False).head()
    ml_5.index = np.arange(1, len(ml_5) + 1)
    st.header("Top 5 Job Salary in Machine Learning Engineer")
    st.dataframe(ml_5)

sorted_sectors = job['Sector'].unique()
sorted_sectors = sorted(sorted_sectors)

selected_sector = st.sidebar.selectbox('Select Sector', sorted_sectors)
sector_wise_5 = st.sidebar.radio("Do you want to see Sector Wise Top 5 Jobs?", ['Yes', 'No'])

st.header('Sector-wise Company List')
if selected_sector and sector_wise_5 == 'Yes':
    s1 = job[job['Sector'] == selected_sector].sort_values(by='AVG. Salary in USD', ascending=False).head()
    s1.index = np.arange(1, len(s1) + 1)
    st.dataframe(s1)
else:
    s1 = job[job['Sector'] == selected_sector]
    s1.index = np.arange(1, len(s1) + 1)
    st.dataframe(s1)

option = st.selectbox(
    "Choose Stream:",
    ('Data Science', 'Data Analytics', 'Data Engineering', 'Machine Learning'),
)

data = job[job['Job Category'] == option]

plt.figure(figsize=(10, 6))
sns.kdeplot(data['AVG. Salary in USD'], bw_adjust=0.2, fill=True)
plt.title('KDE Plot of Average Salary')
plt.xlabel('Average Salary in USD')
plt.ylabel('Density')

# Display the plot in Streamlit
st.pyplot(plt)
