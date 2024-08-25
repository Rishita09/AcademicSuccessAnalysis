import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('new-student-data.csv')  
dfv = pd.read_csv("preprocessed_student_data.csv")

st.set_page_config(page_title="Predictive Analytics for Student Success", layout="wide")

st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ['Introduction', 'Problem Statement', 'About Dataset', 'Data Preparation','Model Performance', 'Data Visualization', 'Recommendations', 'Reflection', 'Team and Responsibilities'])

if options == 'Introduction':
    st.title("Predicting Student Success: A Comparative Analysis of Machine Learning Classifiers to Identify Key Performance Factors")
    st.title("1. Introduction")

    st.write("### Overview of the Project")
    st.write(
        """
        This project focuses on leveraging predictive analytics to enhance student success within educational institutions. 
        By analyzing historical student data, the goal is to identify patterns and trends that can help educators and administrators 
        make data-driven decisions to improve student outcomes.
        """
    )
    
    st.write("### Importance of Predictive Analytics in Education")
    st.write(
        """
        Predictive analytics in education allows institutions to anticipate student performance, identify at-risk students, 
        and allocate resources effectively. It provides insights that can help in improving student outcomes, optimizing curriculum design, 
        and enhancing overall educational quality.
        """
    )
    
    st.write("### Goals of the Project")
    st.write(
        """
        The primary goal is to develop a predictive model that can forecast student success based on various factors such as demographics, 
        academic performance, and extracurricular activities. This model will assist in identifying students who may need additional support 
        and help in devising strategies for their success.
        """
    )

    st.image("analytics.png") 

elif options == 'Problem Statement':
    
    st.write("### 2. Identify Domain: Understand and Frame the Problem")
    
    st.write("**a. Main Problem or Challenge in the Case Study:**")
    st.write(
        """
        The main problem in this case study is the prediction of student success in educational institutions.
        With a large number of students enrolled in various programs, identifying the factors that contribute 
        to student success and predicting which students are at risk of underperforming is a significant challenge.
        This is especially critical as educational institutions aim to improve student outcomes, graduation rates, 
        and overall academic performance.
        """
    )
    
    st.write("**b. Impact on the Organization or Stakeholders:**")
    st.write(
        """
        The problem directly impacts multiple stakeholders:
        
        - **Students:** Accurate predictions can help identify students who need additional support, thereby improving their academic outcomes.
        - **Educational Institutions:** By understanding and addressing factors influencing student success, institutions can improve their overall academic standing, reduce dropout rates, and allocate resources more effectively.
        - **Faculty and Administration:** Faculty can tailor their teaching methods and interventions based on predictive insights, while administrators can develop policies and programs aimed at improving student success rates.
        - **Parents and Guardians:** They gain insights into their children's academic progress and can work collaboratively with the institution to support their success.
        """
    )
    
    st.write("**c. Example of Trends and Patterns:**")
    st.write(
        """
        In the context of this educational case study, trends and patterns might include:
        
        - **Attendance and Engagement:** Analyzing how student attendance and participation in extracurricular activities impact their academic performance.
        - **Demographic Factors:** Understanding how factors such as socioeconomic status, geographical location, and parental education levels influence student success.
        - **Other factors:** Identifying how student grades are affected due to underlying causes such as alchohol consumption and health.
        """
    )



elif options == 'About Dataset':
    st.write("### 3. Collect and Preprocess Data")
   
    st.write("### Overview of the Dataset")
    st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns. It has a mix of the original dataset and synthetic data.")
    st.write("Here are the first few rows of the dataset:")
    st.dataframe(df.head())
    
    st.write("### Dataset Description")
    st.write(
        """
        The dataset contains the following features:
        
        - **school**: student's school (binary: "GP" or "MS")
        - **sex**: student's sex (binary: "F" - female or "M" - male)
        - **age**: student's age (numeric: from 15 to 22)
        - **address**: student's home address type (binary: "U" - urban or "R" - rural)
        - **famsize**: family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
        - **Pstatus**: parent's cohabitation status (binary: "T" - living together or "A" - apart)
        - **Medu**: mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)
        - **Fedu**: father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)
        - **Mjob**: mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
        - **Fjob**: father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
        - **reason**: reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
        - **guardian**: student's guardian (nominal: "mother", "father" or "other")
        - **traveltime**: home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
        - **studytime**: weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
        - **failures**: number of past class failures (numeric: n if 1<=n<3, else 4)
        - **schoolsup**: extra educational support (binary: yes or no)
        - **famsup**: family educational support (binary: yes or no)
        - **paid**: extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
        - **activities**: extra-curricular activities (binary: yes or no)
        - **nursery**: attended nursery school (binary: yes or no)
        - **higher**: wants to take higher education (binary: yes or no)
        - **internet**: Internet access at home (binary: yes or no)
        - **romantic**: with a romantic relationship (binary: yes or no)
        - **famrel**: quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
        - **freetime**: free time after school (numeric: from 1 - very low to 5 - very high)
        - **goout**: going out with friends (numeric: from 1 - very low to 5 - very high)
        - **Dalc**: workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
        - **Walc**: weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
        - **health**: current health status (numeric: from 1 - very bad to 5 - very good)
        - **absences**: number of school absences (numeric: from 0 to 93)
        - **passed**: did the student pass the final exam or not (binary: yes or no)
        """
    )
    
    st.write("### Statistical Summary")
    st.write(df.describe())

elif options == 'Data Preparation':
    st.write("### 4. Data Preparation")
    st.write("#### Mapping Categorical Data")
    st.write(
        """
        The dataset's categorical variables have been encoded into numerical values to facilitate machine learning processes. 
        This includes transforming school names, gender, address types, and various other categorical attributes into numerical formats.
        """
    )

    st.write("#### Feature Scaling")
    st.write(
        """
        To enhance the performance of machine learning algorithms, feature scaling has been applied. 
        This process normalizes the range of numerical features, ensuring that variables with larger scales do not dominate those with smaller scales. 
        Features with values above a certain threshold are standardized, while others are normalized to a range between 0 and 1.
        """
    )

    st.write("Here are the first few rows of the preprocessed dataset:")
    st.dataframe(dfv.head())

elif options == 'Data Visualization':
    st.write("### 6. Data Visualization")
    
    st.write("### Distribution of Features")
    selected_column = st.selectbox("Select a column to visualize:", df.columns)
    plt.figure(figsize=(10, 6))
    sns.histplot(df[selected_column], kde=True)
    st.pyplot(plt)

    st.write("### Correlation Matrix (Numeric Data Only)")
    
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if numeric_df.empty:
        st.write("No numeric columns available for correlation matrix.")
    else:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(plt)
    col1, col2 = st.columns(2)

    with col1:
        st.image("internet.png", caption="Student Status by Internet Accessibility", use_column_width=True)

    with col2:
        st.image("job.png", caption="Influence of Mother's Job", use_column_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.image("failure.png", caption="Student Status by Failure", use_column_width=True)

    with col4:
        st.image("health.png", caption="Student Status by Health", use_column_width=True)

    st.image("matrix.png", caption="Confusion Matrix and ROC Curve(Logistic Regression)", use_column_width=True)


elif options == 'Model Performance':
    st.write("### 5. Model Performance")
    st.subheader("Logistic Regression")
    st.write(
        """
        **Accuracy:** 59.47%  
        **F1 Score:** 0.41  
        Logistic Regression showed moderate performance with a balanced accuracy. 
        It performed better in detecting positive classes but struggled with the negative ones.
        """
    )

    st.subheader("XGBoost")
    st.write(
        """
        **Accuracy:** 60%  
        **F1 Score:** 0.60  
        XGBoost provided a better balance between precision and recall across classes. 
        It demonstrated stronger generalization and slightly improved overall performance.
        """
    )

    st.subheader("LightGBM")
    st.write(
        """
        **Accuracy:** 61%  
        **F1 Score:** 0.60  
        LightGBM also showed solid performance, particularly in recall for the positive class, 
        making it effective for imbalanced data scenarios.
        """
    )

    st.subheader("Voting Classifier")
    st.write(
        """
        **Accuracy:** 61%  
        **F1 Score:** 0.60  
        The Voting Classifier, which combines multiple models, matched LightGBM in accuracy 
        and overall performance. It leverages the strengths of various models to achieve a more robust prediction.
        """
    )

    st.subheader("Random Forest Classifier")
    st.write(
        """
        **Accuracy:** 70.2%  
        **F1 Score:** 0.57  
        The Random Forest Classifier outperformed the others in terms of accuracy. 
        It also provided a balanced precision and recall, making it the best performing model overall.
        """
    )
    st.subheader("Conclusion")
    st.write(
        """
        While the Random Forest Classifier had the highest accuracy, XGBoost and LightGBM were also strong contenders 
        with balanced F1 scores and strong generalization. Depending on this specific application, any of these models 
        could be considered, with the Random Forest being the overall best performer in this case.
        """
    )

# Recommendations
elif options == 'Recommendations':
    st.write("### 7. Recommendations to Improve Student Success and Model Performance")

    recommendations = """
    ##### 1. Promote Healthy Habits
    Encourage students to prioritize their physical and mental well-being. This includes getting enough sleep, eating nutritious food, and exercising regularly. Schools could offer workshops on stress management and healthy lifestyle choices.

    ##### 2. Address Internet Access Disparities
    Ensure that all students have reliable access to the internet. This may involve providing subsidized internet services or creating dedicated computer labs in schools.

    ##### 3. Targeted Support for Struggling Students
    Identify students who are at risk of failing and provide them with additional support. This could include tutoring, mentoring, or counseling services.

    ##### 4. Encourage Parental Involvement
    Foster strong partnerships between parents and schools. This could involve regular communication about student progress, workshops for parents on how to support their children's education, and opportunities for parents to volunteer at school.

    ##### 5. Optimize Study Habits
    Encourage students to develop effective study habits. This includes setting realistic goals, managing time effectively, and creating a conducive study environment. Schools could offer workshops on study skills and time management techniques.

    ##### 6. Address Social and Emotional Factors
    Recognize that social and emotional factors can significantly impact academic performance. Provide support for students experiencing difficulties with family, relationships, or mental health.

    ##### 7. Promote a Growth Mindset
    Encourage students to believe that their abilities can be developed through hard work and dedication. This can be achieved through positive reinforcement, feedback that focuses on effort and progress, and opportunities for students to learn from their mistakes.

    ##### 8. Personalized Learning
    Tailor educational approaches to individual student needs and learning styles. This could involve offering a variety of learning resources, using technology to personalize learning experiences, and providing flexible learning pathways.

    ##### 9. Early Intervention
    Identify and address learning difficulties early on. This could involve regular assessments, screening for learning disabilities, and providing timely interventions.

    ##### 10. Focus on Engagement
    Create a learning environment that is engaging and stimulating for students. This could involve using active learning techniques, incorporating real-world examples, and providing opportunities for students to collaborate and participate in discussions.
    
    ##### Additional Technical Recommendations for Improving Model Performance

    * **Feature engineering:** Can explore creating new features or transforming existing ones to capture more relevant information for the models.
    * **Hyperparameter optimization:** Fine-tune the model parameters using techniques like GridSearchCV or RandomizedSearchCV to find the optimal settings for our data.
    * **Ensemble methods:** Combine multiple models to leverage their strengths and potentially improve overall performance.
    * **Address class imbalance:** Consider using more techniques to balance the classes and improve model performance on the minority class.
    * **Data quality:** Using more specific preprocessing techniques relevant to out dataset.
    * **Regular evaluation:** Continuously monitor and evaluate the model's performance on new data to identify potential areas for improvement.
    
    
    """

    st.markdown(recommendations)

# Reflection
elif options == 'Reflection':
    st.write("### 8. Reflection")
    st.write("**Challenges Faced:** - Data Quality: Handling missing and inconsistent data. - Model Tuning: Balancing model accuracy and interpretability.")
    st.write("**Future Directions:** - Explore additional features that may improve model performance. - Conduct longitudinal studies to track student success over time.")

elif options == "Team and Responsibilities":
    st.write("### 9. Team Roles and Responsibilities")

    st.write('Hitesh Salimath     2347118')
    st.write('Keerthana C     2347134')
    st.write('Rishita Shah    2347143')
    st.write('Kalpana N   2347229')

    roles_and_responsibilities = """
    #### 1. Team Roles and Responsibilities

    **Hitesh and Rishita**:
    - **Roles:** Collection and knowledge generation
    - **Responsibilities:** Collecting data from various sources, generating insights, and providing foundational knowledge necessary for the project.

    **Keerthana and Kalpana**:
    - **Roles:** Implementation and Visualization
    - **Responsibilities:** Implementing models and algorithms, creating visualizations to represent data and results effectively.

    **Planning, Monitoring, and Report Preparation**:
    - **Roles:** All team members
    - **Responsibilities:** Jointly responsible for planning project activities, monitoring progress, and preparing final reports.

    #### 2. Monitoring Progress

    Progress is monitored through:
    - **Regular Team Meetings:** Scheduled meetings to review progress, address issues, and adjust plans as necessary.
    - **Project Management Tools:** Utilization of tools like Trello or Asana to track tasks, deadlines, and milestones.
    - **Progress Reports:** Weekly or bi-weekly reports summarizing accomplishments, challenges, and next steps.

    #### 3. Measuring Success of Completed Project

    Success is measured by:
    - **Achievement of Project Goals:** Assessing whether the project objectives and deliverables have been met.
    - **Quality of Deliverables:** Evaluating the quality and accuracy of the models, visualizations, and reports.
    - **Stakeholder Feedback:** Gathering feedback from stakeholders or evaluators to gauge the effectiveness and impact of the project.
    - **Performance Metrics:** Analyzing performance metrics such as model accuracy, F1 score, and other relevant KPIs.

    #### 4. Professional Strengths and Weaknesses

    **Strengths:**
    - **Hitesh and Rishita:** Strong analytical skills and knowledge generation abilities contribute to a solid data foundation and insightful findings.
    - **Keerthana and Kalpana:** Proficiency in implementation and visualization ensures effective model deployment and clear presentation of results.
    - **Overall Team:** Collaborative approach and diverse skill set enhance project execution and problem-solving.

    **Weaknesses:**
    - **Potential Skill Gaps:** Identified gaps in specific technical skills or knowledge areas may need to be addressed through additional training or external expertise.
    - **Resource Constraints:** Limited resources or time constraints could impact the depth of analysis or quality of deliverables.

    **Relevance to Completion:**
    - **Strengths** support efficient project execution and high-quality results, while **weaknesses** highlight areas for improvement. Addressing weaknesses proactively ensures that the project is completed successfully and meets the desired objectives.
    """

    st.markdown(roles_and_responsibilities)

