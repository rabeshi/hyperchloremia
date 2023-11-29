
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import openai
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Set the page config
st.set_page_config(page_title='Hyperchloremia Analysis', layout='wide')


# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('hyperchloremia.csv')

df = load_data()

# Function to load the model
#def load_model():
#    with open('model.pkl', 'rb') as file:
#        model = pickle.load(file)
#    return model
# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title('Hyperchloremia Analysis')


# Sidebar for user input
st.sidebar.header('Hyperchloremia Diagnosis')
#st.sidebar.write(hyperchloremia_text)

# Sidebar content
st.sidebar.write('''<small>Hyperchloremia is a condition characterized by an elevated level of chloride in the blood. It can occur in adults with Diabetic Ketoacidosis (DKA), which is a severe complication of diabetes. DKA primarily affects individuals with type 1 diabetes, but it can also occur in those with type 2 diabetes.</small>''', unsafe_allow_html=True)

st.sidebar.markdown("""
  ---
""")

# Display citation on the sidebar
st.sidebar.markdown("""
    ### Team Members:
    
    - XXXXXXX
    
    ---
    ## Publication Example
    
    XXX, XX., & XXX, X. (2023). **Title of the Paper**. *Journal of XXX X*, 10(2), 123-145.
    
    DOI: [10.123456/example](https://doi.org/10.123456/example) [PDF](https://example.com/paper.pdf)

    ---
""")

feedback = st.sidebar.slider('Rate this app?', min_value=0, max_value=5, step=1)

if feedback:
    st.header("Thank you for rating the app!")
    st.info("Caution: This information is provided for informational purposes only and should not be considered as medical advice. If you are experiencing persistent symptoms, we strongly recommend consulting with a qualified healthcare professional or doctor for proper evaluation and guidance.")

st.sidebar.markdown("""
            Made at night by [![@george_obaido](https://img.shields.io/twitter/follow/Geobaido?style=social)](https://twitter.com/Geobaido)
            """,
            unsafe_allow_html=True
            )

# Tab creation
tab1, tab2, tab3 = st.tabs(["Visualization", "Model Analysis", "Clinical Notes"])

# Function to create bar chart
#def create_bar_chart(data, column, title):
#    count = data[column].value_counts().reset_index()
#    count.columns = [column, 'count']
#    fig = px.bar(count, x=column, y='count', title=title)
#    return fig


# Function to create histogram
def create_histogram(data, column, title):
    fig = px.histogram(data, x=column, title=title)
    return fig

def create_chlorine_histogram(data, chlorine_col, hyperchloremia_col, title):
    fig = px.histogram(data, x=chlorine_col, color=hyperchloremia_col, barmode='overlay', title=title)
    return fig

def create_gender_histogram(data, gender_col, hyperchloremia_col, title):
    fig = px.histogram(data, x=gender_col, color=hyperchloremia_col, barmode='overlay', title=title)
    return fig

def create_age_histogram(data, age_col, hyperchloremia_col, title):
    fig = px.histogram(data, x=age_col, color=hyperchloremia_col, barmode='overlay', title=title)
    return fig

def create_weight_histogram(data, weight_col, hyperchloremia_col, title):
    fig = px.histogram(data, x=weight_col, color=hyperchloremia_col, barmode='overlay', title=title)
    return fig


# Function to create scatter plot
#def create_scatter_plot(data, x_col, y_col, title):
#    fig = px.scatter(data, x=x_col, y=y_col, title=title)
#    return fig


# Function to create pie chart
#def create_pie_chart(data, column, title):
#    count = data[column].value_counts().reset_index()
#    count.columns = [column, 'count']
#    fig = px.pie(count, names=column, values='count', title=title)
#    return fig


# Function to create box plot
def create_box_plot(data, column, category, title):
    fig = px.box(data, x=category, y=column, title=title)
    return fig


# Display visualizations in three columns
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_gender_histogram(df, 'gender', 'Hyperchloremia', 'Distribution of Gender by Hyperchloremia'))

        st.plotly_chart(create_age_histogram(df, 'anchor_age', 'Hyperchloremia', 'Age Distribution by Hyperchloremia Status'))

        st.plotly_chart(create_box_plot(df, 'BMI (kg/m2)', 'Hyperchloremia', 'BMI Distribution by Hyperchloremia Status'))


    with col2:
        st.plotly_chart(create_histogram(df, 'BMI (kg/m2)', 'BMI Distribution'))

        st.plotly_chart(create_weight_histogram(df, 'Weight (Lbs)', 'Hyperchloremia', 'Weight (Lbs) Distribution by Hyperchloremia Status'))

        st.plotly_chart(create_chlorine_histogram(df, 'Chlorine', 'Hyperchloremia', 'Distribution of Chlorine by Hyperchloremia'))
        #st.plotly_chart(create_scatter_plot(df, 'BMI (kg/m2)', 'Creatinine', 'BMI vs. Creatinine Levels'))

        #st.markdown('#### Hyperchloremia Distribution')
        #st.plotly_chart(create_pie_chart(df, 'Hyperchloremia', 'Distribution of Hyperchloremia'))

        # Add more visualizations as per the data and requirements

with tab2:
        #st.header("Model Analysis")
        # Initialize LabelEncoders for categorical features
        le_gender = LabelEncoder()
        le_marital_status = LabelEncoder()
        le_race = LabelEncoder()
        le_diabetes_acidosis = LabelEncoder()
        le_diabetestype = LabelEncoder()
        le_sepsis = LabelEncoder()

        # This function should be adapted to match the exact preprocessing used during model training
        def preprocess_input(user_input):
            # Apply Label Encoding for all categorical features
            user_input['gender'] = le_gender.fit_transform(user_input['gender'].astype(str))
            user_input['marital_status'] = le_marital_status.fit_transform(user_input['marital_status'].astype(str))
            user_input['race'] = le_race.fit_transform(user_input['race'].astype(str))
            user_input['diabetestype'] = le_diabetestype.fit_transform(user_input['diabetestype'].astype(str))
            #user_input['sepsis'] = le_diabetestype.fit_transform(user_input['sepsis'].astype(str))
            
            # Preprocess the blood pressure
            #bp_values = user_input['Blood Pressure'].str.split('/', expand=True)
            #user_input['Systolic_BP'] = pd.to_numeric(bp_values[0], errors='coerce')
            #user_input['Diastolic_BP'] = pd.to_numeric(bp_values[1], errors='coerce')
            #user_input.drop('Blood Pressure', axis=1, inplace=True)

            return user_input

        def predict_hyperchloremia(input_data):
            processed_data = preprocess_input(input_data)
            prediction = model.predict(processed_data)
            return prediction

        # Streamlit app layout
        #st.title("Hyperchloremia Prediction")

        # Create two columns for input and output
        col1, col2 = st.columns(2)

        # Input features in the first column
        with col1:
            st.markdown("### Patient Information")
            
            # Input fields
            gender = st.selectbox("Gender", ['Male', 'Female'])
            age = st.slider("Age", min_value=18, max_value=91, value=30)
            marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Widowed', 'Separated', 'Unknown'])
            race = st.selectbox("Race", ['White', 'Black', 'Hispanic', 'Asian', 'Other'])
            weight = st.slider("Weight (Lbs)", min_value=0, max_value=357, value=20)
            height = st.slider("Height (Inches)", min_value=50, max_value=96, value=20)
            bmi = st.number_input("BMI (kg/m2)", min_value=0.0)
            #blood_pressure = st.text_input("Blood Pressure", "120/80")
            lengthstay = st.slider("Length of Stay (Days)", min_value=1, max_value=1859, value=50)
            diabetestype = st.selectbox("Diabetes Type", ['Type 1', 'Type 2', 'None'])

            anion_gap = st.slider("Anion Gap", min_value=10, max_value=30, value=10)
            bicarbonate_urine = st.slider("Bicarbonate Urine", min_value=5, max_value=15, value=5)
            chlorine = st.slider("Chlorine", min_value=90, max_value=120, value=10)
            creatinine = st.slider("Creatinine", min_value=0, max_value=12, value=2)
            creatinine_kinase = st.slider("Creatinine Kinase", min_value=0, max_value=30000, value=10)
            glucose_urine = st.slider("Glucose Urine",  min_value=5, max_value=3400, value=10)
            hemoglobin = st.slider("Hemoglobin", min_value=5, max_value=18, value=2)
            serum_creatinine = st.slider("Serum Creatinine", min_value=1, max_value=5, value=1)
            total_urine = st.slider("Total Urine", min_value=115, max_value=500, value=10)
            vancomycin = st.slider("Vancomycin", min_value=1, max_value=40, value=10)
            sepsis = st.number_input("Sepsis", min_value=0.0)
            systolic_bp = st.slider("Systolic BP", min_value=88, max_value=189, value=50)
            diastolic_bp = st.slider("Diastolic BP", min_value=42, max_value=120, value=50)


            # Collecting inputs into a dataframe
            input_data = pd.DataFrame([[gender, age, marital_status, race, weight, height, bmi, lengthstay, diabetestype, anion_gap, bicarbonate_urine, chlorine, creatinine, creatinine_kinase, glucose_urine, hemoglobin, serum_creatinine, total_urine, vancomycin, sepsis, systolic_bp, diastolic_bp]],
                                columns=['gender', 'anchor_age', 'marital_status', 'race', 'Weight (Lbs)', 'Height (Inches)', 'BMI (kg/m2)', 'lengthstay', 'diabetestype', 'Anion Gap', 'Bicarbonate urine', 'Chlorine', 'Creatinine', 'Creatinine Kinase', 'Glucose Urine', 'Hemoglobin', 'Serum Creatinine', 'Total Urine', 'Vancomycin', 'Sepsis', 'Systolic_BP', 'Diastolic_BP'])


        # Output in the second column
        with col2:
            st.markdown("### Prediction and Analysis")
            # Create and display a scatterpolar plot of the features
            fig = px.scatter_polar(input_data, r=input_data.values.flatten(), 
                                    theta=input_data.columns, 
                                    color_discrete_sequence=px.colors.sequential.Plasma_r)
            st.plotly_chart(fig)

            if input_data is not None:
                # Make prediction
                prediction = predict_hyperchloremia(input_data)

                # Display prediction
                if prediction[0] == 1:
                    st.error("The patient is likely to have hyperchloremia considering the factors.")
                else:
                    st.success("The patient is unlikely to have hyperchloremia considering the factors.")

with tab3:
        #st.header("Clinical Note")
        data = pd.read_csv('radiology_data.csv')

        # Function to query GPT API
        def query_gpt(prompt, model="text-davinci-003", temperature=0.7, max_tokens=150):
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].text.strip()

        # Set your API key
        openai.api_key = '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'

        # Streamlit App
        def main():
            st.markdown("### Clinical Note Analysis")

            # Interacting with individual notes
            note_id = st.selectbox("Select Note ID", data['note_id'].unique())
            selected_note = data[data['note_id'] == note_id]['text'].iloc[0]

            st.write("### Original Clinical Note:")
            st.write(selected_note)

            # Horizontal rule
            st.markdown("---")

            # Use Markdown for a compact, larger label
            st.markdown("#### Ask a question about this note:")
            question = st.text_area("")

            if st.button('Answer Question'):
                with st.spinner('Generating answer...'):
                    answer = query_gpt(selected_note + "\n\nQ: " + question + "\nA:")
                    st.write("### Answer:")
                    st.write(answer)

            # Rest of your code...

        if __name__ == "__main__":
            main()
