# Streamlit app
# in conda env my_streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
from scipy.interpolate import make_interp_spline
import time
import altair as alt


# App Title
st.title("Post-Surgery Outcome Predictor for BMI and Type 2 Diabetes")

# Check session state for data_shared and visualize_bmi
if "data_shared" not in st.session_state:
    st.session_state["data_shared"] = False
if "visualize_bmi" not in st.session_state:
    st.session_state["visualize_bmi"] = False

# Display introductory message only if data has not been shared
if not st.session_state["data_shared"]:
    st.write("## Please share Patients medical data")
    st.write("Fill out the details in the sidebar to calculate and visualize post-surgery BMI progression.")

# Initialize Patients_data in session state
if 'Patients_data' not in st.session_state:
    st.session_state['Patients_data'] = pd.DataFrame()  # Create an empty DataFrame

# Initialize default values for variables to prevent errors
birthday_formatted = "Not provided"
age = None

# Set the minimum and maximum date range
min_date = date(1900, 1, 1)
max_date = datetime.now().date()

# Input: Birthday
birthday = st.sidebar.date_input(
    "Enter your birthday:",
    value=None,  # Default value
    min_value=min_date,  # Allow dates back to 1900
    max_value=max_date,  # Prevent future dates
)

if birthday:
    birthday_formatted = birthday.strftime("%d.%m.%Y")
    current_date = datetime.now().date()
    age = (current_date - birthday).days // 365
    st.sidebar.write(f"Calculated Age: {age} years")
else:
    age = None
    st.sidebar.write("No birthday selected.")

# Input: BMI
bmi = st.sidebar.number_input("Enter BMI (Body Mass Index):", min_value=0.0, step=0.1, format="%.1f")

# Input: Sex
sex = st.sidebar.radio(
    "Select Sex:",
    options=["Select an option", "Male", "Female"],
)

if sex == "Select an option":
    sex = None

# Input: Binary data
st.sidebar.subheader("Medical Conditions (Yes/No):")
hypertension = st.sidebar.checkbox("Hypertension")
diabetes = st.sidebar.checkbox("Diabetes Mellitus Type 2")
osas = st.sidebar.checkbox("Obstructive Sleep Apnea Syndrome (OSAS)")
depression = st.sidebar.checkbox("Depression")
hyperlipidemia = st.sidebar.checkbox("Hyperlipidemia")

# Input: Planned surgery
surgery = st.sidebar.radio(
    "Planned Surgery:",
    options=["Select an option", "Laparoscopic Sleeve Gastrectomy (LSG)", "Laparoscopic Roux-en-Y Gastric Bypass (LRYGB)"],
)

if surgery == "Select an option":
    surgery = None

# Add 'All Data Provided' button
if st.sidebar.button("All Data Provided"):
    # Check if all required fields are filled
    if age is None or bmi == 0.0 or sex is None or surgery is None:
        st.warning("Please fill out all required information in the sidebar.")
    else:
        # Comment based on age and BMI
        if age < 18 and bmi > 30:
            st.info("Note: The patient's AGE is outside the range of the training data used for the machine learning algorithm. The prediction may not be reliable.")
        elif age >= 18 and bmi < 30:
            st.info("Note: The patient's BMI is outside the range of the training data used for the machine learning algorithm. The prediction may not be reliable.")
        elif age < 18 and bmi < 30:
            st.info(
                "Note: The patient's AGE and BMI is outside the range of the training data used for the machine learning algorithm. The prediction may not be reliable.")
        else:
            st.success("The provided data falls within the range of the training dataset.")

        # Mark data as shared
        st.session_state["data_shared"] = True

        # Save patient data to session state
        data = {
            'sex (1 = female, 2=male)': [1 if sex == "Female" else 2],
            'age_years': [age],
            'hypertension': [int(hypertension)],
            'hyperlipidemia': [int(hyperlipidemia)],
            'depression': [int(depression)],
            'DMII_preoperative': [int(diabetes)],
            'osas_preoperative': [int(osas)],
            'surgery': [surgery],
            'BMI before surgery': [bmi]
        }
        Patients_data = pd.DataFrame(data)
        st.session_state['Patients_data'] = Patients_data

# Display Input Summary and Visualization Button
if st.session_state["data_shared"]:
    st.header("Input Summary")
    st.write(f"**Age**: {age} years (Birthday: {birthday_formatted})")
    st.write(f"**BMI**: {bmi}")
    st.write(f"**Sex**: {sex}")
    st.write("**Medical Conditions**:")
    st.write(f"- Hypertension: {'Yes' if hypertension else 'No'}")
    st.write(f"- Diabetes Mellitus Type 2: {'Yes' if diabetes else 'No'}")
    st.write(f"- OSAS: {'Yes' if osas else 'No'}")
    st.write(f"- Depression: {'Yes' if depression else 'No'}")
    st.write(f"- Hyperlipidemia: {'Yes' if hyperlipidemia else 'No'}")
    st.write(f"**Planned Surgery**: {surgery}")

    if st.button("Visualize BMI Drop"):
        st.session_state["visualize_bmi"] = True

#import altair as alt

# Visualization Part
if st.session_state["visualize_bmi"]:
    import altair as alt
    import time

    # Function for predicting BMI
    def calculate_bmi_over_years(bmi_before_surgery):
        bmi_drop_percentages = {1: 0.60, 2: 0.40, 4: 0.30, 5: 0.35}
        bmi_over_years = {0: bmi_before_surgery}  # Add year 0 with initial BMI
        for year, drop in bmi_drop_percentages.items():
            bmi_over_years[year] = bmi_before_surgery * (1 - drop)
        return bmi_over_years

    # Simulate BMI drop including year 0
    bmi_before_surgery = st.session_state['Patients_data']['BMI before surgery'].iloc[0]
    bmi_over_years = calculate_bmi_over_years(bmi_before_surgery)
    years = np.array(list(bmi_over_years.keys()))
    bmi_values = np.array(list(bmi_over_years.values()))

    # Hard-coded MAE values
    mae_values = np.array([1.5, 1.9, 2.3, 3.2, 3.5])  # Adjust these as needed

    # Calculate upper and lower bounds for MAE
    bmi_upper = bmi_values + mae_values
    bmi_lower = bmi_values - mae_values

    # Create a smoother curve using interpolation
    smooth_years = np.linspace(years.min(), 5, 300)  # Ensure interpolation extends to year 5
    smooth_bmi_values = make_interp_spline(years, bmi_values)(smooth_years)
    smooth_bmi_upper = make_interp_spline(years, bmi_upper)(smooth_years)
    smooth_bmi_lower = make_interp_spline(years, bmi_lower)(smooth_years)

    # Healthy BMI Thin Line
    healthy_bmi_line = pd.DataFrame({
        "Year": smooth_years,
        "BMI": [25] * len(smooth_years),
        "Line": ["Healthy BMI"] * len(smooth_years)
    })

    # Prepare data for Altair chart
    chart_data = pd.DataFrame({
        "Year": np.concatenate([smooth_years, smooth_years]),
        "BMI": np.concatenate([smooth_bmi_upper, smooth_bmi_lower]),
        "Line": ["Mean Absolute Error"] * len(smooth_years) + ["Mean Absolute Error"] * len(smooth_years)
    })
    bmi_chart_data = pd.DataFrame({
        "Year": smooth_years,
        "BMI": smooth_bmi_values,
        "Line": ["Predicted BMI"] * len(smooth_years)
    })

    # Animation using Streamlit
    st.header("Predicted BMI Progression After Surgery")
    st.write("In the graph, the predicted BMI outcome up to 5 years post-surgery is displayed.")
    st.markdown("""
    <div style="font-size:16px;">
        <span style="color:blue;">Predicted BMI Progression</span> is shown in blue, 
        <span style="color:orange;">Mean Absolute Error (MAE)</span> is indicated in orange, 
        and the threshold to a <span style="color:green;">Healthy BMI (<25)</span> is highlighted in green.
    </div>
    """, unsafe_allow_html=True)

    chart_placeholder = st.empty()  # Placeholder for the chart

    # Build animated chart
    total_steps = len(smooth_years)  # Total steps for smooth animation
    for i in range(1, total_steps + 1, max(1, total_steps // 100)):  # Adjust steps for full range
        # Update chart dynamically
        current_smooth_years = smooth_years[:i]
        current_smooth_bmi_values = smooth_bmi_values[:i]
        current_smooth_bmi_upper = smooth_bmi_upper[:i]
        current_smooth_bmi_lower = smooth_bmi_lower[:i]

        # Create data for dynamic charts
        current_chart_data = pd.DataFrame({
            "Year": np.concatenate([current_smooth_years, current_smooth_years]),
            "BMI": np.concatenate([current_smooth_bmi_upper, current_smooth_bmi_lower]),
            "Line": ["Mean Absolute Error"] * len(current_smooth_years) + ["Mean Absolute Error"] * len(current_smooth_years)
        })
        current_bmi_chart_data = pd.DataFrame({
            "Year": current_smooth_years,
            "BMI": current_smooth_bmi_values,
            "Line": ["Predicted BMI"] * len(current_smooth_years)
        })

        # Create Altair charts
        bmi_chart = alt.Chart(current_bmi_chart_data).mark_line(
            strokeWidth=3  # Change size of Predicted BMI line
        ).encode(
            x=alt.X("Year", title="Years After Surgery"),
            y=alt.Y("BMI", title="Body Mass Index (BMI)"),
            color=alt.Color(
                "Line",
                scale=alt.Scale(
                    domain=["Predicted BMI", "Mean Absolute Error", "Healthy BMI"],
                    range=["blue", "orange", "green"]  # Explicit color mapping
                ),
                legend=alt.Legend(title="Legend")
            ),
            tooltip=["Year", "BMI"]
        )

        bounds_chart = alt.Chart(current_chart_data).mark_line(
            strokeDash=[2, 2],  # Change dash style of MAE lines
            strokeWidth=0.5  # Change size of MAE lines
        ).encode(
            x="Year",
            y="BMI",
            color=alt.Color(
                "Line",
                scale=alt.Scale(
                    domain=["Predicted BMI", "Mean Absolute Error", "Healthy BMI"],
                    range=["blue", "orange", "green"]  # Explicit color mapping
                ),
                legend=None
            ),
            tooltip=["Year", "BMI"]
        )

        healthy_line_chart = alt.Chart(healthy_bmi_line[:i]).mark_line(
            strokeWidth=1.5  # Change size of Healthy BMI line
        ).encode(
            x="Year",
            y="BMI",
            color=alt.Color(
                "Line",
                scale=alt.Scale(
                    domain=["Predicted BMI", "Mean Absolute Error", "Healthy BMI"],
                    range=["blue", "orange", "green"]  # Explicit color mapping
                ),
                legend=None
            ),
            tooltip=["Year", "BMI"]
        )

        # Combine charts and display
        final_chart = (bounds_chart + healthy_line_chart + bmi_chart).properties(
            width=700, height=400
        ).configure_legend(
            orient="bottom",  # Explicitly place legend below the chart
            title=None,
            labelFontSize=12,
            padding=10  # Add spacing between legend and chart
        ).configure_view(
            strokeWidth=1
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )

        chart_placeholder.altair_chart(final_chart)
        time.sleep(0.1)  # Animation delay

    # Final Display: Table
    st.write("### Final BMI Progression")
    bmi_data_table = pd.DataFrame({
        "Year": years,
        "BMI": bmi_values,
        "BMI Upper Bound": bmi_upper,
        "BMI Lower Bound": bmi_lower
    })
    st.write(bmi_data_table)

