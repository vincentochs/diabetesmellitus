# -*- coding: utf-8 -*-


###############################################################################
# Load libraries

# App
import streamlit as st
import altair as alt
from streamlit_carousel import carousel

# Utils
import pandas as pd
import pickle as pkl
import numpy as np
import os
import pickle
import time
from scipy.interpolate import make_interp_spline

# Format of numbers
print('Libraries loaded')

###############################################################################
# Make a dictionary of categorical features
dictionary_categorical_features = {'sex (1 = female, 2=male)' : {'Male' : 2,
                                                                 'Female' : 1},
                                   'prior_abdominal_surgery' :  {'Yes' : 1,
                                                                 'No' : 0},
                                   'hypertension' : {'Yes' : 1,
                                                     'No' : 0},
                                   'hyperlipidemia' : {'Yes' : 1,
                                                       'No' : 0},
                                   'depression' :  {'Yes' : 1,
                                                    'No' : 0},
                                   'DMII_preoperative' : {'Yes' : 1,
                                                          'No' : 0},
                                   'antidiab_drug_preop_Oral_anticogulation' : {'Yes' : 1,
                                                                                'No' : 0},
                                   'antidiab_drug_preop_Insulin' : {'Yes' : 1,
                                                                    'No' : 0},
                                   'osas_preoperative' : {'Yes' : 1,
                                                          'No' : 0},
                                   'surgery' : {'Laparoscopic Sleeve Gastrectomy (LSG)' : 1,
                                                'Laparoscopic Roux-en-Y Gastric Bypass (LRYGB)' : 2},
                                   'normal_dmII_pattern' : {'Yes' : 1,
                                                            'No' : 0},
                                   'antidiab_drug_preop_no_therapy' : {'Yes' : 1,
                                                                       'No' : 0},
                                   'antidiab_drug_preop_glp1_analogen' : {'Yes' : 1,
                                                                          'No' : 0},
                                   }

inverse_dictionary = {feature: {v: k for k, v in mapping.items()}
                      for feature, mapping in dictionary_categorical_features.items()}
# MAE from training notebook for make confident intervals
training_mae = 2.5

##############################################################################
# Section when the app initialize and load the required information
@st.cache_data() # We use this decorator so this initialization doesn't run every time the user change into the page
def initialize_app():
    # Load Regression Model
    # IMPORTANT: Ensure your model files are in a 'models' subdirectory.
    model_path_reg = os.path.join('003_Regression_Model_App.sav')
    if not os.path.exists(model_path_reg):
        st.error(f"Regression model not found at {model_path_reg}. Please ensure the 'models' directory and its contents are correctly placed.")
        st.stop()
    with open(model_path_reg , 'rb') as export_model:
        regression_model = pickle.load(export_model)

    # Load Classification Model
    model_path_clf = os.path.join('003_Classification_Model_App.sav')
    if not os.path.exists(model_path_clf):
        st.error(f"Classification model not found at {model_path_clf}. Please ensure the 'models' directory and its contents are correctly placed.")
        st.stop()
    with open(model_path_clf, 'rb') as export_model:
        classification_model = pickle.load(export_model)

    print('App Initialized correctly!')

    return regression_model, classification_model

###############################################################################
# Generic function to create animated chart for BMI or Weight
def create_animated_chart(summary_df, y_col, y_title, ci_lower_col, ci_upper_col, target_line_val=None, target_line_color='green'):
    """
    Creates an animated Altair chart for BMI, Weight, or % Weight Loss evolution.
    """
    st.subheader(f"Predicted {y_title} Evolution")

    # Chart placeholder
    chart_placeholder = st.empty()

    time_map = {'Pre': 0, '3m': 3, '6m': 6, '12m': 12, '18m': 18, '2y': 24, '3y': 36, '4y': 48, '5y': 60}
    time_points_values = list(time_map.values())

    # Get values and confidence intervals for the selected column
    y_values = summary_df[y_col].round(2).values
    ci_lower_values = summary_df[ci_lower_col].round(2).values
    ci_upper_values = summary_df[ci_upper_col].round(2).values

    # Calculate y-axis domain
    y_min = min(summary_df[ci_lower_col].min(), 0) * 0.9
    y_max = summary_df[ci_upper_col].max() * 1.1

    # Pre-calculate all smooth curves for animation
    total_steps = 120 # Higher number for smoother animation
    smooth_months = np.linspace(0, 60, total_steps)
    smooth_y_values = make_interp_spline(time_points_values, y_values)(smooth_months)
    smooth_ci_lower = make_interp_spline(time_points_values, ci_lower_values)(smooth_months)
    smooth_ci_upper = make_interp_spline(time_points_values, ci_upper_values)(smooth_months)

    legend_html = f"""
        <div style="font-size:14px; text-align: center; margin-bottom: 10px;">
            <span style="color:#0068c9;">■</span> Predicted {y_col} & 95% CI
            <span style="color:{target_line_color};">- - -</span> Target
    """
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)


    # Build animated chart
    for i in range(1, total_steps + 1):
        # Create data for the current animation frame
        current_data = pd.DataFrame({
            'Months': smooth_months[:i],
            y_col: smooth_y_values[:i],
            'CI_Lower': smooth_ci_lower[:i],
            'CI_Upper': smooth_ci_upper[:i]
        })

        current_data['Months'] = current_data['Months'].round(2)
        current_data[y_col] = current_data[y_col].round(2)
        current_data['CI_Lower'] = current_data['CI_Lower'].round(2)
        current_data['CI_Upper'] = current_data['CI_Upper'].round(2)

        # Main prediction line
        line = alt.Chart(current_data).mark_line(
            color='#0068c9',
            size=3
        ).encode(
            x=alt.X('Months', title='Postoperative Months', scale=alt.Scale(domain=[0, 60])),
            y=alt.Y(y_col, title=y_title, scale=alt.Scale(domain=[y_min, y_max]))
        )

        # Confidence interval area
        area = alt.Chart(current_data).mark_area(
            opacity=0.3,
            color='#0068c9'
        ).encode(
            x='Months',
            y='CI_Lower',
            y2='CI_Upper'
        )

        # Combine charts
        combined_chart = area + line

        # Add target line if specified
        if target_line_val is not None:
            target_line = alt.Chart(pd.DataFrame({'y': [target_line_val]})).mark_rule(color=target_line_color, strokeDash=[3,3], size=2).encode(y='y')
            combined_chart += target_line

        final_chart = combined_chart.properties(
            width=600,
            height=400
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )

        chart_placeholder.altair_chart(final_chart, use_container_width=True)
        time.sleep(0.01)

# Added height_m parameter to calculate weight
def parser_user_input(dataframe_input, reg_model, clf_model, height_m):
    """
    Parses user input, runs predictions, and returns a comprehensive dataframe.
    """
    # Encode categorical features
    for i in dictionary_categorical_features.keys():
        if i in dataframe_input.columns:
            dataframe_input[i] = dataframe_input[i].map(dictionary_categorical_features[i])

    predictions_df_regression = pd.DataFrame(reg_model.predict(dataframe_input[reg_model.feature_names_in_.tolist()]))
    predictions_df_regression.columns = ['bmi3','bmi6','bmi12','bmi18','bmi2y','bmi3y','bmi4y','bmi5y']


    # Apply adjustments based on comorbidities, making weight loss slightly less effective
    if dataframe_input['hypertension'].values[0] == 1:
        # Increase BMI slightly over time, simulating less effective weight loss
        adjustment_factors = np.linspace(1, 1.13, len(predictions_df_regression.columns)) # from 100% to 113%
        predictions_df_regression *= adjustment_factors

    if dataframe_input['hyperlipidemia'].values[0] == 1:
        adjustment_factors = np.linspace(1, 1.125, len(predictions_df_regression.columns)) # from 100% to 112.5%
        predictions_df_regression *= adjustment_factors

    if dataframe_input['osas_preoperative'].values[0] == 1:
        adjustment_factors = np.linspace(1, 1.12, len(predictions_df_regression.columns)) # from 100% to 112%
        predictions_df_regression *= adjustment_factors
        
    if dataframe_input['sex (1 = female, 2=male)'].values[0] == 1:
        adjustment_factors = np.linspace(1, 1.10, len(predictions_df_regression.columns)) # from 100% to 110%
        predictions_df_regression *= adjustment_factors



    # Adjust BMI curve based on surgery effectiveness
    surgery_name = dataframe_input['surgery'].values[0]
    if surgery_name == 2: # LRYGB is more effective in this model's data
        predictions_df_regression *= 0.85

    # Classification part
    df_classification = pd.concat([dataframe_input, predictions_df_regression], axis=1)

    _x = clf_model.predict_proba(df_classification[clf_model.feature_names_in_.tolist()])
    _x = [value[0][1] for value in _x]
    probas_df_classification = pd.DataFrame([_x])
    probas_df_classification.columns = ['dm3m_prob','dm6m_prob','dm12m_prob','dm18m_prob','dm2y_prob', 'dm3y_prob','dm4y_prob','dm5y_prob']

    predictions_df_classification = (probas_df_classification > 0.5).astype(int)
    predictions_df_classification.columns = ['dm3m','dm6m','dm12m','dm18m','dm2y','dm3y' , 'dm4y','dm5y']

    df_final = pd.concat([df_classification, predictions_df_classification, probas_df_classification], axis=1)
    df_final['DMII_preoperative_prob'] = df_final['DMII_preoperative'].astype(float)

    ###########################################################################
    # BMI-based DM probability adjustment based on BMI variation between time steps

    # Only apply BMI-based adjustments if patient has preoperative diabetes
    if df_final['DMII_preoperative'].iloc[0] == 1:

        # Function to calculate adjusted probability based on BMI change
        def adjust_probability_by_bmi_change_enhanced(bmi_change, current_prob, current_bmi, surgery_type,
                                                  time_step, medications, base_adjustment=0.05):
            """
            Enhanced function to adjust DM probability based on BMI change, surgery type, and medications
            """

            # Surgery-specific BMI thresholds
            surgery_thresholds = {
                1: {'3m': 35.0, '6m': 32.0, '12m': 30.0, '18m': 28.0, '2y': 27.0, '3y': 26.0, '4y': 25.0, '5y': 25.0},
                2: {'3m': 33.0, '6m': 30.0, '12m': 28.0, '18m': 26.0, '2y': 25.0, '3y': 24.0, '4y': 23.0, '5y': 23.0},
                5: {'3m': 34.0, '6m': 31.0, '12m': 29.0, '18m': 27.0, '2y': 26.0, '3y': 25.0, '4y': 24.0, '5y': 24.0}
            }

            time_step_map = {'dm3m_prob': '3m', 'dm6m_prob': '6m', 'dm12m_prob': '12m', 'dm18m_prob': '18m', 'dm2y_prob': '2y', 'dm3y_prob': '3y', 'dm4y_prob': '4y', 'dm5y_prob': '5y'}

            surgery_key = surgery_type if surgery_type in surgery_thresholds else 1
            time_key = time_step_map.get(time_step, '3m')
            threshold_bmi = surgery_thresholds[surgery_key].get(time_key, 30.0)

            if current_bmi <= threshold_bmi:
                bmi_difference = threshold_bmi - current_bmi
                effectiveness_multiplier = {1: 1.30, 2: 2.5, 5: 1.15}.get(surgery_type, 1.30)
                reduction_factor = min(base_adjustment * (1 + bmi_difference / 10) * effectiveness_multiplier, 0.8)
                bmi_adjusted_prob = current_prob * (1 - reduction_factor)
            else:
                bmi_difference = current_bmi - threshold_bmi
                increase_factor = min(base_adjustment * (bmi_difference / 10), 0.3)
                bmi_adjusted_prob = current_prob * (1 + increase_factor)

            medication_reduction = 0.0
            if medications.get('antidiab_drug_preop_glp1_analogen', 0) == 1: medication_reduction += 0.15
            if medications.get('antidiab_drug_preop_Oral_anticogulation', 0) == 1: medication_reduction += 0.10
            if medications.get('antidiab_drug_preop_Insulin', 0) == 1: medication_reduction += 0.08

            final_adjusted_prob = bmi_adjusted_prob * (1 - medication_reduction)
            return max(0.0, min(1.0, final_adjusted_prob))

        def apply_enhanced_bmi_adjustments(df_final, clf_model):
            """
            Apply enhanced BMI adjustments with surgery-specific thresholds and medication effects
            """
            bmi_transitions = [
                {'prev_bmi': 'BMI before surgery', 'current_bmi': 'bmi3', 'prob_col': 'dm3m_prob'},
                {'prev_bmi': 'bmi3', 'current_bmi': 'bmi6', 'prob_col': 'dm6m_prob'},
                {'prev_bmi': 'bmi6', 'current_bmi': 'bmi12', 'prob_col': 'dm12m_prob'},
                {'prev_bmi': 'bmi12', 'current_bmi': 'bmi18', 'prob_col': 'dm18m_prob'},
                {'prev_bmi': 'bmi18', 'current_bmi': 'bmi2y', 'prob_col': 'dm2y_prob'},
                {'prev_bmi': 'bmi2y', 'current_bmi': 'bmi3y', 'prob_col': 'dm3y_prob'},
                {'prev_bmi': 'bmi3y', 'current_bmi': 'bmi4y', 'prob_col': 'dm4y_prob'},
                {'prev_bmi': 'bmi4y', 'current_bmi': 'bmi5y', 'prob_col': 'dm5y_prob'}
            ]

            if df_final['DMII_preoperative'].iloc[0] == 1:
                surgery_type = df_final['surgery'].iloc[0]
                medications = {k: df_final[k].iloc[0] for k in ['antidiab_drug_preop_glp1_analogen', 'antidiab_drug_preop_Oral_anticogulation', 'antidiab_drug_preop_Insulin', 'antidiab_drug_preop_no_therapy']}

                for transition in bmi_transitions:
                    prev_bmi = df_final[transition['prev_bmi']].iloc[0]
                    current_bmi = df_final[transition['current_bmi']].iloc[0]
                    current_prob = df_final[transition['prob_col']].iloc[0]
                    bmi_change = current_bmi - prev_bmi

                    adjusted_prob = adjust_probability_by_bmi_change_enhanced(bmi_change, current_prob, current_bmi, surgery_type, transition['prob_col'], medications)

                    comorbidity_factor = 1.0
                    if df_final['hypertension'].iloc[0] == 1: comorbidity_factor *= 1.35
                    if df_final['hyperlipidemia'].iloc[0] == 1: comorbidity_factor *= 1.20
                    if df_final['osas_preoperative'].iloc[0] == 1: comorbidity_factor *= 1.10

                    final_adjusted_prob = min(1.0, adjusted_prob * comorbidity_factor)
                    df_final.loc[0, transition['prob_col']] = final_adjusted_prob
                    df_final.loc[0, transition['prob_col'].replace('_prob', '')] = 1 if final_adjusted_prob > 0.5 else 0

            return df_final
        df_final = apply_enhanced_bmi_adjustments(df_final, clf_model)

    # Confidence interval creation
    bmi_columns = ['BMI before surgery', 'bmi3', 'bmi6', 'bmi12', 'bmi18', 'bmi2y', 'bmi3y', 'bmi4y', 'bmi5y']

    for col in bmi_columns:
        df_final[f'{col}_ci_lower'] = df_final[col] - training_mae
        df_final[f'{col}_ci_upper'] = df_final[col] + training_mae

    # Create a clean summary dataframe
    summary_df = pd.DataFrame({'Time': ['Pre', '3m', '6m', '12m', '18m', '2y', '3y', '4y', '5y']})
    summary_df['BMI'] = df_final[bmi_columns].iloc[0].values
    summary_df['BMI CI Lower'] = df_final[[f'{c}_ci_lower' for c in bmi_columns]].iloc[0].values
    summary_df['BMI CI Upper'] = df_final[[f'{c}_ci_upper' for c in bmi_columns]].iloc[0].values

    # Calculate predicted weight if height is available
    if height_m > 0:
        summary_df['Weight'] = summary_df['BMI'] * (height_m ** 2)
        summary_df['Weight CI Lower'] = summary_df['BMI CI Lower'] * (height_m ** 2)
        summary_df['Weight CI Upper'] = summary_df['BMI CI Upper'] * (height_m ** 2)

        # --- NEW: CALCULATE % WEIGHT LOSS ---
        initial_weight = summary_df['Weight'].iloc[0]
        initial_weight_ci_lower = summary_df['Weight CI Lower'].iloc[0]
        initial_weight_ci_upper = summary_df['Weight CI Upper'].iloc[0]

        # Calculate percentage weight loss
        summary_df['Perc Weight Loss'] = ((initial_weight - summary_df['Weight']) / initial_weight) * 100 if initial_weight > 0 else 0

        # For CI, lower weight loss corresponds to upper weight prediction and vice-versa
        summary_df['Perc Weight Loss CI Lower'] = ((initial_weight_ci_upper - summary_df['Weight CI Upper']) / initial_weight_ci_upper) * 100 if initial_weight_ci_upper > 0 else 0
        summary_df['Perc Weight Loss CI Upper'] = ((initial_weight_ci_lower - summary_df['Weight CI Lower']) / initial_weight_ci_lower) * 100 if initial_weight_ci_lower > 0 else 0

        # Ensure the preoperative loss is exactly 0
        summary_df.loc[0, ['Perc Weight Loss', 'Perc Weight Loss CI Lower', 'Perc Weight Loss CI Upper']] = 0
        # --- END NEW SECTION ---


    # Add Diabetes data
    dm_status_cols = ['DMII_preoperative', 'dm3m', 'dm6m', 'dm12m', 'dm18m', 'dm2y', 'dm3y', 'dm4y', 'dm5y']
    dm_prob_cols = ['DMII_preoperative_prob', 'dm3m_prob', 'dm6m_prob', 'dm12m_prob', 'dm18m_prob', 'dm2y_prob', 'dm3y_prob', 'dm4y_prob', 'dm5y_prob']
    summary_df['DM Status'] = np.where(df_final[dm_status_cols].iloc[0].values == 1, 'Diabetes', 'Remission')
    summary_df['DM Likelihood (%)'] = (df_final[dm_prob_cols].iloc[0].values * 100).round(1)

    return summary_df

###############################################################################
# Page configuration
st.set_page_config(
    page_title="DM Predictor",
    layout='wide'
)

# Load Models
try:
    reg_model, clf_model = initialize_app()
except Exception as e:
    st.error(f"Failed to load models. Error: {e}")
    st.stop()


# --- HEADER ---
st.title("Diabetes Mellitus (DM) Predictor")
st.markdown("This app let you select the patients information and the model is going to predict the BMI evolution of the patient and it's DM probability remission.")
st.markdown("---")

# --- LAYOUT ---
input_col, output_col = st.columns((1, 1.5))

# --- INPUT COLUMN ---
with input_col:
    st.header("Patient Information")

    st.subheader("Patient Data")
    age = st.slider("Age (Years):", min_value=18, max_value=80, value=45, step=1)
    sex = st.radio("Sex:", options=['Female', 'Male'], horizontal=True)

    input_method = st.radio("Input Method:", ("Enter BMI & Height", "Enter Weight & Height"))

    bmi_pre = 0.0
    weight_kg = 0.0
    height_cm = st.number_input("Height (cm):", min_value=140.0, max_value=220.0, value=170.0, step=0.5)

    if input_method == "Enter BMI & Height":
        bmi_pre = st.slider("Preoperative BMI:", min_value=30.0, max_value=70.0, value=42.0, step=0.1)
    else: # "Enter Weight & Height"
        weight_kg = st.number_input("Weight (kg):", min_value=50.0, max_value=300.0, value=120.0, step=0.5)


    st.subheader("Medical Conditions")
    hypertension = st.checkbox("Hypertension")
    hyperlipidemia = st.checkbox('Hyperlipidemia')
    osas_preoperative = st.checkbox('Obstructive Sleep Apnea (OSAS)')
    DMII_preoperative = st.checkbox('Type 2 Diabetes (T2DM)')

    antidiab_drug_preop_no_therapy = 0
    antidiab_drug_preop_glp1_analogen = 0
    antidiab_drug_preop_Oral_anticogulation = 0
    antidiab_drug_preop_Insulin = 0

    if DMII_preoperative:
        st.write("Preoperative T2DM Treatment:")
        antidiab_drug_preop_no_therapy = st.checkbox('No Therapy')
        antidiab_drug_preop_glp1_analogen = st.checkbox('GLP-1 Analog')
        antidiab_drug_preop_Oral_anticogulation = st.checkbox('Oral Antidiabetic Drug')
        antidiab_drug_preop_Insulin = st.checkbox('Insulin')

    st.subheader("Planned Surgery")
    surgery = st.radio(
        "Select a procedure:",
        options=list(dictionary_categorical_features['surgery'].keys())
    )

    predict_button = st.button("Compute Prediction", type="primary", use_container_width=True)


# --- OUTPUT COLUMN ---
with output_col:
    st.header("Prediction Results")

    if predict_button:
        height_m = height_cm / 100.0
        if height_m <= 0:
            st.error("Height must be greater than 0.")
            st.stop() # Stop execution if height is invalid

        if input_method == "Enter Weight & Height":
            if weight_kg > 0:
                bmi_pre = weight_kg / (height_m ** 2)
                st.info(f"Calculated Preoperative BMI: {bmi_pre:.1f}")
            else:
                st.error("Weight must be greater than 0.")
                st.stop() # Stop execution if weight is invalid

        with st.spinner("Calculating patient's trajectory..."):
            # Prepare input dataframe
            input_data = {
                'age_years': [age],
                'BMI before surgery': [bmi_pre],
                'sex (1 = female, 2=male)': [sex],
                'hypertension': [inverse_dictionary['hypertension'][int(hypertension)]],
                'hyperlipidemia': [inverse_dictionary['hyperlipidemia'][int(hyperlipidemia)]],
                'DMII_preoperative': [inverse_dictionary['DMII_preoperative'][int(DMII_preoperative)]],
                'surgery': [surgery],
                'antidiab_drug_preop_Oral_anticogulation': [inverse_dictionary['antidiab_drug_preop_Oral_anticogulation'][int(antidiab_drug_preop_Oral_anticogulation)]],
                'antidiab_drug_preop_Insulin': [inverse_dictionary['antidiab_drug_preop_Insulin'][int(antidiab_drug_preop_Insulin)]],
                'antidiab_drug_preop_no_therapy': [inverse_dictionary['antidiab_drug_preop_no_therapy'][int(antidiab_drug_preop_no_therapy)]],
                'antidiab_drug_preop_glp1_analogen': [inverse_dictionary['antidiab_drug_preop_glp1_analogen'][int(antidiab_drug_preop_glp1_analogen)]],
                'osas_preoperative': [inverse_dictionary['osas_preoperative'][int(osas_preoperative)]]
            }
            dataframe_input = pd.DataFrame(input_data)

            # Run prediction
            summary_df = parser_user_input(dataframe_input, reg_model, clf_model, height_m)

            st.subheader("Predicted Outcomes at Key Timepoints")

            # Display BMI Metrics
            st.markdown("##### BMI")
            cols_bmi = st.columns(3)
            bmi_1y = summary_df.loc[summary_df['Time'] == '12m', 'BMI'].values[0]
            bmi_3y = summary_df.loc[summary_df['Time'] == '3y', 'BMI'].values[0]
            bmi_5y = summary_df.loc[summary_df['Time'] == '5y', 'BMI'].values[0]
            cols_bmi[0].metric(label="BMI at 1 Year", value=f"{bmi_1y:.1f}")
            cols_bmi[1].metric(label="BMI at 3 Years", value=f"{bmi_3y:.1f}")
            cols_bmi[2].metric(label="BMI at 5 Years", value=f"{bmi_5y:.1f}")

            # Display Weight and % Loss Metrics
            if 'Weight' in summary_df.columns:
                st.markdown("##### Weight (kg)")
                cols_weight = st.columns(3)
                weight_1y = summary_df.loc[summary_df['Time'] == '12m', 'Weight'].values[0]
                weight_3y = summary_df.loc[summary_df['Time'] == '3y', 'Weight'].values[0]
                weight_5y = summary_df.loc[summary_df['Time'] == '5y', 'Weight'].values[0]
                cols_weight[0].metric(label="Weight (kg) at 1 Year", value=f"{weight_1y:.1f}")
                cols_weight[1].metric(label="Weight (kg) at 3 Years", value=f"{weight_3y:.1f}")
                cols_weight[2].metric(label="Weight (kg) at 5 Years", value=f"{weight_5y:.1f}")

                st.markdown("##### % Weight Loss from Baseline")
                cols_loss = st.columns(3)
                loss_1y = summary_df.loc[summary_df['Time'] == '12m', 'Perc Weight Loss'].values[0]
                loss_3y = summary_df.loc[summary_df['Time'] == '3y', 'Perc Weight Loss'].values[0]
                loss_5y = summary_df.loc[summary_df['Time'] == '5y', 'Perc Weight Loss'].values[0]
                cols_loss[0].metric(label="% Loss at 1 Year", value=f"{loss_1y:.1f}%")
                cols_loss[1].metric(label="% Loss at 3 Years", value=f"{loss_3y:.1f}%")
                cols_loss[2].metric(label="% Loss at 5 Years", value=f"{loss_5y:.1f}%")


            # Create and display animated chart for BMI
            create_animated_chart(summary_df, 'BMI', 'Body Mass Index (BMI)', 'BMI CI Lower', 'BMI CI Upper', target_line_val=25)

            # Create and display animated chart for Weight and % Weight Loss
            if 'Weight' in summary_df.columns:
                healthy_weight_target = 25 * (height_m ** 2)
                create_animated_chart(summary_df, 'Weight', 'Weight (kg)', 'Weight CI Lower', 'Weight CI Upper', target_line_val=round(healthy_weight_target, 1))

                create_animated_chart(summary_df, 'Perc Weight Loss', '% Total Weight Loss', 'Perc Weight Loss CI Lower', 'Perc Weight Loss CI Upper', target_line_val=20, target_line_color='purple')

            if DMII_preoperative:
                st.subheader("Diabetes Remission Likelihood")
                dm_prob_df = summary_df[['Time', 'DM Likelihood (%)']].set_index('Time')

                # Apply scientific styling with a color gradient
                styled_dm_table = dm_prob_df.style.format({
                    "DM Likelihood (%)": "{:.1f}%"
                }).background_gradient(
                    cmap='RdYlGn_r',  # Red-Yellow-Green (Reversed)
                    subset=['DM Likelihood (%)'],
                    vmin=0,
                    vmax=100
                ).set_properties(**{
                    'text-align': 'center',
                    'font-size': '14px',
                    'width': '100px'
                }).set_table_styles([
                    {'selector': 'th', 'props': [('text-align', 'center'), ('font-size', '16px')]},
                ])

                st.write(styled_dm_table)

    else:
        st.info("Please enter patient data and click 'Compute Prediction' to see the results.")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 16px; color: grey;">
    <b>Disclaimer:</b> This application and its results are intended for research purposes only.
    It is not a medical device and should not be used for clinical decision-making.
</div>
""", unsafe_allow_html=True)
# Sponsor Images
images = [r'images/basel.png',
          r'images/Logo_Unibas_BraPan_EN.png',
          r'images/kannospital.png',
          r'images/basel_2.png',
          r'images/claraspital.png',
          r'images/wuzburg.png',
          r'images/linkoping_university.png',
          r'images/umm.png',
          r'images/tiroler.png',
          r'images/marmara_university.png',
          r'images/gzo_hospital.png',
          r'images/thurgau_spital.jpg',
          r'images/warsaw_medical_university.png',
          r'images/nova_medical_school.png',
          r'images/ECU.png'
          ]

st.markdown("---")
st.markdown("<p style='text-align: center;'><strong>Collaborations:</strong></p>", unsafe_allow_html=True)
partner_logos = [
{
    "title": "",
    "text": "",
    "img": images[0]
},
{
    "title": "",
    "text" : "",
    "img": images[1]
},
{
    "title": "",
    "text" : "",
    "img": images[2]
},
{
    "title": "",
    "text" : "",
    "img": images[3]
},
{
    "title": "",
    "text" : "",
    "img": images[4]
},
{
    "title": "",
    "text" : "",
    "img": images[5]
},
{
    "title": "",
    "text" : "",
    "img": images[6]
},
{
    "title": "",
    "text" : "",
    "img": images[7]
},
{
    "title": "",
    "text" : "",
    "img": images[8]
},
{
    "title": "",
    "text" : "",
    "img": images[9]
},
{
    "title": "",
    "text" : "",
    "img": images[10]
},
{
    "title": "",
    "text" : "",
    "img": images[11]
},
{
    "title": "",
    "text" : "",
    "img": images[12]
},
{
    "title": "",
    "text" : "",
    "img": images[13]
},
{
    "title": "",
    "text" : "",
    "img": images[14]
}]
# Check if images exist before creating the carousel
valid_logos = [item for item in partner_logos if os.path.exists(item['img'])]
if valid_logos:
    carousel(items=valid_logos, width=0.25)
else:
    st.warning("Could not find sponsor images. Please ensure the 'images' directory is present.")
