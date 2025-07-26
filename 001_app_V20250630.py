# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:31:05 2025

@author: Vincent Ochs

This script is used for generating an app for regression and classification
task.

Refactored on Fri Jul 25 2025 to match the style of:
https://bariatric-weight-trajectory-prediction.univ-lille.fr/
"""

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
    model_path = os.path.join('001_Regression_Model_App.sav')
    with open(model_path , 'rb') as export_model:
        regression_model = pickle.load(export_model) 
    # Load Classification Model
    model_path = os.path.join('001_Classification_Model_App.sav')
    with open(model_path, 'rb') as export_model:
        classification_model = pickle.load(export_model) 

    print('App Initialized correctly!')
    
    return regression_model , classification_model

###############################################################################
def create_prediction_chart(summary_df, y_axis_variable, y_axis_title, y_axis_format=".1f"):
    """
    Creates a static Altair chart for BMI or Weight evolution.

    Args:
        summary_df (pd.DataFrame): DataFrame containing the data to plot.
        y_axis_variable (str): The column name for the Y-axis (e.g., 'BMI', 'Weight').
        y_axis_title (str): The title for the Y-axis.
        y_axis_format (str): The format for the tooltip values.

    Returns:
        alt.Chart: The Altair chart object.
    """
    # Round to 2 decimal
    summary_df[y_axis_variable] = summary_df[y_axis_variable].round(2)
    summary_df[f'{y_axis_variable} CI Lower'] = summary_df[f'{y_axis_variable} CI Lower'].round(2)
    summary_df[f'{y_axis_variable} CI Upper'] = summary_df[f'{y_axis_variable} CI Upper'].round(2)
    
    time_map = {'Pre': 0, '3m': 3, '6m': 6, '12m': 12, '18m': 18, '2y': 24, '3y': 36, '4y': 48, '5y': 60}
    summary_df['Months'] = summary_df['Time'].map(time_map)
    
    # Define the y-axis domain
    y_min = summary_df[f'{y_axis_variable} CI Lower'].min() * 0.9
    y_max = summary_df[f'{y_axis_variable} CI Upper'].max() * 1.1

    # Main prediction line
    line = alt.Chart(summary_df).mark_line(
        color='#0068c9',
        size=3
    ).encode(
        x=alt.X('Months', title='Postoperative Months'),
        y=alt.Y(y_axis_variable, title=y_axis_title, scale=alt.Scale(domain=[y_min, y_max])),
        tooltip=[
            alt.Tooltip('Months', title='Postop. Months'),
            alt.Tooltip(y_axis_variable, title=y_axis_title, format=y_axis_format)
        ]
    )

    # Confidence interval area
    area = alt.Chart(summary_df).mark_area(
        opacity=0.3,
        color='#0068c9'
    ).encode(
        x='Months',
        y=alt.Y(f'{y_axis_variable} CI Lower', title=''),
        y2=alt.Y2(f'{y_axis_variable} CI Upper', title='')
    )
    
    # Healthy BMI line (only for BMI chart)
    if y_axis_variable == 'BMI':
        healthy_bmi_line = alt.Chart(pd.DataFrame({'y': [25]})).mark_rule(color='green', strokeDash=[3,3], size=2).encode(y='y')
        chart = (area + line + healthy_bmi_line)
        st.markdown("""
        <div style="font-size:14px; text-align: center; margin-bottom: 10px;">
            <span style="color:#0068c9;">■</span> Predicted BMI & 95% CI
            &nbsp;&nbsp;&nbsp;
            <span style="color:green;">- - -</span> Healthy BMI Target (25)
        </div>
        """, unsafe_allow_html=True)
    else:
        chart = area + line

    return chart.properties(
        width=600,
        height=400
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).interactive()

# Parser input information
def parser_user_input(dataframe_input, reg_model, clf_model, height_cm):
    """
    Parses user input, runs predictions, and returns a comprehensive dataframe.
    """
    ##########################################################################
    # Regression part
    
    # Encode categorical features
    for i in dictionary_categorical_features.keys():
        if i in dataframe_input.columns:
            dataframe_input[i] = dataframe_input[i].map(dictionary_categorical_features[i])
            
    predictions_df_regression = pd.DataFrame(reg_model.predict(dataframe_input[reg_model.feature_names_in_.tolist()]))
    predictions_df_regression.columns = ['bmi3','bmi6','bmi12','bmi18','bmi2y','bmi3y','bmi4y','bmi5y']
    
    # Adjust BMI curve based on surgery effectiveness
    surgery_name = dataframe_input['surgery'].values[0]
    if surgery_name == 1: # LSG
        predictions_df_regression *= 0.85
    # For LRYGB (2), we use the default prediction (1.0)
    
    ###########################################################################
    # Classification part
    df_classification = pd.concat([dataframe_input, predictions_df_regression], axis=1)
    
    # Note: The logic for DM prediction remains complex and internal to this function
    # It will be simplified for this example, but the original logic is preserved below
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
            
            Parameters:
            - bmi_change: Change in BMI (negative = reduction, positive = increase)
            - current_prob: Current DM probability
            - current_bmi: Current BMI value
            - surgery_type: Type of surgery (1=LSG, 2=LRYGB, 5=OAGB)
            - time_step: Time step identifier
            - medications: Dictionary with medication flags
            - base_adjustment: Base adjustment factor per BMI unit change
            """
            
            # Surgery-specific BMI thresholds based on your table
            # Using the "Lite threshold BMI" values from your image
            surgery_thresholds = {
                # LSG (Laparoscopic Sleeve Gastrectomy) - surgery type 1
                1: {
                    '3m': 35.0,   # 3 months
                    '6m': 32.0,   # 6 months  
                    '12m': 30.0,  # 12 months
                    '18m': 28.0,  # 18 months
                    '2y': 27.0,   # 2 years
                    '3y': 26.0,   # 3 years
                    '4y': 25.0,   # 4 years
                    '5y': 25.0    # 5 years
                },
                # LRYGB (Laparoscopic Roux-en-Y Gastric Bypass) - surgery type 2
                2: {
                    '3m': 33.0,   # Typically more aggressive weight loss
                    '6m': 30.0,   
                    '12m': 28.0,  
                    '18m': 26.0,  
                    '2y': 25.0,   
                    '3y': 24.0,   
                    '4y': 23.0,   
                    '5y': 23.0    
                },
                # OAGB (One Anastomosis Gastric Bypass) - surgery type 5
                5: {
                    '3m': 34.0,   # Between LSG and LRYGB
                    '6m': 31.0,   
                    '12m': 29.0,  
                    '18m': 27.0,  
                    '2y': 26.0,   
                    '3y': 25.0,   
                    '4y': 24.0,   
                    '5y': 24.0    
                }
            }
            
            # Time step mapping
            time_step_map = {
                'dm3m_prob': '3m',
                'dm6m_prob': '6m', 
                'dm12m_prob': '12m',
                'dm18m_prob': '18m',
                'dm2y_prob': '2y',
                'dm3y_prob': '3y',
                'dm4y_prob': '4y',
                'dm5y_prob': '5y'
            }
            
            # Get the appropriate threshold for this surgery and time step
            surgery_key = surgery_type if surgery_type in surgery_thresholds else 1  # Default to LSG
            time_key = time_step_map.get(time_step, '3m')  # Default to 3m if not found
            threshold_bmi = surgery_thresholds[surgery_key].get(time_key, 30.0)  # Default threshold
            
            # Calculate BMI-based adjustment
            if current_bmi <= threshold_bmi:
                # BMI is below threshold - higher chance of remission (lower DM probability)
                bmi_difference = threshold_bmi - current_bmi
                # Surgery-specific effectiveness multiplier
                if surgery_type == 2:  # LRYGB - most effective
                    effectiveness_multiplier = 1.15
                elif surgery_type == 5:  # OAGB - moderately effective  
                    effectiveness_multiplier = 1.15
                else:  # LSG - baseline effectiveness
                    effectiveness_multiplier = 2.5
                    
                # The more below the threshold, the greater the reduction
                reduction_factor = min(base_adjustment * (1 + bmi_difference / 10) * effectiveness_multiplier, 0.8)
                bmi_adjusted_prob = current_prob * (1 - reduction_factor)
            else:
                # BMI is above threshold - lower chance of remission (higher DM probability)
                bmi_difference = current_bmi - threshold_bmi
                # The more above the threshold, the smaller the reduction (or even increase)
                increase_factor = min(base_adjustment * (bmi_difference / 10), 0.3)
                bmi_adjusted_prob = current_prob * (1 + increase_factor)
            
            # Apply medication effects (medications reduce DM probability)
            medication_reduction = 0.0
            
            # GLP-1 analogs - strong effect on diabetes remission
            if medications.get('antidiab_drug_preop_glp1_analogen', 0) == 1:
                medication_reduction += 0.15  # 15% reduction
            
            # Oral antidiabetic drugs - moderate effect
            if medications.get('antidiab_drug_preop_Oral_anticogulation', 0) == 1:
                medication_reduction += 0.10  # 10% reduction
                
            # Insulin - indicates more severe diabetes, but still some protective effect
            if medications.get('antidiab_drug_preop_Insulin', 0) == 1:
                medication_reduction += 0.08  # 8% reduction
            
            # No therapy - no additional benefit
            if medications.get('antidiab_drug_preop_no_therapy', 0) == 1:
                medication_reduction += 0.0  # No additional reduction
            
            # Apply medication reduction
            final_adjusted_prob = bmi_adjusted_prob * (1 - medication_reduction)
            
            # Ensure probability stays within [0, 1] bounds
            final_adjusted_prob = max(0.0, min(1.0, final_adjusted_prob))
            
            return final_adjusted_prob
        
        # Apply BMI change-based adjustments to each time step
        def apply_enhanced_bmi_adjustments(df_final, clf_model):
            """
            Apply enhanced BMI adjustments with surgery-specific thresholds and medication effects
            """
            # Define BMI time step transitions
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
            
            # Only apply BMI-based adjustments if patient has preoperative diabetes
            if df_final['DMII_preoperative'].iloc[0] == 1:
                
                # Get surgery type and medications
                surgery_type = df_final['surgery'].iloc[0]
                medications = {
                    'antidiab_drug_preop_glp1_analogen': df_final['antidiab_drug_preop_glp1_analogen'].iloc[0],
                    'antidiab_drug_preop_Oral_anticogulation': df_final['antidiab_drug_preop_Oral_anticogulation'].iloc[0],
                    'antidiab_drug_preop_Insulin': df_final['antidiab_drug_preop_Insulin'].iloc[0],
                    'antidiab_drug_preop_no_therapy': df_final['antidiab_drug_preop_no_therapy'].iloc[0]
                }
                
                # Apply enhanced BMI change-based adjustments to each time step
                for transition in bmi_transitions:
                    prev_bmi_col = transition['prev_bmi']
                    current_bmi_col = transition['current_bmi']
                    prob_col = transition['prob_col']
                    
                    prev_bmi = df_final[prev_bmi_col].iloc[0]
                    current_bmi = df_final[current_bmi_col].iloc[0]
                    current_prob = df_final[prob_col].iloc[0]
                    
                    # Calculate BMI change
                    bmi_change = current_bmi - prev_bmi
                    
                    # Calculate adjusted probability using enhanced function
                    adjusted_prob = adjust_probability_by_bmi_change_enhanced(
                        bmi_change=bmi_change,
                        current_prob=current_prob, 
                        current_bmi=current_bmi,
                        surgery_type=surgery_type,
                        time_step=prob_col,
                        medications=medications
                    )
                    
                    # Apply comorbidity adjustments (hypertension and hyperlipidemia)
                    comorbidity_factor = 1.0
                    
                    # Hypertension increases DM probability by 10-15%
                    if df_final['hypertension'].iloc[0] == 1:
                        comorbidity_factor *= 1.35
                    
                    # Hyperlipidemia increases DM probability by 8-12%
                    if df_final['hyperlipidemia'].iloc[0] == 1:
                        comorbidity_factor *= 1.20
                    # OSAS increase DM probability by 10%
                    if df_final['osas_preoperative'].iloc[0] == 1:
                        comorbidity_factor *= 1.10
                    
                    # Apply comorbidity factor
                    final_adjusted_prob = min(1.0, adjusted_prob * comorbidity_factor)
                    
                    # Update the probability in the dataframe
                    df_final[prob_col] = final_adjusted_prob
                    
                    # Also update the binary prediction based on adjusted probability
                    binary_col = prob_col.replace('_prob', '')
                    df_final[binary_col] = 1 if final_adjusted_prob > 0.5 else 0
                    
                    # Get surgery name for logging
                    surgery_names = {1: 'LSG', 2: 'LRYGB', 5: 'OAGB'}
                    surgery_name = surgery_names.get(surgery_type, 'Unknown')
                    
                    print(f"Transition {prev_bmi_col} -> {current_bmi_col}: "
                          f"Surgery={surgery_name}, BMI={current_bmi:.1f}, BMI change={bmi_change:.1f}, "
                          f"Original prob={current_prob:.3f}, "
                          f"Adjusted prob={adjusted_prob:.3f}, "
                          f"Final prob (with comorbidities)={final_adjusted_prob:.3f}")
            
            return df_final
        df_final = apply_enhanced_bmi_adjustments(df_final, clf_model)
    
    ###########################################################################
    # Confidence interval and summary creation
    bmi_columns = ['BMI before surgery', 'bmi3', 'bmi6', 'bmi12', 'bmi18', 'bmi2y', 'bmi3y', 'bmi4y', 'bmi5y']

    for col in bmi_columns:
        df_final[f'{col}_ci_lower'] = df_final[col] - training_mae
        df_final[f'{col}_ci_upper'] = df_final[col] + training_mae
        
    # Calculate Weight and Weight CI
    height_m_sq = (height_cm / 100) ** 2
    for col in bmi_columns:
        weight_col_name = col.replace('bmi', 'weight').replace('BMI before surgery', 'weight_preop')
        df_final[weight_col_name] = df_final[col] * height_m_sq
        df_final[f'{weight_col_name}_ci_lower'] = df_final[f'{col}_ci_lower'] * height_m_sq
        df_final[f'{weight_col_name}_ci_upper'] = df_final[f'{col}_ci_upper'] * height_m_sq

    # Create a clean summary dataframe for plotting and metrics
    summary_df = pd.DataFrame({
        'Time': ['Pre', '3m', '6m', '12m', '18m', '2y', '3y', '4y', '5y'],
    })

    # Add BMI and Weight data to summary
    summary_df['BMI'] = df_final[bmi_columns].iloc[0].values
    summary_df['BMI CI Lower'] = df_final[[f'{c}_ci_lower' for c in bmi_columns]].iloc[0].values
    summary_df['BMI CI Upper'] = df_final[[f'{c}_ci_upper' for c in bmi_columns]].iloc[0].values
    
    weight_cols = [c.replace('bmi', 'weight').replace('BMI before surgery', 'weight_preop') for c in bmi_columns]
    summary_df['Weight'] = df_final[weight_cols].iloc[0].values
    summary_df['Weight CI Lower'] = df_final[[f'{c}_ci_lower' for c in weight_cols]].iloc[0].values
    summary_df['Weight CI Upper'] = df_final[[f'{c}_ci_upper' for c in weight_cols]].iloc[0].values
    
    # Add Diabetes data
    dm_status_cols = ['DMII_preoperative', 'dm3m', 'dm6m', 'dm12m', 'dm18m', 'dm2y', 'dm3y', 'dm4y', 'dm5y']
    dm_prob_cols = ['DMII_preoperative_prob', 'dm3m_prob', 'dm6m_prob', 'dm12m_prob', 'dm18m_prob', 'dm2y_prob', 'dm3y_prob', 'dm4y_prob', 'dm5y_prob']
    summary_df['DM Status'] = np.where(df_final[dm_status_cols].iloc[0].values == 1, 'Diabetes', 'Remission')
    summary_df['DM Likelihood (%)'] = (df_final[dm_prob_cols].iloc[0].values * 100).round(1)

    return summary_df

###############################################################################
# Page configuration
st.set_page_config(
    page_title="Bariatric Surgery Outcome Predictor",
    layout='wide'
)

# Load Models
reg_model, clf_model = initialize_app()

# --- HEADER ---
st.title("Bariatric Surgery Weight Trajectory Predictor")
st.markdown("This tool predicts the Body Mass Index (BMI) and weight evolution for up to 5 years after bariatric surgery.")
st.markdown("---")

# --- LAYOUT ---
input_col, output_col = st.columns((1, 1.5))

# --- INPUT COLUMN ---
with input_col:
    st.header("Patient Information")

    st.subheader("Patient Data")
    age = st.slider("Age (Years):", min_value=18, max_value=80, value=45, step=1)
    height = st.slider("Height (cm):", min_value=140, max_value=220, value=170, step=1)
    bmi_pre = st.slider("Preoperative BMI:", min_value=30.0, max_value=70.0, value=42.0, step=0.1)
    sex = st.radio("Sex:", options=['Female', 'Male'], horizontal=True)

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
            summary_df = parser_user_input(dataframe_input, reg_model, clf_model, height)
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Key Metrics", "BMI Evolution", "Weight Evolution"])

            with tab1:
                st.subheader("Predicted Outcomes at Key Timepoints")
                
                # Metrics at 1, 3, 5 years
                weight_preop = summary_df.loc[summary_df['Time'] == 'Pre', 'Weight'].values[0]
                ideal_weight = 25 * ((height / 100) ** 2)
                excess_weight = weight_preop - ideal_weight

                for year, time_code in zip([1, 3, 5], ['12m', '3y', '5y']):
                    st.markdown(f"**After {year} Year{'s' if year > 1 else ''}:**")
                    cols = st.columns(3)
                    
                    bmi_pred = summary_df.loc[summary_df['Time'] == time_code, 'BMI'].values[0]
                    weight_pred = summary_df.loc[summary_df['Time'] == time_code, 'Weight'].values[0]
                    weight_loss = weight_preop - weight_pred
                    ewl = (weight_loss / excess_weight * 100) if excess_weight > 0 else 0

                    cols[0].metric(label="Predicted BMI", value=f"{bmi_pred:.1f}")
                    cols[1].metric(label="Total Weight Loss", value=f"{weight_loss:.1f} kg")
                    cols[2].metric(label="% Excess Weight Loss", value=f"{ewl:.1f} %")
                
                if DMII_preoperative:
                    st.subheader("Diabetes Remission Likelihood")
                    dm_prob_df = summary_df[['Time', 'DM Likelihood (%)']].set_index('Time')
                    st.dataframe(dm_prob_df, use_container_width=True)


            with tab2:
                st.subheader("Predicted BMI Evolution")
                bmi_chart = create_prediction_chart(summary_df, 'BMI', 'Body Mass Index (BMI)')
                st.altair_chart(bmi_chart, use_container_width=True)

            with tab3:
                st.subheader("Predicted Weight Evolution")
                weight_chart = create_prediction_chart(summary_df, 'Weight', 'Weight (kg)')
                st.altair_chart(weight_chart, use_container_width=True)
    
    else:
        st.info("Please enter patient data and click 'Compute Prediction' to see the results.")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 12px; color: grey;">
    <b>Disclaimer:</b> This application and its results are intended for research purposes only. 
    It is not a medical device and should not be used for clinical decision-making.
</div>
""", unsafe_allow_html=True)
# Sponsor Images
images = [r'images/basel.png',
          r'images/basel_university.jpeg',
          r'images/claraspital.png',
          r'images/wuzburg.png',
          r'images/linkoping_university.png',
          r'images/umm.png',
          r'images/tiroler.png',
          r'images/marmara_university.png',
          r'images/gzo_hospital.png',
          r'images/thurgau_spital.jpg',
          r'images/warsaw_medical_university.png',            
          r'images/nova_medical_school.png'         
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
}]
carousel(items=partner_logos, width=0.25)
