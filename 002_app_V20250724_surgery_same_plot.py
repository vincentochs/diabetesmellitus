# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:31:05 2025

@author: Vincent Ochs 

This script is used for generating an app for regression and classification
task
"""

###############################################################################
# Load libraries

# App
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_carousel import carousel
import altair as alt

# Utils
import pandas as pd
import pickle as pkl
import numpy as np
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import time
import altair as alt

# UTils
import pickle
from scipy.interpolate import make_interp_spline
import os

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
                                                #'OAGB' : 5},
                                   
                                   'normal_dmII_pattern' : {'Yes' : 1,
                                                            'No' : 0},
                                   'months' :  {'Preoperative' : 0,
                                                   '3 Months' : 1,
                                                   '6 Months' : 2,
                                                   '12 Months' : 3,
                                                   '18 Months' : 4,
                                                   '2 Years' : 5,
                                                   '3 Years' : 6,
                                                   '4 Years' : 7,
                                                   '5 Years' : 8},
                                   'time_step' : {'BMI before surgery' : 0,
                                                  'bmi3' : 1,
                                                  'bmi6' : 2,
                                                  'bmi12' : 3,
                                                  'bmi18' : 4,
                                                  'bmi2y' : 5,
                                                  'bmi3y' : 6,
                                                  'bmi4y' : 7,
                                                  'bmi5y' : 8},
                                   'asa_score' : {'1' : 1,
                                                  '2' : 2,
                                                  '3' : 3,
                                                  '4' : 4},
                                   'antidiab_drug_preop_no_therapy' : {'Yes' : 1,
                                                                       'No' : 0},
                                   'antidiab_drug_preop_glp1_analogen' : {'Yes' : 1,
                                                                          'No' : 0},
                                   'comorbid_1_Myocardial_infarct' : {'Yes' : 1,
                                                                      'No' : 0},
                                   'comorbid_2_heart_failure' : {'Yes' : 1,
                                                                 'No' : 0},
                                   'comorbid_6_pulmonary_disease' : {'Yes' : 1,
                                                                     'No' : 0}
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
def create_animated_evolution_chart(df_final_dict, clf_model, predictions_df, threshold_values=None):
    # If dict is passed (comparison mode)
    if isinstance(df_final_dict, dict):
        st.header("BMI Progression Comparison: LSG vs LRYGB")
        st.write("Comparing predicted BMI outcomes for both surgery types up to 5 years post-surgery.")
        
        # Get data for both surgeries
        lsg_data = df_final_dict['Laparoscopic Sleeve Gastrectomy (LSG)']
        lrygb_data = df_final_dict['Laparoscopic Roux-en-Y Gastric Bypass (LRYGB)']
        
        # Same time mapping and column setup as original
        time_map = {'Pre': 0, '3m': 3, '6m': 6, '12m': 12, '18m': 18, '2y': 24, '3y': 36, '4y': 48, '5y': 60}
        time_points_values = list(time_map.values())
        bmi_columns = ['BMI before surgery', 'bmi3', 'bmi6', 'bmi12', 'bmi18', 'bmi2y', 'bmi3y', 'bmi4y', 'bmi5y']
        
        # Get BMI values for both surgeries
        lsg_bmi_values = lsg_data[bmi_columns].iloc[0].values
        lrygb_bmi_values = lrygb_data[bmi_columns].iloc[0].values
        
        # Get confidence intervals for both surgeries
        lsg_ci_lower = lsg_data[[f'{col}_ci_lower' for col in bmi_columns]].iloc[0].values
        lsg_ci_upper = lsg_data[[f'{col}_ci_upper' for col in bmi_columns]].iloc[0].values
        lrygb_ci_lower = lrygb_data[[f'{col}_ci_lower' for col in bmi_columns]].iloc[0].values
        lrygb_ci_upper = lrygb_data[[f'{col}_ci_upper' for col in bmi_columns]].iloc[0].values
        
        # Calculate y-axis domain
        all_values = np.concatenate([lsg_ci_lower, lsg_ci_upper, lrygb_ci_lower, lrygb_ci_upper])
        bmi_min = max(np.min(all_values) * 0.5, 10)
        bmi_max = np.max(all_values) * 1.8
        
        # Create smooth curves for both surgeries
        total_steps = 100
        smooth_years = np.linspace(0, 60, total_steps)
        
        # LSG curves
        smooth_lsg_bmi = make_interp_spline(time_points_values, lsg_bmi_values)(smooth_years)
        smooth_lsg_upper = make_interp_spline(time_points_values, lsg_ci_upper)(smooth_years)
        smooth_lsg_lower = make_interp_spline(time_points_values, lsg_ci_lower)(smooth_years)
        
        # LRYGB curves  
        smooth_lrygb_bmi = make_interp_spline(time_points_values, lrygb_bmi_values)(smooth_years)
        smooth_lrygb_upper = make_interp_spline(time_points_values, lrygb_ci_upper)(smooth_years)
        smooth_lrygb_lower = make_interp_spline(time_points_values, lrygb_ci_lower)(smooth_years)
        

        # Get DM probabilities for both surgeries
        dm_prob_columns = ['DMII_preoperative_prob', 'dm3m_prob', 'dm6m_prob', 'dm12m_prob', 
                          'dm18m_prob', 'dm2y_prob', 'dm3y_prob', 'dm4y_prob', 'dm5y_prob']
        
        lsg_dm_probabilities = lsg_data[dm_prob_columns].iloc[0].values
        lrygb_dm_probabilities = lrygb_data[dm_prob_columns].iloc[0].values
        
        # Create smooth DM probability curves
        smooth_lsg_dm_probabilities = make_interp_spline(time_points_values, lsg_dm_probabilities)(smooth_years)
        smooth_lsg_dm_probabilities = np.array([min(1, max(0, x)) for x in smooth_lsg_dm_probabilities])
        
        smooth_lrygb_dm_probabilities = make_interp_spline(time_points_values, lrygb_dm_probabilities)(smooth_years)
        smooth_lrygb_dm_probabilities = np.array([min(1, max(0, x)) for x in smooth_lrygb_dm_probabilities])
        
        # Healthy BMI line
        healthy_bmi_line = pd.DataFrame({
            "Year": smooth_years,
            "BMI": [25] * len(smooth_years),
            "Surgery": ["Healthy BMI"] * len(smooth_years)
        })
        
        chart_placeholder = st.empty()
        
        # Animation loop
        for i in range(1, total_steps + 1, 1):
            current_years = smooth_years[:i]
            
            # Create combined dataframe for both surgeries
            combined_bmi_data = pd.DataFrame({
                "Year": np.concatenate([current_years, current_years]),
                "BMI": np.concatenate([smooth_lsg_bmi[:i], smooth_lrygb_bmi[:i]]),
                "Surgery": ["LSG"] * len(current_years) + ["LRYGB"] * len(current_years)
            })
            
            # Combined CI data
            combined_ci_data = pd.DataFrame({
                "Year": np.concatenate([current_years, current_years, current_years, current_years]),
                "BMI": np.concatenate([smooth_lsg_upper[:i], smooth_lsg_lower[:i], 
                                     smooth_lrygb_upper[:i], smooth_lrygb_lower[:i]]),
                "Surgery": ["LSG CI"] * len(current_years) + ["LSG CI"] * len(current_years) + 
                          ["LRYGB CI"] * len(current_years) + ["LRYGB CI"] * len(current_years)
            })
            
            # Create Altair charts
            bmi_chart = alt.Chart(combined_bmi_data).mark_line(strokeWidth=3).encode(
                x=alt.X("Year", title="Months After Surgery"),
                y=alt.Y("BMI", title="Body Mass Index (BMI)", scale=alt.Scale(domain=[bmi_min, bmi_max])),
                color=alt.Color("Surgery", 
                    scale=alt.Scale(domain=["LSG", "LRYGB", "LSG CI", "LRYGB CI", "Healthy BMI"],
                                  range=["blue", "red", "lightblue", "pink", "green"])),
                tooltip=["Year:Q", "BMI:Q", "Surgery:N"]
            )
            
            ci_chart = alt.Chart(combined_ci_data).mark_line(strokeDash=[2, 2], strokeWidth=2).encode(
                x="Year",
                y=alt.Y("BMI", scale=alt.Scale(domain=[bmi_min, bmi_max])),
                color=alt.Color("Surgery", 
                    scale=alt.Scale(domain=["LSG", "LRYGB", "LSG CI", "LRYGB CI", "Healthy BMI"],
                                  range=["blue", "red", "lightblue", "pink", "green"]),
                    legend=None),
                tooltip=["Year:Q", "BMI:Q", "Surgery:N"]
            )
            
            healthy_chart = alt.Chart(healthy_bmi_line[:i]).mark_line(strokeWidth=1.5).encode(
                x="Year",
                y=alt.Y("BMI", scale=alt.Scale(domain=[bmi_min, bmi_max])),
                color=alt.Color("Surgery", 
                    scale=alt.Scale(domain=["LSG", "LRYGB", "LSG CI", "LRYGB CI", "Healthy BMI"],
                                  range=["blue", "red", "lightblue", "pink", "green"]),
                    legend=None)
            )
            
            final_chart = (ci_chart + healthy_chart + bmi_chart).properties(width=850, height=600)
            chart_placeholder.altair_chart(final_chart)
                       
            # Display current DM probability for both surgeries during animation
            if lsg_data['DMII_preoperative'].values[0] == 1:
                current_time_months = smooth_years[i-1]
                
                # Get current probabilities for both surgeries
                lsg_current_dm_probability = smooth_lsg_dm_probabilities[i-1]
                lrygb_current_dm_probability = smooth_lrygb_dm_probabilities[i-1]
                
                # Determine risk levels for both surgeries
                def get_risk_info(prob):
                    if prob > 0.8:
                        return "High Risk", "red"
                    elif prob > 0.5:
                        return "Middle Risk", "orange"
                    else:
                        return "Low Risk", "green"
                
                lsg_risk_status, lsg_prob_color = get_risk_info(lsg_current_dm_probability)
                lrygb_risk_status, lrygb_prob_color = get_risk_info(lrygb_current_dm_probability)
                
                # Convert months to readable format
                if current_time_months == 0:
                    time_label = "Pre-operative"
                elif current_time_months < 12:
                    time_label = f"{current_time_months:.0f} months"
                elif current_time_months == 12:
                    time_label = "1 year"
                else:
                    years = current_time_months / 12
                    time_label = f"{years:.1f} years"
                
            time.sleep(0.0001)    
        # Cards for the full information - COMPARISON MODE
        if lsg_data['DMII_preoperative'].values[0] == 1:
            st.subheader("Surgery Comparison: Diabetes Risk and BMI at Key Time Points")
            
            # Define time points and their corresponding indices
            time_points_display = [
                (0, "Pre-operative", "BMI before surgery", "DMII_preoperative_prob"),
                (3, "3 months", "bmi3", "dm3m_prob"),
                (6, "6 months", "bmi6", "dm6m_prob"),
                (12, "12 months", "bmi12", "dm12m_prob"),
                (18, "18 months", "bmi18", "dm18m_prob"),
                (24, "2 years", "bmi2y", "dm2y_prob"),
                (36, "3 years", "bmi3y", "dm3y_prob"),
                (48, "4 years", "bmi4y", "dm4y_prob"),
                (60, "5 years", "bmi5y", "dm5y_prob")
            ]
            
            # Create side-by-side comparison cards
            for idx, (months, time_label, bmi_col, prob_col) in enumerate(time_points_display):
                
                # Get data for both surgeries
                lsg_bmi = lsg_data[bmi_col].values[0]
                lsg_dm_probability = lsg_data[prob_col].values[0]
                lrygb_bmi = lrygb_data[bmi_col].values[0]
                lrygb_dm_probability = lrygb_data[prob_col].values[0]
                
                # Determine risk levels
                def get_risk_info(prob):
                    if prob > 0.8:
                        return "High Risk", "red"
                    elif prob > 0.5:
                        return "Middle Risk", "orange"
                    else:
                        return "Low Risk", "green"
                
                lsg_risk_status, lsg_prob_color = get_risk_info(lsg_dm_probability)
                lrygb_risk_status, lrygb_prob_color = get_risk_info(lrygb_dm_probability)
                
                # Calculate differences
                bmi_diff = lsg_bmi - lrygb_bmi
                dm_prob_diff = lsg_dm_probability - lrygb_dm_probability
                
                # Determine better option
                if abs(dm_prob_diff) > 0.01:
                    better_option = "LRYGB" if dm_prob_diff > 0 else "LSG"
                    diff_color = "green" if dm_prob_diff > 0 else "blue"
                else:
                    better_option = "Similar"
                    diff_color = "gray"
                
                # Create three columns: LSG, Comparison, LRYGB
                col1, col2, col3 = st.columns([2, 1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; background-color: #e6f3ff; border-radius: 10px; margin: 10px 0; border-left: 4px solid blue;">
                        <h4 style="margin: 0; color: #333;">LSG - {time_label}</h4>
                        <p style="font-size: 16px; margin: 5px 0; color: #666;">
                            <strong>BMI: {lsg_bmi:.1f}</strong>
                        </p>
                        <p style="font-size: 16px; margin: 5px 0; color: {lsg_prob_color}; font-weight: bold;">
                            DM Risk: {lsg_dm_probability:.1%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; background-color: #f0f2f6; border-radius: 10px; margin: 10px 0;">
                        <h5 style="margin: 0; color: #333;">vs</h5>
                        <p style="font-size: 12px; margin: 5px 0; color: #666;">
                            BMI Δ: {bmi_diff:+.1f}
                        </p>
                        <p style="font-size: 12px; margin: 5px 0; color: {diff_color}; font-weight: bold;">
                            DM Δ: {dm_prob_diff:+.1%}
                        </p>
                        <p style="font-size: 12px; margin: 5px 0; color: {diff_color};">
                            <strong>{better_option}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; background-color: #ffe6e6; border-radius: 10px; margin: 10px 0; border-left: 4px solid red;">
                        <h4 style="margin: 0; color: #333;">LRYGB - {time_label}</h4>
                        <p style="font-size: 16px; margin: 5px 0; color: #666;">
                            <strong>BMI: {lrygb_bmi:.1f}</strong>
                        </p>
                        <p style="font-size: 16px; margin: 5px 0; color: {lrygb_prob_color}; font-weight: bold;">
                            DM Risk: {lrygb_dm_probability:.1%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add a separator line between time points (except for the last one)
                if idx < len(time_points_display) - 1:
                    st.markdown("---")
            
            # Add overall summary
            st.subheader("Overall Surgery Comparison Summary")
            
            # Calculate overall statistics
            lsg_avg_dm_risk = np.mean([lsg_data[col].values[0] for _, _, _, col in time_points_display[1:]])  # Exclude preop
            lrygb_avg_dm_risk = np.mean([lrygb_data[col].values[0] for _, _, _, col in time_points_display[1:]])  # Exclude preop
            
            lsg_final_bmi = lsg_data['bmi5y'].values[0]
            lrygb_final_bmi = lrygb_data['bmi5y'].values[0]
            
            overall_better = "LRYGB" if lsg_avg_dm_risk > lrygb_avg_dm_risk else "LSG"
            overall_color = "red" if overall_better == "LRYGB" else "blue"
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: #e6f3ff; border-radius: 10px; margin: 10px 0; border-left: 4px solid blue;">
                    <h4 style="margin: 0; color: #333;">LSG Summary</h4>
                    <p style="font-size: 16px; margin: 10px 0; color: #666;">
                        <strong>Final BMI (5y): {lsg_final_bmi:.1f}</strong>
                    </p>
                    <p style="font-size: 16px; margin: 10px 0; color: #666;">
                        <strong>Avg DM Risk: {lsg_avg_dm_risk:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with summary_col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: #ffe6e6; border-radius: 10px; margin: 10px 0; border-left: 4px solid red;">
                    <h4 style="margin: 0; color: #333;">LRYGB Summary</h4>
                    <p style="font-size: 16px; margin: 10px 0; color: #666;">
                        <strong>Final BMI (5y): {lrygb_final_bmi:.1f}</strong>
                    </p>
                    <p style="font-size: 16px; margin: 10px 0; color: #666;">
                        <strong>Avg DM Risk: {lrygb_avg_dm_risk:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Overall recommendation
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px; margin: 20px 0; border: 2px solid {overall_color};">
                <h3 style="margin: 0; color: {overall_color};">Overall Recommendation</h3>
                <p style="font-size: 18px; margin: 15px 0; color: #333;">
                    Based on average diabetes risk reduction: <strong style="color: {overall_color};">{overall_better}</strong>
                </p>
                <p style="font-size: 14px; margin: 5px 0; color: #666;">
                    Average DM risk difference: <strong>{abs(lsg_avg_dm_risk - lrygb_avg_dm_risk):.1%}</strong>
                </p>
                <p style="font-size: 14px; margin: 5px 0; color: #666;">
                    Final BMI difference: <strong>{abs(lsg_final_bmi - lrygb_final_bmi):.1f} points</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add interpretation guide
            st.markdown("""
            **Interpretation Guide:**
            - **Blue cards**: LSG (Laparoscopic Sleeve Gastrectomy) results
            - **Red cards**: LRYGB (Laparoscopic Roux-en-Y Gastric Bypass) results  
            - **BMI Δ**: Difference in BMI (LSG - LRYGB). Positive = LSG higher BMI
            - **DM Δ**: Difference in Diabetes Risk (LSG - LRYGB). Positive = LSG higher risk
            - **Better option**: Surgery with lower diabetes risk at that time point
            """)
    else:
        # Define time points mapping
        df_final = df_final_dict()
        time_map = {
            'Pre': 0, '3m': 3, '6m': 6, '12m': 12,
            '18m': 18, '2y': 24, '3y': 36, '4y': 48, '5y': 60
        }
        time_points = list(time_map.keys())
        time_points_values = list(time_map.values())
        
        # Get BMI column names in correct order
        bmi_columns = ['BMI before surgery', 'bmi3', 'bmi6', 'bmi12', 
                      'bmi18', 'bmi2y', 'bmi3y', 'bmi4y', 'bmi5y']
        
        # Get DM probabilities columns in correct order
        dm_prob_columns = ['DMII_preoperative_prob', 'dm3m_prob', 'dm6m_prob', 'dm12m_prob', 
                          'dm18m_prob', 'dm2y_prob', 'dm3y_prob', 'dm4y_prob', 'dm5y_prob']
        
        # Create chart placeholder
        chart_placeholder = st.empty()
        prob_placeholder = st.empty()  # Placeholder for probability display
        
        # Get all BMI values and confidence intervals
        bmi_values = df_final[bmi_columns].iloc[0].values
        ci_lower_values = df_final[[f'{col}_ci_lower' for col in bmi_columns]].iloc[0].values
        ci_upper_values = df_final[[f'{col}_ci_upper' for col in bmi_columns]].iloc[0].values
        
        # Get DM probabilities
        dm_probabilities = df_final[dm_prob_columns].iloc[0].values
        
        # Calculate y-axis domain based on actual data ranges with reasonable BMI floor
        bmi_min = max(np.min(ci_lower_values) * 0.5, 10)  # Don't go below BMI 15
        bmi_max = np.max(ci_upper_values) * 1.8
        
        # Pre-calculate all smooth curves for animation
        total_steps = 100
        # Create a smoother curve using interpolation
        smooth_years = np.linspace(0, 60, total_steps)  # Ensure interpolation extends to year 5
        smooth_bmi_values = make_interp_spline(time_points_values, bmi_values)(smooth_years)
        smooth_bmi_upper = make_interp_spline(time_points_values, ci_upper_values)(smooth_years)
        smooth_bmi_lower = make_interp_spline(time_points_values, ci_lower_values)(smooth_years)
        
        smooth_dm_probabilities = make_interp_spline(time_points_values, dm_probabilities)(smooth_years)
        smooth_dm_probabilities = np.array(list(map(lambda x: 1 if x > 1 else x, smooth_dm_probabilities)))
        
        if threshold_values is None:
            smooth_threshold_bmi = [0] * len(smooth_years)
        else:
            smooth_threshold_bmi = make_interp_spline(time_points_values, threshold_values['BMI Threshold'].values.tolist() + [0])(smooth_years)
            smooth_threshold_bmi = np.array(list(map(lambda x: 0 if x < 0 else x , smooth_threshold_bmi)))
        
        
        # Healthy BMI Thin Line
        healthy_bmi_line = pd.DataFrame({
            "Year": smooth_years,
            "BMI": [25] * len(smooth_years),
            "Line": ["Healthy BMI"] * len(smooth_years)
        })
        
        
        st.header("Predicted BMI Progression After Surgery")
        st.write("In the graph, the predicted BMI outcome up to 5 years post-surgery is displayed.")
        st.markdown("""
        <div style="font-size:16px;">
            <span style="color:blue;">Predicted BMI Progression</span> is shown in blue, 
            <span style="color:orange;">its CI</span> is shown in orange, 
            and the threshold to a <span style="color:green;">Healthy BMI (<25)</span> is highlighted in green.
        </div>
        """, unsafe_allow_html=True)
    
        chart_placeholder = st.empty()  # Placeholder for the chart
        prob_placeholder = st.empty()   # Placeholder for probability display
        total_steps = len(smooth_years)  # Total steps for smooth animation
        print(f"Total Steps: {total_steps}. Len smooth years:{len(smooth_years)}")
        
        # Build animated chart
        for i in range(1, total_steps + 1, 1):  # Adjust steps for full range
            # Update chart dynamically
            current_smooth_years = smooth_years[:i]
            current_smooth_bmi_values = smooth_bmi_values[:i]
            current_smooth_bmi_upper = smooth_bmi_upper[:i]
            current_smooth_bmi_lower = smooth_bmi_lower[:i]
            current_smooth_bmi_threshold = smooth_threshold_bmi[:i]
            current_smooth_dm_prob = smooth_dm_probabilities[:i]
            
            # Get current time point and probability for display
            current_time_months = smooth_years[i-1]
            current_dm_probability = smooth_dm_probabilities[i-1]
            
            # Create data for dynamic charts
            current_chart_data = pd.DataFrame({
                "Year": np.concatenate([current_smooth_years, current_smooth_years]),
                "BMI": np.concatenate([current_smooth_bmi_upper, current_smooth_bmi_lower]),
                "Line": ["CI"] * len(current_smooth_years) + ["CI"] * len(current_smooth_years)
            })
            current_bmi_chart_data = pd.DataFrame({
                "Year": current_smooth_years,
                "BMI": current_smooth_bmi_values,
                "Line": ["Predicted BMI"] * len(current_smooth_years)
            })
            
            threshold_chart_data = pd.DataFrame({'Year' : current_smooth_years,
                                                 'Threshold': current_smooth_bmi_threshold,
                                                 'Line' : ['Threshold BMI'] * len(current_smooth_years)})
    
            # Create Altair charts
            # For the bmi_chart, you can use:
    
            bmi_chart = alt.Chart(current_bmi_chart_data).mark_line(
                strokeWidth=3  # Change size of Predicted BMI line
            ).encode(
                x=alt.X("Year", title="Months After Surgery"),
                y=alt.Y("BMI", title="Body Mass Index (BMI)", scale=alt.Scale(domain=[bmi_min, bmi_max])),
                color=alt.Color(
                    "Line",
                    scale=alt.Scale(
                        domain=["Predicted BMI", "CI", "Healthy BMI"],
                        range=["blue", "orange", "green"]  # Explicit color mapping
                    ),
                    legend=alt.Legend(title="Legend")
                ),
                tooltip=[
                    alt.Tooltip("Year:Q", title="Months After Surgery"),
                    alt.Tooltip("BMI:Q", title="BMI", format=".1f"),
                    alt.Tooltip("Line:N", title="Type")
                ]
            )
    
            bounds_chart = alt.Chart(current_chart_data).mark_line(
                strokeDash=[2, 2],  # Change dash style of MAE lines
                strokeWidth=4.0  # Change size of MAE lines
            ).encode(
                x="Year",
                y=alt.Y("BMI", title="Body Mass Index (BMI)", scale=alt.Scale(domain=[bmi_min, bmi_max])),
                color=alt.Color(
                    "Line",
                    scale=alt.Scale(
                        domain=["Predicted BMI", "CI", "Healthy BMI"],
                        range=["blue", "orange", "green"]  # Explicit color mapping
                    ),
                    legend=None
                ),
                tooltip=["Year:Q", "BMI:Q", "Line:N"]
            )
    
            healthy_line_chart = alt.Chart(healthy_bmi_line[:i]).mark_line(
                strokeWidth=1.5  # Change size of Healthy BMI line
            ).encode(
                x="Year",
                y=alt.Y("BMI", title="Body Mass Index (BMI)", scale=alt.Scale(domain=[bmi_min, bmi_max])),
                color=alt.Color(
                    "Line",
                    scale=alt.Scale(
                        domain=["Predicted BMI", "CI", "Healthy BMI"],
                        range=["blue", "orange", "green" ]  # Explicit color mapping
                    ),
                    legend=None
                ),
                tooltip=["Year:Q", "BMI:Q", "Line:N"]
            )
                
               
            # Combine charts and display
            final_chart = (bounds_chart + healthy_line_chart +  bmi_chart).properties(
                width=850, height=600
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
            
            # Display current DM probability below the chart
            if current_dm_probability > 0.8:
                prob_status = "High Risk"
                prob_color = "red"
            elif current_dm_probability > 0.5:
                prob_status = "Middle Risk"
                prob_color = "orange"
            else:
                prob_status = "Low Risk"
                prob_color = "green"
            
            # Convert months to a more readable format
            if current_time_months == 0:
                time_label = "Pre-operative"
            elif current_time_months < 12:
                time_label = f"{current_time_months:.0f} months"
            elif current_time_months == 12:
                time_label = "1 year"
            else:
                years = current_time_months / 12
                time_label = f"{years:.1f} years"
            
            time.sleep(0.0001)  # Animation delay
        
            # Only render the risk advise if the patient has preoperatibe Diabettes Mellitus Type 2
            if df_final['DMII_preoperative'].values[0] == 1:
                prob_placeholder.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin: 10px 0;">
                    <h3 style="margin: 0; color: #333;">Current Diabetes Risk Assessment</h3>
                    <p style="font-size: 18px; margin: 10px 0; color: #666;">Time Point: <strong>{time_label}</strong></p>
                    <p style="font-size: 24px; margin: 10px 0; color: {prob_color}; font-weight: bold;">
                        Diabetes Probability: {current_dm_probability:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
        # Cards for the full information
        if df_final['DMII_preoperative'].values[0] == 1:
            st.subheader("Diabetes Risk and BMI at Key Time Points")
            
            # Define time points and their corresponding indices
            time_points_display = [
                (0, "Pre-operative", "BMI before surgery", "DMII_preoperative_prob"),
                (3, "3 months", "bmi3", "dm3m_prob"),
                (6, "6 months", "bmi6", "dm6m_prob"),
                (12, "12 months", "bmi12", "dm12m_prob"),
                (18, "18 months", "bmi18", "dm18m_prob"),
                (24, "2 years", "bmi2y", "dm2y_prob"),
                (36, "3 years", "bmi3y", "dm3y_prob"),
                (48, "4 years", "bmi4y", "dm4y_prob"),
                (60, "5 years", "bmi5y", "dm5y_prob")
            ]
            
            # Create columns for better layout
            cols = st.columns(3)
            
            for idx, (months, time_label, bmi_col, prob_col) in enumerate(time_points_display):
                col_idx = idx % 3
                
                current_bmi = df_final[bmi_col].values[0]
                current_dm_probability = df_final[prob_col].values[0]
                
                if current_dm_probability > 0.8:
                    prob_status = "High Risk"
                    prob_color = "red"
                elif current_dm_probability > 0.5:
                    prob_status = "Middle Risk"
                    prob_color = "orange"
                else:
                    prob_status = "Low Risk"
                    prob_color = "green"
                
                with cols[col_idx]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; background-color: #f0f2f6; border-radius: 10px; margin: 10px 0;">
                        <h4 style="margin: 0; color: #333;">{time_label}</h4>
                        <p style="font-size: 16px; margin: 5px 0; color: #666;">
                            <strong>BMI: {current_bmi:.1f}</strong>
                        </p>
                        <p style="font-size: 16px; margin: 5px 0; color: {prob_color}; font-weight: bold;">
                            DM Risk: {current_dm_probability:.1%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

###############################################################################
def find_bmi_thresholds_speed(clf_model, df_final, time_points=['Pre', '3m', '6m', '12m', '18m', '2y', '3y', '4y']):
    thresholds = []
    bmi_range = np.arange(15, 50, 0.5)  # Test BMI values from 20 to 40
    
    time_map = {
        'Pre': ('BMI before surgery', 'dm3m', 0, 'DMII_preoperative'),
        '3m': ('bmi3', 'dm6m', 1, 'dm3m'),
        '6m': ('bmi6', 'dm12m', 2, 'dm6m'),
        '12m': ('bmi12', 'dm18m', 3, 'dm12m'),
        '18m': ('bmi18', 'dm2y', 4, 'dm18m'),
        '2y': ('bmi2y', 'dm3y', 5, 'dm2y'),
        '3y': ('bmi3y', 'dm4y', 6, 'dm3y'),
        '4y': ('bmi4y', 'dm5y', 7, 'dm4y'),
    }
    
    for time_point in time_points:
        bmi_col, next_dm_col, time_step, dm_current = time_map[time_point]
        # Expand df_final for all BMI values
        n_samples = len(df_final)
        n_bmi = len(bmi_range)
        # Repeat df_final n_bmi times (efficient broadcasting)
        expanded_data = pd.concat([df_final] * n_bmi, ignore_index=True)
        expanded_data[bmi_col] = np.tile(bmi_range, n_samples)
        # Set DM(t) and time_step
        expanded_data['DM(t)'] = expanded_data[dm_current]
        expanded_data['time_step'] = time_step
        # Select relevant features
        features = expanded_data[clf_model.feature_names_in_]
        # Make batch predictions
        pred_probs = clf_model.predict_proba(features)[:, 1]
        # Reshape to (n_samples, n_bmi) to find last occurrence where prob < 0.5
        pred_probs = pred_probs.reshape(n_samples, n_bmi)
        mask = pred_probs < 0.5
        
        # Find the last BMI value where probability is < 0.5
        # First flip the arrays horizontally to find the first True from the right
        flipped_mask = np.fliplr(mask)
        flipped_pred_probs = np.fliplr(pred_probs)
        threshold_indices = np.argmax(flipped_mask, axis=1)
        valid_thresholds = flipped_mask[np.arange(n_samples), threshold_indices]
        
        # Convert the indices back to the original array orientation
        threshold_indices = n_bmi - 1 - threshold_indices
        
        # Extract threshold BMI or mark as "No threshold found"
        threshold_bmi = np.where(valid_thresholds, bmi_range[threshold_indices], "No threshold found")
        prob_at_threshold = np.where(valid_thresholds, pred_probs[np.arange(n_samples), threshold_indices], None)
        
        # Store results
        thresholds.append(pd.DataFrame({
            'Time Point': time_point,
            'BMI Threshold': threshold_bmi,
            'Next Time Point': next_dm_col,
            'Probability at Threshold': prob_at_threshold
        }))
    
    return pd.concat(thresholds, ignore_index=True)


###############################################################################
def display_threshold_analysis(threshold_df):
    """
    Display the threshold analysis results with a nice format
    """
    st.subheader("BMI Thresholds for Diabetes Remission")
    st.markdown("""
    The table below shows the BMI thresholds at each time point that predict
    diabetes remission in the next period. If a patient's BMI is below these
    thresholds, they are more likely to achieve diabetes remission in the
    following period.
    """)
    
    # Format the dataframe for display
    styled_df = threshold_df.style.format({
        'BMI Threshold': lambda x: f'{x:.1f}' if isinstance(x, (int, float)) else x,
        'Probability at Threshold': lambda x: f'{x:.1%}' if isinstance(x, (int, float)) else '-'
    })
    
    st.dataframe(styled_df)
###############################################################################

# Parser input information
def parser_user_input(dataframe_input , reg_model , clf_model, compare_surgeries=False):
    ###########################################################################
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
    ###########################################################################
    # Section for the comparison of the two surgeries
    if compare_surgeries:
        # Create dataframes for both surgeries
        surgeries = ['Laparoscopic Sleeve Gastrectomy (LSG)', 'Laparoscopic Roux-en-Y Gastric Bypass (LRYGB)']
        results = {}
        
        for surgery_name in surgeries:
            df_surgery = dataframe_input.copy()
            df_surgery['surgery'] = surgery_name
            
            # Encode categorical features
            for i in dictionary_categorical_features.keys():
                if i in df_surgery.columns:
                    df_surgery[i] = df_surgery[i].map(dictionary_categorical_features[i])
            
            # Run regression and classification for this surgery
            predictions_df_regression = pd.DataFrame(reg_model.predict(df_surgery[reg_model.feature_names_in_.tolist()]))
            predictions_df_regression.columns = ['bmi3','bmi6','bmi12','bmi18','bmi2y','bmi3y','bmi4y','bmi5y']
            # Change BMI curve, depending of the effectivenes of surgery
            if surgery_name == 'Laparoscopic Sleeve Gastrectomy (LSG)':
                print(f"Original BMI curve: {predictions_df_regression}")
                predictions_df_regression = predictions_df_regression * 0.85
                print(f"Modified BMI curve for {surgery_name}: {predictions_df_regression}")
            if surgery_name == 'Laparoscopic Roux-en-Y Gastric Bypass (LRYGB)':
                print(f"Original BMI curve: {predictions_df_regression}")
                predictions_df_regression = predictions_df_regression * 1.0
                print(f"Modified BMI curve for {surgery_name}: {predictions_df_regression}")
            
            df_classification = pd.concat([df_surgery, predictions_df_regression], axis=1)
            predictions_df_classification = pd.DataFrame(clf_model.predict(df_classification[clf_model.feature_names_in_.tolist()]))
            predictions_df_classification.columns = ['dm3m','dm6m','dm12m','dm18m','dm2y','dm3y','dm4y','dm5y']
            
            _x = clf_model.predict_proba(df_classification[clf_model.feature_names_in_.tolist()])
            _x = [value[0][1] for value in _x]
            probas_df_classification = pd.DataFrame([_x])
            probas_df_classification.columns = ['dm3m_prob','dm6m_prob','dm12m_prob','dm18m_prob','dm2y_prob','dm3y_prob','dm4y_prob','dm5y_prob']
            
            df_final = pd.concat([df_classification, predictions_df_classification, probas_df_classification], axis=1)
            df_final['DMII_preoperative_prob'] = df_final['DMII_preoperative'].astype(float)
            
            # Apply BMI adjustments (existing logic)
            df_final = apply_enhanced_bmi_adjustments(df_final, clf_model)
            
            # Add confidence intervals (existing logic)
            bmi_columns = ['BMI before surgery', 'bmi3', 'bmi6', 'bmi12', 'bmi18', 'bmi2y', 'bmi3y', 'bmi4y', 'bmi5y']
            for col in bmi_columns:
                df_final[f'{col}_ci_lower'] = df_final[col]
                df_final[f'{col}_ci_upper'] = df_final[col]
                if col != 'BMI before surgery':
                    df_final[f'{col}_ci_lower'] = df_final[col] - training_mae
                    df_final[f'{col}_ci_upper'] = df_final[col] + training_mae
                    
            results[surgery_name] = df_final
            
            df_final = results
            
    ##########################################################################
    else: # Normal pattern
        # Regression part
        
        # Encode categorical features
        for i in dictionary_categorical_features.keys():
            if i in dataframe_input.columns:
                dataframe_input[i] = dataframe_input[i].map(dictionary_categorical_features[i])
        predictions_df_regression = pd.DataFrame(reg_model.predict(dataframe_input[reg_model.feature_names_in_.tolist()]))
        predictions_df_regression.columns = ['bmi3','bmi6','bmi12','bmi18','bmi2y','bmi3y','bmi4y','bmi5y']
        print(f"Regression df: {predictions_df_regression}")
       
        ###########################################################################
        # Classification part
        df_classification = pd.concat([dataframe_input,
                                       predictions_df_regression] , axis = 1)
        predictions_df_classification = pd.DataFrame(clf_model.predict(df_classification[clf_model.feature_names_in_.tolist()]))
        predictions_df_classification.columns = ['dm3m','dm6m','dm12m','dm18m','dm2y','dm3y' , 'dm4y','dm5y']
        print(f"Classification df: {predictions_df_classification}")
        _x = clf_model.predict_proba(df_classification[clf_model.feature_names_in_.tolist()])
        _x = [value[0][1] for value in _x]
        probas_df_classification = pd.DataFrame([_x])
        probas_df_classification.columns = ['dm3m_prob','dm6m_prob','dm12m_prob','dm18m_prob','dm2y_prob', 'dm3y_prob','dm4y_prob','dm5y_prob']
        print(f"Classification probas df: {probas_df_classification}")
        
        df_final = pd.concat([df_classification,
                              predictions_df_classification,
                              probas_df_classification] , axis = 1)
        df_final['DMII_preoperative_prob'] = df_final['DMII_preoperative'].astype(float)
    
        ###########################################################################
        # BMI-based DM probability adjustment based on BMI variation between time steps
           
        # Only apply BMI-based adjustments if patient has preoperative diabetes
        if df_final['DMII_preoperative'].iloc[0] == 1:
            
            df_final = apply_enhanced_bmi_adjustments(df_final, clf_model)
        
        ###########################################################################
        # Confidence interval calculation (existing code)
        
        # Add confidence intervals (95% confidence level)
        confidence_level = 0.90
        z_score = 1.67  # 99% confidence level
    
        # Calculate confidence intervals for BMI predictions
        bmi_columns = ['BMI before surgery', 'bmi3', 'bmi6', 'bmi12', 'bmi18', 'bmi2y', 'bmi3y', 'bmi4y', 'bmi5y']
        
        # Initialize confidence interval columns
        for col in bmi_columns:
            df_final[f'{col}_ci_lower'] = df_final[col]
            df_final[f'{col}_ci_upper'] = df_final[col]
            
            if col != 'BMI before surgery':  # Skip the initial BMI which is known
                std_error = df_final[col].std() if not pd.isna(df_final[col].std()) else df_final[col].mean() * 0.1
                df_final[f'{col}_ci_lower'] = df_final[col] - training_mae
                df_final[f'{col}_ci_upper'] = df_final[col] + training_mae
        
        # Create a summary dataframe for display
        summary_df = pd.DataFrame({
            'Time': ['Pre', '3m', '6m', '12m', '18m', '2y', '3y', '4y', '5y'],
            'BMI': df_final[bmi_columns].iloc[0].values,
            'BMI CI Lower': df_final[[f'{col}_ci_lower' for col in bmi_columns]].iloc[0].values,
            'BMI CI Upper': df_final[[f'{col}_ci_upper' for col in bmi_columns]].iloc[0].values,
            'DM Status': df_final[['DMII_preoperative', 'dm3m', 'dm6m', 'dm12m', 'dm18m', 'dm2y', 'dm3y', 'dm4y', 'dm5y']].iloc[0].values,
            'DM Likelihood (%)': (df_final[['DMII_preoperative_prob', 'dm3m_prob', 'dm6m_prob', 'dm12m_prob', 'dm18m_prob', 'dm2y_prob', 'dm3y_prob', 'dm4y_prob', 'dm5y_prob']].iloc[0].values * 100).round(2)
        })
        
        summary_df['DM Status'] = np.select(condlist = [summary_df[ 'DM Status'] == 1],
                                            choicelist = ['Diabetes'],
                                            default = 'Healed')  
    
    # Create the chart with thresholds
    chart = create_animated_evolution_chart(df_final, clf_model, predictions_df_regression)
    
    return df_final
###############################################################################
# Page configuration
st.set_page_config(
    page_title="AL Prediction App",
    layout = 'wide'
)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Initialize app
reg_model , clf_model = initialize_app()

# Option Menu configuration
with st.sidebar:
    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['Home' , 'Prediction'],
        icons = ['house' , 'book'],
        menu_icon = 'cast',
        default_index = 0,
        orientation = 'Vertical')
######################
# Home page layout
######################
if selected == 'Home':
    st.title('DM and BMI Prediction App')
    st.markdown("""
    This app contains 2 sections which you can access from the horizontal menu above.\n
    The sections are:\n
    Home: The main page of the app.\n
    **Prediction:** On this section you can select the patients information and
    the models iterate over all posible anastomotic configuration and surgeon experience for suggesting
    the best option.\n
    \n
    \n
    \n
    **Disclaimer:** This application and its results are only approved for research purposes.
    """)
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

###############################################################################
# Prediction page layout
if selected == 'Prediction':
    st.title('Prediction Section')
    st.subheader("Description")
    st.subheader("To predict BMI and DM Curve, you need to follow the steps below:")
    st.markdown("""
    1. Enter clinical parameters of patient on the left side bar.
    2. Press the "Predict" button and wait for the result.
    \n
    \n
    \n
    **Disclaimer:** This application and its results are only approved for research purposes.
    """)
    st.markdown("""
    This model predicts the probabilities DM on each time step.
    """)
    # Sidebar layout
    st.sidebar.title("Patiens Info")
    st.sidebar.subheader("Please choose parameters")
    
    # Input features
    st.sidebar.subheader("Basic Information:")
    age = st.sidebar.number_input("Age(Years):" , step = 1.0 , min_value = 19.0 , max_value = 99.0)
    bmi_pre = st.sidebar.number_input("Preoperative BMI:" , step = 0.5 , min_value = 30.0)
    
    sex = st.sidebar.radio(
        "Select Sex:",
        options = tuple(dictionary_categorical_features['sex (1 = female, 2=male)'].keys()),
    )
    
    #asa_score = st.sidebar.radio(
    #    "Select ASA Score:",
    #    options = tuple(dictionary_categorical_features['asa_score'].keys()),
    #)
    
    # Input: Binary data
    st.sidebar.subheader("Medical Conditions (Yes/No):")
    hypertension = int(st.sidebar.checkbox("Hypertension"))
    hyperlipidemia = int(st.sidebar.checkbox('Hyperlipidemia'))
    osas_preoperative = int(st.sidebar.checkbox('Obstructive Sleep Apnea Syndrome (OSAS)'))
    #depression = int(st.sidebar.checkbox('Depression'))
    DMII_preoperative = int(st.sidebar.checkbox('Diabetes Mellitus Type 2'))
    #antidiab_drug_preop_Oral_anticogulation = int(st.sidebar.checkbox('Antidiabetes drug preoperative oral anticogulation'))
    #antidiab_drug_preop_Insulin = int(st.sidebar.checkbox('Antidiabetes drug preoperative insulin'))
    #prior_abdominal_surgery = int(st.sidebar.checkbox('Prior abdominal surgery'))
    if DMII_preoperative == 1:
        antidiab_drug_preop_no_therapy = int(st.sidebar.checkbox('No Therapy'))
        antidiab_drug_preop_glp1_analogen = int(st.sidebar.checkbox('Antidiabetic Drug GLP1 Analogen'))
        antidiab_drug_preop_Oral_anticogulation = int(st.sidebar.checkbox('Oral Antidiabetic Drug'))
        antidiab_drug_preop_Insulin = int(st.sidebar.checkbox('Antidiabetic Drug Insulin'))
    else:
        antidiab_drug_preop_no_therapy = 0
        antidiab_drug_preop_glp1_analogen = 0
        antidiab_drug_preop_Oral_anticogulation = 0
        antidiab_drug_preop_Insulin = 0
        
        
        
        #comorbid_1_Myocardial_infarct = int(st.sidebar.checkbox('Myocardial Infarct'))
        #comorbid_2_heart_failure = int(st.sidebar.checkbox('Heart Failure'))
        #comorbid_6_pulmonary_disease = int(st.sidebar.checkbox('Pulmonary Disease'))
    #normal_dmII_pattern = int(st.sidebar.checkbox('Normal Diabetes Mellitus Type 2 pattern'))
    
    # Surgery Type
    st.sidebar.subheader("Surgery Comparison:")
    compare_surgeries = st.sidebar.checkbox("Compare both surgeries", value=True)
    if not compare_surgeries:
        surgery = st.sidebar.radio(
            "Select Surgery:",
            options = tuple(dictionary_categorical_features['surgery'].keys()),
        )
    else:
        surgery = None  # Will process both surgeries
    
    
    
    # Map binary options
    hypertension = inverse_dictionary['hypertension'][hypertension]
    hyperlipidemia = inverse_dictionary['hyperlipidemia'][hyperlipidemia]
    DMII_preoperative = inverse_dictionary['DMII_preoperative'][DMII_preoperative]
    antidiab_drug_preop_no_therapy = inverse_dictionary['antidiab_drug_preop_no_therapy'][antidiab_drug_preop_no_therapy]
    antidiab_drug_preop_glp1_analogen = inverse_dictionary['antidiab_drug_preop_glp1_analogen'][antidiab_drug_preop_glp1_analogen]
    antidiab_drug_preop_Oral_anticogulation = inverse_dictionary['antidiab_drug_preop_Oral_anticogulation'][antidiab_drug_preop_Oral_anticogulation]
    antidiab_drug_preop_Insulin = inverse_dictionary['antidiab_drug_preop_Insulin'][antidiab_drug_preop_Insulin]
    osas_preoperative = inverse_dictionary['osas_preoperative'][osas_preoperative]
    # Create dataframe with the input data
    dataframe_input = pd.DataFrame({'age_years' : [age],
                                'BMI before surgery' : [bmi_pre],
                                'sex (1 = female, 2=male)' : [sex],
                                'hypertension' : [hypertension],
                                'hyperlipidemia' : [hyperlipidemia],
                                'DMII_preoperative' : [DMII_preoperative],
                                'surgery' : [surgery if not compare_surgeries else 'Laparoscopic Sleeve Gastrectomy (LSG)'],  # Default value when comparing
                                'antidiab_drug_preop_Oral_anticogulation' : [antidiab_drug_preop_Oral_anticogulation],
                                'antidiab_drug_preop_Insulin' : [antidiab_drug_preop_Insulin],
                                'antidiab_drug_preop_no_therapy' : [antidiab_drug_preop_no_therapy],
                                'antidiab_drug_preop_glp1_analogen' : [antidiab_drug_preop_glp1_analogen],
                                'osas_preoperative' : [osas_preoperative]})
    # Parser input and make predictions
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    predict_button = st.button('Predict')
    if predict_button:
        with st.spinner("Making predictions..."):
            try:
                predictions = parser_user_input(dataframe_input, reg_model, clf_model, compare_surgeries)
                st.session_state.prediction_results = predictions
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        # Sponsor Images
        images = [r'images/collaborators.png']
        
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
