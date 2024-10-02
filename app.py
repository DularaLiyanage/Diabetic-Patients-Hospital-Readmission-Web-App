from flask import Flask,request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import joblib

app = Flask(__name__)

with open('xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# pre processing part
def preprocess_data(data):
    """
    Preprocess data for prediction.

    Args
        data: dict of feature names and values for the model.
    """
    data_series = pd.Series(data)
    df = pd.DataFrame(data_series).transpose()

    # ************** MAPPING **************
    # admission_type_id
    admission_type_id_map = {
        1: 'emergency',
        2: 'urgent',
        3: 'other',
        4: 'other',
        5: 'other',
        6: 'other',
        7: 'other',
        8: 'other'
    }
    
    df['admission_type_id'] = df['admission_type_id'].map(admission_type_id_map)
    
    # discharge_disposition_id
    discharge_disposition_id_map = {
        1: 'home',
        2: 'facility',
        3: 'facility',
        4: 'facility',
        5: 'facility',
        6: 'home',
        7: 'other',
        8: 'home',
        9: 'other',
        10: 'facility',
        11: 'facility',
        12: 'facility',
        14: 'other',
        15: 'hospice',
        16: 'hospice',
        17: 'facility',
        23: 'other',
        24: 'facility',
        28: 'facility',
    }
    
    df['discharge_disposition_id'] = df['discharge_disposition_id'].map(discharge_disposition_id_map)

    # admission_source_id
    admission_source_id_map = {
        1: 'referral',
        2: 'referral',
        3: 'referral',
        4: 'transfer',
        5: 'transfer',
        6: 'transfer',
        7: 'emergency',
        8: 'other',
        9: 'other',
        10: 'transfer',
        11: 'transfer',
        13: 'transfer',
        14: 'transfer',
        17: 'transfer',
        20: 'transfer',
        22: 'transfer',
        25: 'other'
    }
    
    df['admission_source_id'] = df['admission_source_id'].map(admission_source_id_map)

    # diag_1, diag_2, diag_3
    def map_diag(x):
        diabetes = [
            '250.7', '250.6', '250.4', '250.32', '250.13', '250.03', '250.8', '250.42',
            '250.41', '250.02', '250.22', '250.82', '250.83', '250.33', '250.12',
            '250.01', '250.11', '250.3', '250.43', '250.2', '250.9', '250.21', '250.93',
            '250.91', '250.81', '250.53', '250.92', '250.51', '250.31', '250.01', '250'
        ]
    
        val = 'No'
        if x in diabetes:
            val = 'Yes'
        return val
    
    
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].apply(map_diag)

    # drug columns
    def get_dosage_changes(row, val, start_col_idx, end_col_idx):
        val_count = 0
        for col in row.index[start_col_idx:end_col_idx]:
            if row[col] == val:
                val_count += 1
        return val_count

    # Add new columns to dataframe to get the counts
    df['drugs_steady'] = np.zeros(df.shape[0])
    df['drugs_up'] = np.zeros(df.shape[0])
    df['drugs_down'] = np.zeros(df.shape[0])
    
    df['drugs_steady'] = df.apply(lambda row: get_dosage_changes(row, 'Steady', 19, 26), axis=1)
    df['drugs_up'] = df.apply(lambda row: get_dosage_changes(row, 'Up', 19, 26), axis=1)
    df['drugs_down'] = df.apply(lambda row: get_dosage_changes(row, 'Down', 19, 26), axis=1)
    # ************** MAPPING END **************

    # ************** ENCODING **************
    # One-Hot encode the features
    cat_col_list = ['race', 'gender', 'age', 'admission_type_id',
       'discharge_disposition_id', 'admission_source_id', 'diag_1',
       'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin',
       'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
       'rosiglitazone', 'insulin', 'change', 'diabetesMed']

    categories = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)',
       '[70-80)', '[80-90)', '[90-100)']
    cols_to_oh = np.delete(cat_col_list, 2)

    col_transformer = ColumnTransformer(transformers=[
        ('ord', OrdinalEncoder(categories=[categories]), ['age']),
        ('oh', OneHotEncoder(dtype=np.int64), cols_to_oh)
    ], remainder='passthrough')
    
    data_enc = col_transformer.fit_transform(df)

    # Simplyfy output feature names
    enc_out_feature_names = col_transformer.get_feature_names_out()
    cln_feature_names = []
    
    for col in enc_out_feature_names:
        cln_feature_names.append(col.split('__')[1])

    data_enc_df = pd.DataFrame(data_enc, columns=cln_feature_names)
    # ************** ENCODING END **************

    # ************** CREATE FULL DATAFRAME **************
    column_lst = ['age', 'race_AfricanAmerican', 'race_Asian', 'race_Caucasian',
       'race_Hispanic', 'race_Other', 'gender_Female', 'gender_Male',
       'admission_type_id_emergency', 'admission_type_id_other',
       'admission_type_id_urgent', 'discharge_disposition_id_facility',
       'discharge_disposition_id_home', 'discharge_disposition_id_hospice',
       'discharge_disposition_id_other', 'admission_source_id_emergency',
       'admission_source_id_other', 'admission_source_id_referral',
       'admission_source_id_transfer', 'diag_1_No', 'diag_1_Yes', 'diag_2_No',
       'diag_2_Yes', 'diag_3_No', 'diag_3_Yes', 'max_glu_serum_>200',
       'max_glu_serum_>300', 'max_glu_serum_Norm', 'max_glu_serum_none',
       'A1Cresult_>7', 'A1Cresult_>8', 'A1Cresult_Norm', 'A1Cresult_none',
       'metformin_Down', 'metformin_No', 'metformin_Steady', 'metformin_Up',
       'glimepiride_Down', 'glimepiride_No', 'glimepiride_Steady',
       'glimepiride_Up', 'glipizide_Down', 'glipizide_No', 'glipizide_Steady',
       'glipizide_Up', 'glyburide_Down', 'glyburide_No', 'glyburide_Steady',
       'glyburide_Up', 'pioglitazone_Down', 'pioglitazone_No',
       'pioglitazone_Steady', 'pioglitazone_Up', 'rosiglitazone_Down',
       'rosiglitazone_No', 'rosiglitazone_Steady', 'rosiglitazone_Up',
       'insulin_Down', 'insulin_No', 'insulin_Steady', 'insulin_Up',
       'change_Ch', 'change_No', 'diabetesMed_No', 'diabetesMed_Yes',
       'time_in_hospital', 'num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses', 'drugs_steady', 'drugs_up',
       'drugs_down']

    full_df = pd.DataFrame(np.zeros(shape=(1, len(column_lst)), dtype=int), columns=column_lst)

    # Add current dataframe values to the full dataframe
    for col in data_enc_df.columns:
        if col in full_df.columns:
            full_df[col] = data_enc_df[col]
    # ************** CREATE FULL DATAFRAME END **************

    # ************** FEATURE ENG **************
    # total_visits & emergency_visit_ratio
    full_df['total_visits'] = full_df['number_outpatient'] + full_df['number_emergency'] + full_df['number_inpatient']
    
    full_df['emergency_visits_ratio'] = round(full_df['number_emergency'] / (full_df['total_visits'] + 1), 3)
    full_df['emergency_visits_ratio'] = full_df['emergency_visits_ratio'].fillna(0)
    
    full_df = full_df.drop(['number_outpatient', 'number_emergency', 'number_inpatient'], axis=1)

    # drugs_steady_ratio, drugs_up_ratio & drugs_down_ratio
    full_df['drugs_steady_ratio'] = round(
        full_df['drugs_steady'] / (full_df['drugs_steady'] + full_df['drugs_up'] + full_df['drugs_down'] + 1), 3)
    
    full_df['drugs_up_ratio'] = round(
        full_df['drugs_up'] / (full_df['drugs_steady'] + full_df['drugs_up'] + full_df['drugs_down'] + 1), 3)
    
    full_df['drugs_down_ratio'] = round(
        full_df['drugs_down'] / (full_df['drugs_steady'] + full_df['drugs_up'] + full_df['drugs_down'] + 1), 3)
    
    full_df[['drugs_steady_ratio', 'drugs_up_ratio', 'drugs_down_ratio']] = full_df[[
        'drugs_steady_ratio', 'drugs_up_ratio', 'drugs_down_ratio']].fillna(0)
    
    full_df = full_df.drop(['drugs_steady', 'drugs_up', 'drugs_down'], axis=1)

    # visits_age_ratio
    full_df['visits_age_ratio'] = round(full_df['total_visits'] / (full_df['age'] + 1), 3)
    
    # diabetes_diag_ratio
    full_df['diabetes_diag_ratio'] = round(
        full_df[['diag_1_Yes', 'diag_2_Yes', 'diag_3_Yes']].sum(axis=1) / 3, 3)

    # combine time_in_hospital, num_procedures, num_lab_procedures & num_medications
    full_df['lab_procedure_hospital_time_ratio'] = round(full_df['num_lab_procedures'] / (full_df['time_in_hospital']), 3)
    
    full_df['num_medications_hospital_ratio'] = round(full_df['num_medications'] / (full_df['time_in_hospital']), 3)
    
    full_df['procedures_and_medications'] = full_df['num_procedures'] * full_df['num_medications']
    
    full_df['lab_procedures_and_medications'] = full_df['num_lab_procedures'] * full_df['num_medications']
    
    full_df = full_df.drop(['num_lab_procedures', 'num_procedures', 'num_medications'], axis=1)
    # ************** FEATURE ENG END **************

    # ************** SCALING **************
    # Load the scaler
    scaler = joblib.load('scaler.pkl')
    df_scaled = scaler.transform(full_df)
    
    df_scaled = pd.DataFrame(df_scaled, columns=full_df.columns.values)
    # ************** SCALING END **************
    
    return df_scaled

#routes
@app.route("/")
def main():
    return render_template('main.html')

# routes
@app.route('/predict', methods=['POST'])
def predict():
    # Capture form data and convert types where needed
    form_data = {
        'race': request.form['race'],
        'age': request.form['age'],  
        'gender': request.form['gender'],
        'admission_type_id': int(request.form['admission_type_id']),  
        'discharge_disposition_id': int(request.form['discharge_disposition_id']),  
        'admission_source_id': int(request.form['admission_source_id']),  
        'time_in_hospital': int(request.form['time_in_hospital']), 
        'num_lab_procedures': int(request.form['num_lab_procedures']),  
        'num_procedures': int(request.form['num_procedures']),  
        'num_medications': int(request.form['num_medications']),  
        'number_outpatient': int(request.form['number_outpatient']),  
        'number_emergency': int(request.form['number_emergency']),  
        'number_inpatient': int(request.form['number_inpatient']),  
        'diag_1': request.form['diag_1'],  
        'diag_2': request.form['diag_2'],
        'diag_3': request.form['diag_3'],
        'number_diagnoses': int(request.form['number_diagnoses']),  
        'max_glu_serum': request.form['max_glu_serum'],  
        'A1Cresult': request.form['A1Cresult'],
        'metformin': request.form['metformin'],
        'glimepiride': request.form['glimepiride'],
        'glipizide': request.form['glipizide'],
        'glyburide': request.form['glyburide'],
        'pioglitazone': request.form['pioglitazone'],
        'rosiglitazone': request.form['rosiglitazone'],
        'insulin': request.form['insulin'],
        'change': request.form['medicinech'],
        'diabetesMed': request.form['diabetesMed']
    }

    # Preprocess the data for model input
    preprocessed_data = preprocess_data(form_data)

    # Make a prediction using the model
    prediction = xgb_model.predict(preprocessed_data)

    # Make sure the model supports predict_proba
    if hasattr(xgb_model, "predict_proba"):
        prediction_proba = xgb_model.predict_proba(preprocessed_data)
        # Get the probability of the positive class
        probability = round(max(prediction_proba[0])* 100, 2)  # Probability for positive class
    else:
        probability = None  # Handle case where probabilities can't be retrieved

    # Format the prediction result
    prediction_result = 'Readmission Likely' if prediction[0] == 1 else 'No Readmission'

    # Render the result on a new page
    return render_template('result.html', result=prediction_result, probability=probability)

if __name__=='__main__':
    app.run(debug=True)