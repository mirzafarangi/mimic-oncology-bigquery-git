"""
Extract oncology patients from MIMIC-IV.
"""

import pandas as pd
from typing import Tuple, Dict, List
from mimic_client import MIMICClient

class OncologyExtractor:
    """Extract and analyze oncology patients."""
    
    def __init__(self, client: MIMICClient):
        self.client = client
        self.oncology_codes = client.config['oncology_codes']
    
    def extract_hematologic_patients(self, limit: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract patients with hematologic malignancies."""
        
        print(f"🔍 Extracting hematologic oncology patients (limit: {limit:,})...")
        
        # Build code conditions for SQL
        code_conditions = " OR ".join([f"d.icd_code LIKE '{code}'" for code in self.oncology_codes])
        
        # Main extraction query
        patient_query = f"""
        WITH oncology_diagnoses AS (
            SELECT DISTINCT
                d.subject_id,
                d.hadm_id,
                d.icd_code,
                CASE 
                    WHEN d.icd_code LIKE 'C81%' THEN 'Hodgkin Lymphoma'
                    WHEN d.icd_code LIKE 'C82%' OR d.icd_code LIKE 'C83%' THEN 'Non-Hodgkin Lymphoma'
                    WHEN d.icd_code LIKE 'C90%' THEN 'Multiple Myeloma'
                    WHEN d.icd_code LIKE 'C91%' OR d.icd_code LIKE 'C92%' THEN 'Leukemia'
                    ELSE 'Other Hematologic'
                END as malignancy_type,
                dd.long_title as diagnosis_name
            FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
            LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses` dd
                ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version
            WHERE d.icd_version = 10
                AND ({code_conditions})
        ),
        
        patient_cohort AS (
            SELECT 
                p.subject_id,
                p.gender,
                p.anchor_age,
                p.dod,
                COUNT(DISTINCT o.hadm_id) as total_admissions,
                STRING_AGG(DISTINCT o.malignancy_type, '; ') as malignancy_types,
                MIN(a.admittime) as first_admission,
                MAX(a.admittime) as last_admission
            FROM `physionet-data.mimiciv_3_1_hosp.patients` p
            INNER JOIN oncology_diagnoses o ON p.subject_id = o.subject_id
            LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a ON o.hadm_id = a.hadm_id
            GROUP BY p.subject_id, p.gender, p.anchor_age, p.dod
            ORDER BY p.subject_id
        )
        
        SELECT * FROM patient_cohort
        LIMIT {limit}
        """
        
        # Execute patient cohort query
        patients = self.client.query(patient_query)
        
        # Get detailed diagnoses for these patients
        patient_ids_str = ','.join([str(pid) for pid in patients['subject_id'].tolist()])
        
        diagnoses_query = f"""
        SELECT 
            d.subject_id,
            d.hadm_id,
            d.icd_code,
            CASE 
                WHEN d.icd_code LIKE 'C81%' THEN 'Hodgkin Lymphoma'
                WHEN d.icd_code LIKE 'C82%' OR d.icd_code LIKE 'C83%' THEN 'Non-Hodgkin Lymphoma'
                WHEN d.icd_code LIKE 'C90%' THEN 'Multiple Myeloma'
                WHEN d.icd_code LIKE 'C91%' OR d.icd_code LIKE 'C92%' THEN 'Leukemia'
                ELSE 'Other Hematologic'
            END as malignancy_type,
            dd.long_title as diagnosis_name,
            a.admittime,
            a.dischtime
        FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
        LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses` dd
            ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version
        LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
            ON d.hadm_id = a.hadm_id
        WHERE d.subject_id IN ({patient_ids_str})
            AND d.icd_version = 10
            AND ({code_conditions})
        ORDER BY d.subject_id, a.admittime
        """
        
        diagnoses = self.client.query(diagnoses_query)
        
        print(f"📊 Extracted {len(patients):,} patients with {len(diagnoses):,} diagnoses")
        
        return patients, diagnoses
    
    def get_cohort_summary(self, patients: pd.DataFrame) -> Dict:
        """Generate cohort summary statistics."""
        
        # Parse malignancy types
        all_malignancies = []
        for types_str in patients['malignancy_types'].dropna():
            all_malignancies.extend([t.strip() for t in types_str.split(';')])
        
        summary = {
            'total_patients': len(patients),
            'demographics': {
                'gender': patients['gender'].value_counts().to_dict(),
                'age_mean': patients['anchor_age'].mean(),
                'age_std': patients['anchor_age'].std(),
                'age_range': [patients['anchor_age'].min(), patients['anchor_age'].max()]
            },
            'malignancies': pd.Series(all_malignancies).value_counts().to_dict(),
            'mortality': {
                'deceased': patients['dod'].notna().sum(),
                'mortality_rate': patients['dod'].notna().mean()
            }
        }
        
        return summary
    
    def save_cohort(self, patients: pd.DataFrame, diagnoses: pd.DataFrame, 
                   output_dir: str = "outputs/") -> Dict[str, str]:
        """Save cohort data to files."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save files
        patients_file = f"{output_dir}/oncology_patients.csv"
        diagnoses_file = f"{output_dir}/oncology_diagnoses.csv"
        
        patients.to_csv(patients_file, index=False)
        diagnoses.to_csv(diagnoses_file, index=False)
        
        print(f"💾 Saved cohort data:")
        print(f"  - Patients: {patients_file}")
        print(f"  - Diagnoses: {diagnoses_file}")
        
        return {
            'patients': patients_file,
            'diagnoses': diagnoses_file
        }


def extract_oncology_cohort(limit: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Main function to extract oncology cohort."""
    
    # Initialize client and extractor
    client = MIMICClient()
    extractor = OncologyExtractor(client)
    
    # Extract patients and diagnoses
    patients, diagnoses = extractor.extract_hematologic_patients(limit=limit)
    
    # Generate summary
    summary = extractor.get_cohort_summary(patients)
    
    # Save data
    saved_files = extractor.save_cohort(patients, diagnoses)
    
    return patients, diagnoses, summary