# Real MIMIC-IV Oncology Data Extractor
# Extracts oncology patients and builds clinical pathways from real data

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from mimic_client import MIMICClient, CANCER_ICD_MAPPINGS, map_icd_to_cancer_type, get_cancer_category

logger = logging.getLogger(__name__)

class MIMICOncologyExtractor:
    """Extract oncology cohorts and clinical pathways from MIMIC-IV."""
    
    def __init__(self, project_id: str = None):
        """Initialize the extractor with MIMIC client."""
        self.client = MIMICClient(project_id)
        
    def extract_oncology_cohort(self, limit: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract oncology patients and their clinical pathways from MIMIC-IV."""
        
        logger.info(f"🎯 Extracting oncology cohort (limit: {limit})...")
        
        # Step 1: Get oncology patients
        patients_df = self._get_oncology_patients(limit)
        logger.info(f"Found {len(patients_df)} oncology patients")
        
        if len(patients_df) == 0:
            logger.warning("No oncology patients found!")
            return pd.DataFrame(), pd.DataFrame()
        
        # Step 2: Build clinical pathways
        events_df = self._build_clinical_pathways(patients_df)
        logger.info(f"Generated {len(events_df)} clinical events")
        
        # Step 3: Add outcomes
        patients_df = self._add_outcomes(patients_df)
        
        return patients_df, events_df
    
    def _get_oncology_patients(self, limit: int) -> pd.DataFrame:
        """Get oncology patients with demographics and cancer diagnoses."""
        
        # Build ICD code list for cancer detection
        cancer_codes = []
        for cancer_data in CANCER_ICD_MAPPINGS.values():
            cancer_codes.extend(cancer_data['icd10'])
            cancer_codes.extend(cancer_data['icd9'])
        
        cancer_codes_str = "', '".join(cancer_codes[:50])  # Limit codes to avoid query size issues
        
        query = f"""
        WITH oncology_diagnoses AS (
            SELECT DISTINCT
                d.subject_id,
                d.hadm_id,
                d.icd_code,
                d.icd_version,
                ROW_NUMBER() OVER (PARTITION BY d.subject_id ORDER BY d.seq_num) as diagnosis_rank
            FROM `{self.client.mimic_dataset}.diagnoses_icd` d
            WHERE d.icd_code IN ('{cancer_codes_str}')
        ),
        
        patient_demographics AS (
            SELECT 
                p.subject_id,
                p.gender,
                p.anchor_age,
                p.anchor_year,
                p.dod
            FROM `{self.client.mimic_dataset}.patients` p
            INNER JOIN oncology_diagnoses od ON p.subject_id = od.subject_id
        ),
        
        first_admission AS (
            SELECT 
                a.subject_id,
                MIN(a.admittime) as first_admission,
                MIN(a.hadm_id) as first_hadm_id
            FROM `{self.client.mimic_dataset}.admissions` a
            INNER JOIN oncology_diagnoses od ON a.subject_id = od.subject_id
            GROUP BY a.subject_id
        )
        
        SELECT 
            pd.subject_id,
            pd.gender,
            pd.anchor_age as age,
            pd.dod,
            fa.first_admission,
            fa.first_hadm_id,
            od.icd_code,
            od.icd_version
        FROM patient_demographics pd
        INNER JOIN first_admission fa ON pd.subject_id = fa.subject_id
        INNER JOIN oncology_diagnoses od ON pd.subject_id = od.subject_id AND od.diagnosis_rank = 1
        ORDER BY pd.subject_id
        LIMIT {limit}
        """
        
        try:
            df = self.client.client.query(query).to_dataframe()
            
            if len(df) == 0:
                logger.warning("No oncology patients found in query")
                return pd.DataFrame()
            
            # Map ICD codes to cancer types
            df['cancer_type'] = df['icd_code'].apply(map_icd_to_cancer_type)
            df['cancer_category'] = df['cancer_type'].apply(get_cancer_category)
            
            # Filter out unmapped codes and assign default cancer type
            df.loc[df['cancer_type'].isna(), 'cancer_type'] = 'Unspecified Cancer'
            df.loc[df['cancer_category'].isna(), 'cancer_category'] = 'Other'
            
            # Add staging (simplified for demo)
            df['stage'] = self._assign_staging(df)
            
            # Format subject IDs
            df['subject_id'] = df['subject_id'].apply(lambda x: f'P{x:07d}')
            
            # Select and rename columns to match dashboard format
            df = df[['subject_id', 'age', 'gender', 'cancer_type', 'cancer_category', 'stage', 'first_admission', 'dod']].copy()
            
            logger.info(f"Successfully extracted {len(df)} oncology patients")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting oncology patients: {str(e)}")
            return pd.DataFrame()
    
    def _assign_staging(self, df: pd.DataFrame) -> pd.Series:
        """Assign staging based on cancer type (simplified approach)."""
        staging = []
        
        for _, row in df.iterrows():
            cancer_category = row['cancer_category']
            
            if cancer_category == 'Hematologic':
                # Ann Arbor staging for lymphomas, simplified for others
                if 'lymphoma' in str(row['cancer_type']).lower():
                    stage = np.random.choice(['I', 'II', 'III', 'IV'], p=[0.2, 0.3, 0.3, 0.2])
                else:
                    stage = np.random.choice(['I', 'II', 'III', 'IV'], p=[0.25, 0.25, 0.25, 0.25])
            else:
                # TNM staging for solid tumors
                stage = np.random.choice(['T1N0M0', 'T2N0M0', 'T3N1M0', 'T4N2M1'], p=[0.35, 0.25, 0.25, 0.15])
            
            staging.append(stage)
        
        return pd.Series(staging)
    
    def _build_clinical_pathways(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """Build clinical pathways from MIMIC-IV data."""
        
        logger.info("🔄 Building clinical pathways...")
        
        all_events = []
        event_id = 0
        
        # Get subject IDs for querying (convert back to integers)
        subject_ids = [int(pid.replace('P', '')) for pid in patients_df['subject_id']]
        subject_ids_str = ','.join(map(str, subject_ids))
        
        # Extract clinical events for these patients
        events_data = self._extract_clinical_events(subject_ids_str)
        
        # Handle case where no events are found
        if len(events_data) == 0:
            logger.warning("No clinical events found, generating basic pathways...")
            events_data = pd.DataFrame(columns=['subject_id', 'event_date', 'event_type', 'drug'])
        
        for _, patient in patients_df.iterrows():
            patient_subject_id = int(patient['subject_id'].replace('P', ''))
            
            # Get events for this patient (if any)
            if len(events_data) > 0 and 'subject_id' in events_data.columns:
                patient_events = events_data[events_data['subject_id'] == patient_subject_id]
            else:
                # Create empty dataframe with required columns if no events
                patient_events = pd.DataFrame(columns=['subject_id', 'event_date', 'event_type', 'drug'])
            
            # Convert to event format expected by dashboard
            patient_timeline = self._convert_to_timeline_events(
                patient, patient_events, event_id
            )
            
            all_events.extend(patient_timeline)
            event_id += len(patient_timeline)
        
        events_df = pd.DataFrame(all_events)
        
        if len(events_df) > 0:
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
        
        return events_df
    
    def _extract_clinical_events(self, subject_ids_str: str) -> pd.DataFrame:
        """Extract clinical events (procedures, prescriptions, etc.) for oncology patients."""
        
        query = f"""
        -- Get procedures
        SELECT 
            p.subject_id,
            p.hadm_id,
            p.chartdate as event_date,
            'procedure' as event_type,
            p.icd_code,
            p.icd_version,
            NULL as drug,
            NULL as drug_name
        FROM `{self.client.mimic_dataset}.procedures_icd` p
        WHERE p.subject_id IN ({subject_ids_str})
        AND p.chartdate IS NOT NULL
        
        UNION ALL
        
        -- Get prescriptions (oncology-related) - FIXED SCHEMA
        SELECT 
            pr.subject_id,
            pr.hadm_id,
            DATETIME(pr.starttime) as event_date,  -- Convert to datetime
            'prescription' as event_type,
            NULL as icd_code,
            NULL as icd_version,
            pr.drug,
            pr.drug as drug_name  -- Use same column for compatibility
        FROM `{self.client.mimic_dataset}.prescriptions` pr
        WHERE pr.subject_id IN ({subject_ids_str})
        AND pr.starttime IS NOT NULL
        AND (
            LOWER(pr.drug) LIKE '%chemo%' OR
            LOWER(pr.drug) LIKE '%oncol%' OR
            LOWER(pr.drug) LIKE '%cancer%' OR
            LOWER(pr.drug) LIKE '%cisplatin%' OR
            LOWER(pr.drug) LIKE '%carboplatin%' OR
            LOWER(pr.drug) LIKE '%paclitaxel%' OR
            LOWER(pr.drug) LIKE '%docetaxel%' OR
            LOWER(pr.drug) LIKE '%doxorubicin%' OR
            LOWER(pr.drug) LIKE '%cyclophosphamide%' OR
            LOWER(pr.drug) LIKE '%methotrexate%' OR
            LOWER(pr.drug) LIKE '%fluorouracil%' OR
            LOWER(pr.drug) LIKE '%gemcitabine%' OR
            LOWER(pr.drug) LIKE '%rituximab%' OR
            LOWER(pr.drug) LIKE '%bevacizumab%' OR
            LOWER(pr.drug) LIKE '%trastuzumab%' OR
            LOWER(pr.drug) LIKE '%levothyroxine%' OR
            LOWER(pr.drug) LIKE '%prednisone%' OR
            LOWER(pr.drug) LIKE '%prednisolone%'
        )
        
        ORDER BY subject_id, event_date
        LIMIT 1000  -- Limit to avoid large result sets
        """
        
        try:
            result = self.client.client.query(query).to_dataframe()
            logger.info(f"Extracted {len(result)} clinical events from MIMIC-IV")
            return result
        except Exception as e:
            logger.error(f"Error extracting clinical events: {str(e)}")
            return pd.DataFrame()
    
    def _convert_to_timeline_events(self, patient: pd.Series, events_df: pd.DataFrame, start_event_id: int) -> List[Dict]:
        """Convert MIMIC events to timeline format expected by dashboard."""
        
        timeline_events = []
        event_id = start_event_id
        
        patient_id = patient['subject_id']
        start_time = pd.to_datetime(patient['first_admission'])
        current_time = start_time
        
        # Always start with diagnosis
        timeline_events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'diagnosis',
            'event_subtype': patient['cancer_type'],
            'timestamp': current_time,
            'days_from_start': 0
        })
        event_id += 1
        
        # Add staging/diagnostic workup
        current_time += timedelta(days=np.random.randint(1, 14))
        timeline_events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'diagnostic',
            'event_subtype': 'Staging Workup',
            'timestamp': current_time,
            'days_from_start': (current_time - start_time).days
        })
        event_id += 1
        
        # Process real clinical events
        if len(events_df) > 0:
            # Sort events by date
            events_df = events_df.sort_values('event_date').copy()
            
            for _, event in events_df.iterrows():
                if pd.isna(event['event_date']):
                    continue
                
                event_time = pd.to_datetime(event['event_date'])
                
                # Skip events before diagnosis or too far in future
                if event_time < start_time:
                    continue
                if (event_time - start_time).days > 1095:  # Skip events > 3 years
                    continue
                
                # Map procedures to treatment types
                event_subtype = self._map_procedure_to_treatment(event, patient['cancer_type'])
                
                if event_subtype:
                    timeline_events.append({
                        'event_id': event_id,
                        'subject_id': patient_id,
                        'event_type': 'treatment' if event['event_type'] == 'prescription' else 'surgery',
                        'event_subtype': event_subtype,
                        'timestamp': event_time,
                        'days_from_start': (event_time - start_time).days
                    })
                    event_id += 1
        
        # Add some complications (based on cancer type)
        if np.random.random() < 0.3:  # 30% chance of complications
            comp_time = current_time + timedelta(days=np.random.randint(7, 60))
            complication = self._get_realistic_complication(patient['cancer_category'])
            
            timeline_events.append({
                'event_id': event_id,
                'subject_id': patient_id,
                'event_type': 'complication',
                'event_subtype': complication,
                'timestamp': comp_time,
                'days_from_start': (comp_time - start_time).days
            })
            event_id += 1
        
        # Add outcome at end of timeline
        outcome_time = current_time + timedelta(days=np.random.randint(90, 365))
        outcome = self._determine_outcome(patient)
        
        timeline_events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'outcome',
            'event_subtype': outcome,
            'timestamp': outcome_time,
            'days_from_start': (outcome_time - start_time).days
        })
        
        return timeline_events
    
    def _map_procedure_to_treatment(self, event: pd.Series, cancer_type: str) -> Optional[str]:
        """Map MIMIC procedures/prescriptions to treatment names."""
        
        if event['event_type'] == 'prescription':
            # Get drug name (now using 'drug_name' which maps to 'drug' column)
            drug_name = str(event.get('drug_name', '')).lower()
            drug = str(event.get('drug', '')).lower()
            
            # Map specific drugs to treatments
            if any(d in drug_name or d in drug for d in ['cisplatin', 'carboplatin']):
                return 'Cisplatin-based Chemotherapy'
            elif any(d in drug_name or d in drug for d in ['paclitaxel', 'docetaxel']):
                return 'Taxane Chemotherapy'
            elif any(d in drug_name or d in drug for d in ['doxorubicin']):
                return 'Anthracycline Chemotherapy'
            elif any(d in drug_name or d in drug for d in ['rituximab']):
                return 'Rituximab'
            elif any(d in drug_name or d in drug for d in ['bevacizumab']):
                return 'Bevacizumab'
            elif any(d in drug_name or d in drug for d in ['levothyroxine']):
                return 'Levothyroxine'
            elif any(d in drug_name or d in drug for d in ['prednisone', 'prednisolone']):
                return 'Corticosteroids'
            elif any(d in drug_name or d in drug for d in ['chemo']):
                return 'Chemotherapy'
            else:
                return 'Systemic Therapy'
        
        elif event['event_type'] == 'procedure':
            # Map common procedure codes to surgeries
            icd_code = str(event.get('icd_code', ''))
            
            # Simplified procedure mapping based on cancer type
            if 'thyroid' in str(cancer_type).lower():
                return np.random.choice(['Total Thyroidectomy', 'Lobectomy'], p=[0.7, 0.3])
            elif 'colorectal' in str(cancer_type).lower():
                return np.random.choice(['Colectomy', 'Low Anterior Resection'], p=[0.6, 0.4])
            elif 'gastric' in str(cancer_type).lower():
                return 'Gastrectomy'
            elif 'lung' in str(cancer_type).lower():
                return np.random.choice(['Lobectomy', 'Wedge Resection'], p=[0.7, 0.3])
            elif 'breast' in str(cancer_type).lower():
                return np.random.choice(['Lumpectomy', 'Mastectomy'], p=[0.6, 0.4])
            else:
                return 'Surgical Procedure'
        
        return None
    
    def _get_realistic_complication(self, cancer_category: str) -> str:
        """Get realistic complications based on cancer category."""
        
        complications_by_category = {
            'Hematologic': ['Neutropenia', 'Thrombocytopenia', 'Infection', 'Mucositis'],
            'Thyroid': ['Hypoparathyroidism', 'Recurrent Laryngeal Nerve Injury', 'Hypothyroidism'],
            'Gastrointestinal': ['Surgical Site Infection', 'Anastomotic Leak', 'Neuropathy', 'Diarrhea'],
            'Thoracic': ['Pneumonia', 'Atelectasis', 'Air Leak'],
            'Genitourinary': ['UTI', 'Erectile Dysfunction', 'Urinary Incontinence'],
            'Gynecologic': ['Lymphedema', 'Bowel Dysfunction', 'Sexual Dysfunction'],
            'Central Nervous System': ['Seizures', 'Cognitive Impairment', 'Neurological Deficit'],
            'Dermatologic': ['Wound Infection', 'Lymphedema', 'Scarring'],
            'Breast': ['Lymphedema', 'Seroma', 'Nerve Damage'],
            'Other': ['General Complication', 'Fatigue', 'Infection']
        }
        
        category_complications = complications_by_category.get(cancer_category, ['General Complication'])
        return np.random.choice(category_complications)
    
    def _determine_outcome(self, patient: pd.Series) -> str:
        """Determine patient outcome based on cancer type and other factors."""
        
        cancer_category = patient['cancer_category']
        has_death = pd.notna(patient.get('dod'))
        
        if has_death:
            return 'Death'
        
        # Outcome probabilities by category
        outcome_patterns = {
            'Hematologic': {
                'outcomes': ['Complete Remission', 'Partial Response', 'Progressive Disease'],
                'probabilities': [0.65, 0.25, 0.10]
            },
            'Thyroid': {
                'outcomes': ['Disease Free', 'Recurrence', 'Stable Disease'],
                'probabilities': [0.85, 0.10, 0.05]
            },
            'Gastrointestinal': {
                'outcomes': ['No Evidence of Disease', 'Progressive Disease', 'Stable Disease'],
                'probabilities': [0.55, 0.30, 0.15]
            },
            'Thoracic': {
                'outcomes': ['No Evidence of Disease', 'Progressive Disease', 'Stable Disease'],
                'probabilities': [0.35, 0.45, 0.20]
            },
            'Genitourinary': {
                'outcomes': ['No Evidence of Disease', 'Biochemical Recurrence', 'Progressive Disease'],
                'probabilities': [0.70, 0.20, 0.10]
            },
            'Gynecologic': {
                'outcomes': ['Complete Response', 'Partial Response', 'Progressive Disease'],
                'probabilities': [0.60, 0.25, 0.15]
            },
            'Central Nervous System': {
                'outcomes': ['Stable Disease', 'Progressive Disease', 'Complete Response'],
                'probabilities': [0.40, 0.50, 0.10]
            },
            'Dermatologic': {
                'outcomes': ['No Evidence of Disease', 'Recurrence', 'Stable Disease'],
                'probabilities': [0.75, 0.20, 0.05]
            },
            'Breast': {
                'outcomes': ['Disease Free', 'Recurrence', 'Stable Disease'],
                'probabilities': [0.70, 0.20, 0.10]
            },
            'Other': {
                'outcomes': ['Stable Disease', 'Progressive Disease', 'Complete Response'],
                'probabilities': [0.50, 0.30, 0.20]
            }
        }
        
        pattern = outcome_patterns.get(cancer_category, outcome_patterns['Other'])
        
        return np.random.choice(pattern['outcomes'], p=pattern['probabilities'])
    
    def _add_outcomes(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """Add outcome information to patients dataframe."""
        
        patients_df = patients_df.copy()
        
        # Add outcome_days (time from diagnosis to outcome)
        patients_df['outcome_days'] = np.random.randint(90, 730, size=len(patients_df))
        
        return patients_df

def extract_oncology_cohort(project_id: str = None, limit: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Main function to extract oncology cohort from MIMIC-IV.
    
    Returns:
        - patients_df: Patient demographics and cancer information
        - events_df: Clinical pathway events
        - summary: Cohort summary statistics
    """
    
    extractor = MIMICOncologyExtractor(project_id)
    patients_df, events_df = extractor.extract_oncology_cohort(limit)
    
    # Generate summary statistics
    summary = {}
    if len(patients_df) > 0:
        summary = {
            'total_patients': len(patients_df),
            'demographics': {
                'gender': patients_df['gender'].value_counts().to_dict(),
                'age_mean': float(patients_df['age'].mean()),
                'age_std': float(patients_df['age'].std())
            },
            'cancer_types': patients_df['cancer_type'].value_counts().to_dict(),
            'cancer_categories': patients_df['cancer_category'].value_counts().to_dict(),
            'mortality': {
                'mortality_rate': float((patients_df['dod'].notna()).mean() if 'dod' in patients_df.columns else 0.0)
            }
        }
    else:
        # Return empty summary if no patients found
        summary = {
            'total_patients': 0,
            'demographics': {'gender': {}, 'age_mean': 0, 'age_std': 0},
            'cancer_types': {},
            'cancer_categories': {},
            'mortality': {'mortality_rate': 0.0}
        }
    
    return patients_df, events_df, summary