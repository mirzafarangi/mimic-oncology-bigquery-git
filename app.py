# Oncology Pathway Mapping Dashboard
# Complete working Streamlit application

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
from lifelines import KaplanMeierFitter
from scipy.stats import chi2_contingency, fisher_exact
import json

# Set page configuration
st.set_page_config(
    page_title="Outcome Path Mapping Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# DATA GENERATION (Replace with real data loader)
# =============================================

@st.cache_data
def generate_sample_data(n_patients: int = 500):
    """Generate comprehensive oncology patient data including hematologic, thyroid, and GI cancers."""
    
    np.random.seed(42)
    random.seed(42)
    
    # Expanded cancer types with realistic distributions
    cancer_types = {
        # Hematologic malignancies (40%)
        'Hodgkin Lymphoma': 0.06,
        'Non-Hodgkin Lymphoma': 0.14, 
        'Multiple Myeloma': 0.08,
        'Acute Leukemia': 0.06,
        'Chronic Leukemia': 0.06,
        
        # Thyroid cancers (15%)
        'Papillary Thyroid Carcinoma': 0.10,
        'Follicular Thyroid Carcinoma': 0.03,
        'Medullary Thyroid Carcinoma': 0.015,
        'Anaplastic Thyroid Carcinoma': 0.005,
        
        # Gastrointestinal cancers (45%)
        'Colorectal Cancer': 0.20,
        'Gastric Cancer': 0.08,
        'Pancreatic Cancer': 0.07,
        'Hepatocellular Carcinoma': 0.06,
        'Esophageal Cancer': 0.04
    }
    
    # Patient demographics
    patients = []
    for i in range(n_patients):
        # Select cancer type based on probabilities
        cancer_type = np.random.choice(
            list(cancer_types.keys()), 
            p=list(cancer_types.values())
        )
        
        # Age distributions vary by cancer type
        if 'Thyroid' in cancer_type:
            base_age = np.random.normal(45, 15)  # Thyroid cancers often younger
        elif cancer_type in ['Colorectal Cancer', 'Gastric Cancer', 'Pancreatic Cancer']:
            base_age = np.random.normal(68, 12)  # GI cancers often older
        else:
            base_age = np.random.normal(65, 15)  # General oncology population
            
        # Gender distributions vary by cancer type
        if cancer_type in ['Papillary Thyroid Carcinoma', 'Follicular Thyroid Carcinoma']:
            gender_prob = [0.25, 0.75]  # Female predominant
        elif cancer_type in ['Esophageal Cancer', 'Hepatocellular Carcinoma']:
            gender_prob = [0.75, 0.25]  # Male predominant
        else:
            gender_prob = [0.55, 0.45]  # Slight male predominance
        
        # Staging system depends on cancer type
        if any(blood_cancer in cancer_type for blood_cancer in ['Lymphoma', 'Leukemia', 'Myeloma']):
            # Hematologic staging (Ann Arbor for lymphomas, others use different systems)
            stage = np.random.choice(['I', 'II', 'III', 'IV'], p=[0.2, 0.3, 0.3, 0.2])
        else:
            # TNM staging for solid tumors
            stage = np.random.choice(['T1N0M0', 'T2N0M0', 'T3N1M0', 'T4N2M1'], p=[0.35, 0.25, 0.25, 0.15])
        
        patients.append({
            'subject_id': f'P{i:04d}',
            'age': base_age,
            'gender': np.random.choice(['M', 'F'], p=gender_prob),
            'cancer_type': cancer_type,
            'cancer_category': _get_cancer_category(cancer_type),
            'stage': stage,
            'first_admission': datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1095))
        })
    
    patients_df = pd.DataFrame(patients)
    patients_df['age'] = patients_df['age'].clip(18, 90).round().astype(int)
    
    # Generate cancer-specific clinical events
    events = []
    event_id = 0
    
    for _, patient in patients_df.iterrows():
        patient_events, event_id = _generate_cancer_specific_events(patient, event_id)
        events.extend(patient_events)
    
    events_df = pd.DataFrame(events)
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    
    # Merge patient outcomes back to patients_df
    outcomes = events_df[events_df['event_type'] == 'outcome'][['subject_id', 'event_subtype', 'days_from_start']]
    outcomes.columns = ['subject_id', 'outcome', 'outcome_days']
    patients_df = patients_df.merge(outcomes, on='subject_id', how='left')
    
    # Fill missing outcomes (shouldn't happen, but safety)
    patients_df['outcome'] = patients_df['outcome'].fillna('Unknown')
    patients_df['outcome_days'] = patients_df['outcome_days'].fillna(365)
    
    return patients_df, events_df

def _get_cancer_category(cancer_type):
    """Categorize cancer types for filtering and analysis."""
    if any(blood_cancer in cancer_type for blood_cancer in ['Lymphoma', 'Leukemia', 'Myeloma']):
        return 'Hematologic'
    elif 'Thyroid' in cancer_type:
        return 'Thyroid'
    else:
        return 'Gastrointestinal'

def _generate_cancer_specific_events(patient, event_id):
    """Generate cancer-specific treatment pathways and events."""
    
    patient_id = patient['subject_id']
    cancer_type = patient['cancer_type']
    cancer_category = patient['cancer_category']
    current_time = patient['first_admission']
    events = []
    
    # Always start with diagnosis
    events.append({
        'event_id': event_id,
        'subject_id': patient_id,
        'event_type': 'diagnosis',
        'event_subtype': cancer_type,
        'timestamp': current_time,
        'days_from_start': 0
    })
    event_id += 1
    current_time += timedelta(days=np.random.randint(1, 14))
    
    # Cancer-specific treatment pathways
    if cancer_category == 'Hematologic':
        events, event_id, current_time = _generate_hematologic_pathway(
            events, event_id, patient_id, cancer_type, current_time, patient['first_admission']
        )
    elif cancer_category == 'Thyroid':
        events, event_id, current_time = _generate_thyroid_pathway(
            events, event_id, patient_id, cancer_type, current_time, patient['first_admission']
        )
    else:  # Gastrointestinal
        events, event_id, current_time = _generate_gi_pathway(
            events, event_id, patient_id, cancer_type, current_time, patient['first_admission']
        )
    
    return events, event_id

def _generate_hematologic_pathway(events, event_id, patient_id, cancer_type, current_time, start_time):
    """Generate hematologic cancer treatment pathway."""
    
    # Treatment selection based on cancer type
    if 'Lymphoma' in cancer_type:
        treatments = ['R-CHOP', 'ABVD', 'Rituximab', 'Radiation Therapy']
        treatment_weights = [0.4, 0.3, 0.2, 0.1]
    elif 'Leukemia' in cancer_type:
        treatments = ['Chemotherapy Induction', 'Consolidation', 'Maintenance', 'Stem Cell Transplant']
        treatment_weights = [0.4, 0.3, 0.2, 0.1]
    else:  # Multiple Myeloma
        treatments = ['Melphalan', 'Lenalidomide', 'Bortezomib', 'Stem Cell Transplant']
        treatment_weights = [0.3, 0.3, 0.3, 0.1]
    
    # Multiple treatment cycles
    n_treatments = np.random.randint(2, 5)
    for _ in range(n_treatments):
        treatment = np.random.choice(treatments, p=treatment_weights)
        events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'treatment',
            'event_subtype': treatment,
            'timestamp': current_time,
            'days_from_start': (current_time - start_time).days
        })
        event_id += 1
        current_time += timedelta(days=np.random.randint(21, 60))  # Treatment cycles
    
    # Complications (higher rate for hematologic cancers)
    if np.random.random() < 0.6:
        complications = ['Neutropenia', 'Thrombocytopenia', 'Infection', 'Mucositis']
        complication = np.random.choice(complications)
        events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'complication',
            'event_subtype': complication,
            'timestamp': current_time,
            'days_from_start': (current_time - start_time).days
        })
        event_id += 1
        current_time += timedelta(days=np.random.randint(7, 30))
    
    # Outcome (hematologic cancers have different outcome patterns)
    outcome_probs = [0.65, 0.25, 0.10]  # Remission, Relapse, Death
    outcome = np.random.choice(['Complete Remission', 'Relapse', 'Death'], p=outcome_probs)
    
    events.append({
        'event_id': event_id,
        'subject_id': patient_id,
        'event_type': 'outcome',
        'event_subtype': outcome,
        'timestamp': current_time,
        'days_from_start': (current_time - start_time).days
    })
    event_id += 1
    
    return events, event_id, current_time

def _generate_thyroid_pathway(events, event_id, patient_id, cancer_type, current_time, start_time):
    """Generate thyroid cancer treatment pathway."""
    
    # Staging workup (common for thyroid cancer)
    workup_events = ['Ultrasound', 'Fine Needle Aspiration', 'Thyroglobulin Level']
    for workup in np.random.choice(workup_events, size=2, replace=False):
        events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'diagnostic',
            'event_subtype': workup,
            'timestamp': current_time,
            'days_from_start': (current_time - start_time).days
        })
        event_id += 1
        current_time += timedelta(days=np.random.randint(7, 21))
    
    # Surgery (primary treatment for thyroid cancer)
    if 'Anaplastic' in cancer_type:
        surgery_types = ['Total Thyroidectomy', 'Palliative Surgery']
        surgery_weights = [0.6, 0.4]
    else:
        surgery_types = ['Lobectomy', 'Total Thyroidectomy', 'Central Neck Dissection']
        surgery_weights = [0.3, 0.6, 0.1]
    
    surgery = np.random.choice(surgery_types, p=surgery_weights)
    events.append({
        'event_id': event_id,
        'subject_id': patient_id,
        'event_type': 'surgery',
        'event_subtype': surgery,
        'timestamp': current_time,
        'days_from_start': (current_time - start_time).days
    })
    event_id += 1
    current_time += timedelta(days=np.random.randint(14, 45))
    
    # Radioactive Iodine Therapy (common adjuvant treatment)
    if np.random.random() < 0.7 and 'Anaplastic' not in cancer_type:
        events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'treatment',
            'event_subtype': 'Radioactive Iodine Therapy',
            'timestamp': current_time,
            'days_from_start': (current_time - start_time).days
        })
        event_id += 1
        current_time += timedelta(days=np.random.randint(30, 90))
    
    # Thyroid hormone replacement
    events.append({
        'event_id': event_id,
        'subject_id': patient_id,
        'event_type': 'treatment',
        'event_subtype': 'Levothyroxine',
        'timestamp': current_time,
        'days_from_start': (current_time - start_time).days
    })
    event_id += 1
    current_time += timedelta(days=np.random.randint(30, 60))
    
    # Complications (lower rate for thyroid cancer)
    if np.random.random() < 0.25:
        complications = ['Hypoparathyroidism', 'Recurrent Laryngeal Nerve Injury', 'Hypothyroidism']
        complication = np.random.choice(complications)
        events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'complication',
            'event_subtype': complication,
            'timestamp': current_time,
            'days_from_start': (current_time - start_time).days
        })
        event_id += 1
        current_time += timedelta(days=np.random.randint(7, 30))
    
    # Outcome (thyroid cancers generally have better prognosis)
    if 'Anaplastic' in cancer_type:
        outcome_probs = [0.2, 0.3, 0.5]  # Poor prognosis
    elif 'Papillary' in cancer_type or 'Follicular' in cancer_type:
        outcome_probs = [0.85, 0.12, 0.03]  # Excellent prognosis
    else:  # Medullary
        outcome_probs = [0.7, 0.2, 0.1]  # Intermediate prognosis
    
    outcome = np.random.choice(['Disease Free', 'Recurrence', 'Death'], p=outcome_probs)
    
    events.append({
        'event_id': event_id,
        'subject_id': patient_id,
        'event_type': 'outcome',
        'event_subtype': outcome,
        'timestamp': current_time,
        'days_from_start': (current_time - start_time).days
    })
    event_id += 1
    
    return events, event_id, current_time

def _generate_gi_pathway(events, event_id, patient_id, cancer_type, current_time, start_time):
    """Generate gastrointestinal cancer treatment pathway."""
    
    # Staging workup
    staging_events = ['CT Scan', 'PET Scan', 'Endoscopy', 'Biopsy']
    for staging in np.random.choice(staging_events, size=2, replace=False):
        events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'diagnostic',
            'event_subtype': staging,
            'timestamp': current_time,
            'days_from_start': (current_time - start_time).days
        })
        event_id += 1
        current_time += timedelta(days=np.random.randint(7, 21))
    
    # Treatment approach varies by cancer type and stage
    if cancer_type == 'Pancreatic Cancer':
        # Often advanced at diagnosis
        treatments = ['FOLFIRINOX', 'Gemcitabine', 'Whipple Procedure', 'Palliative Care']
        treatment_weights = [0.3, 0.3, 0.2, 0.2]
    elif cancer_type == 'Colorectal Cancer':
        treatments = ['Surgery', 'FOLFOX', 'FOLFIRI', 'Bevacizumab', 'Radiation Therapy']
        treatment_weights = [0.4, 0.2, 0.2, 0.1, 0.1]
    else:  # Other GI cancers
        treatments = ['Surgery', 'Chemotherapy', 'Radiation Therapy', 'Immunotherapy']
        treatment_weights = [0.4, 0.3, 0.2, 0.1]
    
    # Multiple treatment modalities
    n_treatments = np.random.randint(1, 4)
    for _ in range(n_treatments):
        treatment = np.random.choice(treatments, p=treatment_weights)
        events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'treatment',
            'event_subtype': treatment,
            'timestamp': current_time,
            'days_from_start': (current_time - start_time).days
        })
        event_id += 1
        current_time += timedelta(days=np.random.randint(21, 90))
    
    # Complications
    if np.random.random() < 0.4:
        complications = ['Surgical Site Infection', 'Anastomotic Leak', 'Neuropathy', 'Diarrhea']
        complication = np.random.choice(complications)
        events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'complication',
            'event_subtype': complication,
            'timestamp': current_time,
            'days_from_start': (current_time - start_time).days
        })
        event_id += 1
        current_time += timedelta(days=np.random.randint(7, 30))
    
    # Outcome (varies significantly by GI cancer type)
    if cancer_type == 'Pancreatic Cancer':
        outcome_probs = [0.3, 0.4, 0.3]  # Poor overall prognosis
    elif cancer_type == 'Colorectal Cancer':
        outcome_probs = [0.6, 0.3, 0.1]  # Better if caught early
    else:
        outcome_probs = [0.5, 0.35, 0.15]  # Intermediate
    
    outcome = np.random.choice(['No Evidence of Disease', 'Progressive Disease', 'Death'], p=outcome_probs)
    
    events.append({
        'event_id': event_id,
        'subject_id': patient_id,
        'event_type': 'outcome',
        'event_subtype': outcome,
        'timestamp': current_time,
        'days_from_start': (current_time - start_time).days
    })
    event_id += 1
    
    return events, event_id, current_time

# =============================================
# HELPER FUNCTIONS
# =============================================

def filter_data(patients_df, events_df, filters):
    """Apply filters to the dataset with expanded cancer type support."""
    
    # Filter patients
    filtered_patients = patients_df.copy()
    
    # Cancer category and type filters
    if filters['cancer_categories']:
        filtered_patients = filtered_patients[
            filtered_patients['cancer_category'].isin(filters['cancer_categories'])
        ]
    
    if filters['cancer_types']:
        filtered_patients = filtered_patients[
            filtered_patients['cancer_type'].isin(filters['cancer_types'])
        ]
    
    # Demographics filters
    if filters['gender'] != 'All':
        filtered_patients = filtered_patients[filtered_patients['gender'] == filters['gender']]
    
    filtered_patients = filtered_patients[
        (filtered_patients['age'] >= filters['age_range'][0]) &
        (filtered_patients['age'] <= filters['age_range'][1])
    ]
    
    # Staging filter
    if filters['stages']:
        filtered_patients = filtered_patients[filtered_patients['stage'].isin(filters['stages'])]
    
    # Outcomes filter
    if filters['outcomes']:
        filtered_patients = filtered_patients[filtered_patients['outcome'].isin(filters['outcomes'])]
    
    # Get patient IDs for event filtering
    patient_ids = filtered_patients['subject_id'].tolist()
    filtered_events = events_df[events_df['subject_id'].isin(patient_ids)].copy()
    
    # Treatment filter - only include patients who received specific treatments
    if filters['selected_treatments']:
        patients_with_treatments = filtered_events[
            (filtered_events['event_type'].isin(['treatment', 'surgery'])) &
            (filtered_events['event_subtype'].isin(filters['selected_treatments']))
        ]['subject_id'].unique()
        
        filtered_patients = filtered_patients[
            filtered_patients['subject_id'].isin(patients_with_treatments)
        ]
        filtered_events = filtered_events[
            filtered_events['subject_id'].isin(patients_with_treatments)
        ]
    
    # Time window filter
    filtered_events = filtered_events[
        filtered_events['days_from_start'] <= filters['time_window']
    ]
    
    # Complications filter
    if not filters['include_complications']:
        filtered_events = filtered_events[filtered_events['event_type'] != 'complication']
    
    # Pathway length filter
    pathway_lengths = filtered_events.groupby('subject_id').size()
    patients_meeting_length = pathway_lengths[
        pathway_lengths >= filters['min_pathway_length']
    ].index.tolist()
    
    filtered_patients = filtered_patients[
        filtered_patients['subject_id'].isin(patients_meeting_length)
    ]
    filtered_events = filtered_events[
        filtered_events['subject_id'].isin(patients_meeting_length)
    ]
    
    return filtered_patients, filtered_events

def create_sankey_data(events_df):
    """Create data for Sankey diagram."""
    
    # Get transitions for each patient
    transitions = []
    
    for patient_id in events_df['subject_id'].unique():
        patient_events = events_df[events_df['subject_id'] == patient_id].sort_values('timestamp')
        
        for i in range(len(patient_events) - 1):
            # Create cleaner node names
            source_type = patient_events.iloc[i]['event_type'].title()
            source_subtype = patient_events.iloc[i]['event_subtype']
            source = f"{source_type}: {source_subtype}"
            
            target_type = patient_events.iloc[i+1]['event_type'].title()
            target_subtype = patient_events.iloc[i+1]['event_subtype']
            target = f"{target_type}: {target_subtype}"
            
            transitions.append((source, target))
    
    # Count transitions
    from collections import Counter
    transition_counts = Counter(transitions)
    
    # Only include transitions with at least 3 patients for clarity
    filtered_transitions = {k: v for k, v in transition_counts.items() if v >= 3}
    
    if not filtered_transitions:
        return None
    
    # Prepare data for Sankey
    all_nodes = set()
    for source, target in filtered_transitions.keys():
        all_nodes.add(source)
        all_nodes.add(target)
    
    node_list = sorted(list(all_nodes))  # Sort for consistency
    node_indices = {node: i for i, node in enumerate(node_list)}
    
    # Assign colors based on event type
    node_colors = []
    for node in node_list:
        if node.startswith('Diagnosis'):
            node_colors.append('rgba(255, 99, 132, 0.8)')  # Red
        elif node.startswith('Treatment'):
            node_colors.append('rgba(54, 162, 235, 0.8)')  # Blue
        elif node.startswith('Complication'):
            node_colors.append('rgba(255, 206, 86, 0.8)')  # Yellow
        elif node.startswith('Outcome'):
            node_colors.append('rgba(75, 192, 192, 0.8)')  # Green
        else:
            node_colors.append('rgba(153, 102, 255, 0.8)')  # Purple
    
    sankey_data = {
        'node_labels': node_list,
        'node_colors': node_colors,
        'links': []
    }
    
    for (source, target), count in filtered_transitions.items():
        sankey_data['links'].append({
            'source': node_indices[source],
            'target': node_indices[target],
            'value': count,
            'label': f'{count} patients: {source} → {target}'
        })
    
    return sankey_data

def build_transition_graph(events_df):
    """Build NetworkX graph from events."""
    
    G = nx.DiGraph()
    transitions = {}
    
    for patient_id in events_df['subject_id'].unique():
        patient_events = events_df[events_df['subject_id'] == patient_id].sort_values('timestamp')
        
        for i in range(len(patient_events) - 1):
            # Create cleaner node names
            source_type = patient_events.iloc[i]['event_type'].title()
            source_subtype = patient_events.iloc[i]['event_subtype']
            source = f"{source_type}: {source_subtype}"
            
            target_type = patient_events.iloc[i+1]['event_type'].title()
            target_subtype = patient_events.iloc[i+1]['event_subtype']
            target = f"{target_type}: {target_subtype}"
            
            if (source, target) in transitions:
                transitions[(source, target)] += 1
            else:
                transitions[(source, target)] = 1
    
    # Add edges to graph (minimum threshold of 2 patients)
    for (source, target), weight in transitions.items():
        if weight >= 2:
            G.add_edge(source, target, weight=weight)
    
    return G, transitions

# =============================================
# SIDEBAR FILTERS
# =============================================

def create_sidebar_filters(patients_df: pd.DataFrame, events_df: pd.DataFrame):
    """Create sidebar filter controls with expanded cancer types."""
    
    st.sidebar.header("🎛️ Filters & Settings")
    
    # Cancer Category Filter (New high-level grouping)
    st.sidebar.subheader("Cancer Category")
    cancer_categories = st.sidebar.multiselect(
        "Select Cancer Categories",
        options=sorted(patients_df['cancer_category'].unique()),
        default=sorted(patients_df['cancer_category'].unique())
    )
    
    # Filter cancer types based on selected categories
    available_cancer_types = patients_df[
        patients_df['cancer_category'].isin(cancer_categories)
    ]['cancer_type'].unique()
    
    # Detailed Cancer Type Filter
    st.sidebar.subheader("Specific Cancer Types")
    cancer_types = st.sidebar.multiselect(
        "Select Specific Cancer Types",
        options=sorted(available_cancer_types),
        default=sorted(available_cancer_types)
    )
    
    # Demographics
    st.sidebar.subheader("Demographics")
    
    age_range = st.sidebar.slider(
        "Age Range",
        min_value=int(patients_df['age'].min()),
        max_value=int(patients_df['age'].max()),
        value=(int(patients_df['age'].min()), int(patients_df['age'].max()))
    )
    
    gender = st.sidebar.selectbox(
        "Gender",
        options=['All', 'M', 'F']
    )
    
    # Staging (now handles both hematologic and solid tumor staging)
    all_stages = sorted(patients_df['stage'].unique())
    stages = st.sidebar.multiselect(
        "Stage/TNM",
        options=all_stages,
        default=all_stages,
        help="Includes both hematologic stages (I-IV) and TNM staging for solid tumors"
    )
    
    # Outcomes (now includes cancer-specific outcomes)
    all_outcomes = sorted(patients_df['outcome'].dropna().unique())
    outcomes = st.sidebar.multiselect(
        "Outcome",
        options=all_outcomes,
        default=all_outcomes
    )
    
    # Treatment Filters (New section)
    st.sidebar.subheader("Treatment Filters")
    
    # Get unique treatment types from events
    treatment_events = events_df[events_df['event_type'].isin(['treatment', 'surgery'])]
    all_treatments = sorted(treatment_events['event_subtype'].unique()) if len(treatment_events) > 0 else []
    
    if all_treatments:
        selected_treatments = st.sidebar.multiselect(
            "Include Patients with Treatments",
            options=all_treatments,
            default=[],
            help="Filter patients who received specific treatments"
        )
    else:
        selected_treatments = []
    
    # Sequence Filters
    st.sidebar.subheader("Pathway Filters")
    
    min_pathway_length = st.sidebar.slider(
        "Minimum Pathway Length",
        min_value=2,
        max_value=15,
        value=3,
        help="Minimum number of events in patient pathway"
    )
    
    time_window = st.sidebar.slider(
        "Time Window (days)",
        min_value=30,
        max_value=1095,  # 3 years
        value=730,  # 2 years default
        help="Include events within this timeframe from first diagnosis"
    )
    
    # Show complications
    include_complications = st.sidebar.checkbox(
        "Include Complications in Analysis",
        value=True,
        help="Include complication events in pathway analysis"
    )
    
    return {
        'cancer_categories': cancer_categories,
        'cancer_types': cancer_types,
        'age_range': age_range,
        'gender': gender,
        'stages': stages,
        'outcomes': outcomes,
        'selected_treatments': selected_treatments,
        'min_pathway_length': min_pathway_length,
        'time_window': time_window,
        'include_complications': include_complications
    }

# =============================================
# DASHBOARD PAGES
# =============================================

def dashboard_home(patients_df, events_df):
    """Dashboard home page with overview statistics for expanded cancer types."""
    
    st.header("🏠 Dashboard Overview")
    
    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", len(patients_df))
    
    with col2:
        st.metric("Mean Age", f"{patients_df['age'].mean():.1f}")
    
    with col3:
        female_pct = (patients_df['gender'] == 'F').mean() * 100
        st.metric("% Female", f"{female_pct:.1f}%")
    
    with col4:
        # Calculate overall good outcome rate (varies by cancer type)
        good_outcomes = ['Complete Remission', 'Disease Free', 'No Evidence of Disease']
        good_outcome_rate = patients_df['outcome'].isin(good_outcomes).mean() * 100
        st.metric("Good Outcome Rate", f"{good_outcome_rate:.1f}%")
    
    # Enhanced visualizations for multiple cancer types
    col1, col2 = st.columns(2)
    
    with col1:
        # Cancer category distribution
        category_counts = patients_df['cancer_category'].value_counts()
        fig = px.pie(
            values=category_counts.values, 
            names=category_counts.index, 
            title="Distribution by Cancer Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Detailed cancer type frequencies
        cancer_counts = patients_df['cancer_type'].value_counts().head(10)  # Top 10 for readability
        fig = px.bar(
            x=cancer_counts.values, 
            y=cancer_counts.index,
            orientation='h',
            title="Top 10 Specific Cancer Types"
        )
        fig.update_layout(
            xaxis_title="Number of Patients", 
            yaxis_title="Cancer Type",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution by cancer category
        fig = px.box(
            patients_df, 
            x='cancer_category', 
            y='age',
            title="Age Distribution by Cancer Category",
            color='cancer_category'
        )
        fig.update_layout(xaxis_title="Cancer Category", yaxis_title="Age")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Outcome comparison across cancer categories
        outcome_category = pd.crosstab(patients_df['cancer_category'], patients_df['outcome'])
        # Convert to percentages for better comparison
        outcome_category_pct = outcome_category.div(outcome_category.sum(axis=1), axis=0) * 100
        
        fig = px.bar(
            outcome_category_pct, 
            title="Outcome Distribution by Cancer Category (%)",
            barmode='group'
        )
        fig.update_layout(
            xaxis_title="Cancer Category", 
            yaxis_title="Percentage of Patients"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics table
    st.subheader("📊 Cancer Category Summary")
    
    summary_stats = []
    for category in patients_df['cancer_category'].unique():
        category_data = patients_df[patients_df['cancer_category'] == category]
        
        good_outcome_rate = category_data['outcome'].isin(good_outcomes).mean() * 100
        
        summary_stats.append({
            'Cancer Category': category,
            'Number of Patients': len(category_data),
            'Mean Age': f"{category_data['age'].mean():.1f}",
            'Female %': f"{(category_data['gender'] == 'F').mean() * 100:.1f}%",
            'Good Outcome Rate': f"{good_outcome_rate:.1f}%",
            'Most Common Type': category_data['cancer_type'].mode().iloc[0] if len(category_data) > 0 else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_stats)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Treatment modalities overview
    st.subheader("🏥 Treatment Modalities Overview")
    
    # Get treatment statistics
    treatment_events = events_df[events_df['event_type'].isin(['treatment', 'surgery'])]
    if len(treatment_events) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Most common treatments overall
            treatment_counts = treatment_events['event_subtype'].value_counts().head(10)
            fig = px.bar(
                x=treatment_counts.values,
                y=treatment_counts.index,
                orientation='h',
                title="Most Common Treatments"
            )
            fig.update_layout(xaxis_title="Number of Patients", yaxis_title="Treatment")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Treatment types by cancer category
            treatment_events_with_category = treatment_events.merge(
                patients_df[['subject_id', 'cancer_category']], 
                on='subject_id'
            )
            
            treatment_by_category = treatment_events_with_category.groupby(
                ['cancer_category', 'event_type']
            ).size().reset_index(name='count')
            
            fig = px.bar(
                treatment_by_category,
                x='cancer_category',
                y='count',
                color='event_type',
                title="Treatment Modalities by Cancer Category",
                barmode='group'
            )
            fig.update_layout(xaxis_title="Cancer Category", yaxis_title="Number of Events")
            st.plotly_chart(fig, use_container_width=True)

def cohort_explorer(patients_df, events_df):
    """Cohort explorer with patient table and search for expanded cancer types."""
    
    st.header("👥 Cohort Explorer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search functionality
        search_id = st.text_input("Search by Patient ID", placeholder="e.g., P0001")
    
    with col2:
        # Quick filter by cancer category
        selected_category = st.selectbox(
            "Quick Filter by Category",
            options=['All'] + sorted(patients_df['cancer_category'].unique())
        )
    
    # Display patients table
    display_df = patients_df.copy()
    
    if search_id:
        display_df = display_df[display_df['subject_id'].str.contains(search_id, case=False)]
    
    if selected_category != 'All':
        display_df = display_df[display_df['cancer_category'] == selected_category]
    
    # Format display
    display_df['first_admission'] = pd.to_datetime(display_df['first_admission']).dt.strftime('%Y-%m-%d')
    
    # Show enhanced table with cancer categories
    st.dataframe(
        display_df[['subject_id', 'age', 'gender', 'cancer_category', 'cancer_type', 'stage', 
                   'first_admission', 'outcome', 'outcome_days']],
        use_container_width=True,
        column_config={
            "subject_id": "Patient ID",
            "cancer_category": "Category",
            "cancer_type": "Specific Type",
            "stage": "Stage/TNM",
            "first_admission": "First Admission",
            "outcome_days": "Time to Outcome (days)"
        }
    )
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Patients in View", len(display_df))
    with col2:
        st.metric("Cancer Categories", display_df['cancer_category'].nunique())
    with col3:
        st.metric("Specific Cancer Types", display_df['cancer_type'].nunique())
    with col4:
        if len(display_df) > 0:
            good_outcomes = ['Complete Remission', 'Disease Free', 'No Evidence of Disease']
            good_rate = display_df['outcome'].isin(good_outcomes).mean() * 100
            st.metric("Good Outcome Rate", f"{good_rate:.1f}%")
    
    # Export functionality
    csv_data = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Current View as CSV",
        data=csv_data,
        file_name="cohort_data.csv",
        mime="text/csv"
    )

def pathway_flow_visualizer(patients_df, events_df):
    """Visualize aggregate pathways using Sankey and network graphs."""
    
    st.header("🌊 Pathway Flow Visualizer")
    
    # Sankey Diagram
    st.subheader("Treatment Flow (Sankey Diagram)")
    
    sankey_data = create_sankey_data(events_df)
    
    if sankey_data and sankey_data['links']:
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="black", width=0.5),
                label=sankey_data['node_labels'],
                color=sankey_data['node_colors']
            ),
            link=dict(
                source=[link['source'] for link in sankey_data['links']],
                target=[link['target'] for link in sankey_data['links']],
                value=[link['value'] for link in sankey_data['links']],
                label=[link['label'] for link in sankey_data['links']],
                color='rgba(0,0,0,0.2)'
            )
        )])
        
        fig.update_layout(
            title_text="Patient Flow Through Treatment Pathways", 
            font_size=12,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        total_patients = len(events_df['subject_id'].unique())
        total_transitions = len(sankey_data['links'])
        
        st.info(f"📊 Showing {total_transitions} common transitions for {total_patients} patients. "
               f"Only transitions with ≥3 patients are displayed for clarity.")
        
    else:
        st.warning("⚠️ No common treatment pathways found with current filters. "
                  "Try expanding your filter criteria or reducing the minimum pathway length.")
    
    # Enhanced transition frequency table
    st.subheader("Event Transition Frequencies")
    
    G, transitions = build_transition_graph(events_df)
    
    if transitions:
        # Create a better formatted transition table
        transition_data = []
        for (source, target), count in sorted(transitions.items(), key=lambda x: x[1], reverse=True):
            # Clean up the source and target names
            source_clean = source.replace(':', ': ').replace('_', ' ').title()
            target_clean = target.replace(':', ': ').replace('_', ' ').title()
            
            # Calculate percentage
            total_patients = len(events_df['subject_id'].unique())
            percentage = (count / total_patients) * 100
            
            transition_data.append({
                'From': source_clean,
                'To': target_clean,
                'Patient Count': count,
                'Percentage of Cohort': f"{percentage:.1f}%"
            })
        
        transition_df = pd.DataFrame(transition_data)
        
        # Show top 15 transitions
        st.dataframe(
            transition_df.head(15), 
            use_container_width=True,
            hide_index=True
        )
        
        # Add summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transitions", len(transition_df))
        with col2:
            st.metric("Most Common Transition", f"{transition_df.iloc[0]['Patient Count']} patients")
        with col3:
            avg_transitions = len(transitions) / len(events_df['subject_id'].unique())
            st.metric("Avg Transitions/Patient", f"{avg_transitions:.1f}")
            
    else:
        st.info("No transitions found with current filters.")
    
    # Optional: Add pathway complexity analysis
    st.subheader("Pathway Complexity Analysis")
    
    # Calculate pathway lengths for each patient
    pathway_lengths = []
    for patient_id in events_df['subject_id'].unique():
        patient_events = events_df[events_df['subject_id'] == patient_id]
        pathway_lengths.append(len(patient_events))
    
    if pathway_lengths:
        complexity_df = pd.DataFrame({
            'Pathway Length': pathway_lengths
        })
        
        fig = px.histogram(
            complexity_df, 
            x='Pathway Length',
            title='Distribution of Pathway Complexity (Number of Events per Patient)',
            nbins=min(15, max(pathway_lengths) - min(pathway_lengths) + 1)
        )
        fig.update_layout(
            xaxis_title="Number of Events per Patient",
            yaxis_title="Number of Patients"
        )
        st.plotly_chart(fig, use_container_width=True)

def patient_timeline_viewer(patients_df, events_df):
    """View individual patient timelines."""
    
    st.header("📅 Patient Timeline Viewer")
    
    # Patient selection
    patient_ids = sorted(patients_df['subject_id'].unique())
    selected_patient = st.selectbox("Select Patient ID", patient_ids)
    
    if selected_patient:
        # Get patient events
        patient_events = events_df[events_df['subject_id'] == selected_patient].sort_values('timestamp')
        patient_info = patients_df[patients_df['subject_id'] == selected_patient].iloc[0]
        
        # Patient summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Age", patient_info['age'])
        with col2:
            st.metric("Gender", patient_info['gender'])
        with col3:
            st.metric("Cancer Type", patient_info['cancer_type'])
        with col4:
            st.metric("Outcome", patient_info['outcome'])
        
        # Timeline visualization
        fig = go.Figure()
        
        colors = {'diagnosis': 'red', 'treatment': 'blue', 'complication': 'orange', 'outcome': 'green'}
        
        for i, (_, event) in enumerate(patient_events.iterrows()):
            fig.add_trace(go.Scatter(
                x=[event['days_from_start']],
                y=[event['event_type']],
                mode='markers',
                marker=dict(
                    size=15,
                    color=colors.get(event['event_type'], 'gray'),
                    symbol='circle'
                ),
                name=event['event_subtype'],
                text=f"Day {event['days_from_start']}: {event['event_subtype']}",
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"Timeline for Patient {selected_patient}",
            xaxis_title="Days from First Event",
            yaxis_title="Event Type",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Event details table
        st.subheader("Event Details")
        display_events = patient_events.copy()
        display_events['timestamp'] = display_events['timestamp'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(
            display_events[['event_type', 'event_subtype', 'timestamp', 'days_from_start']],
            use_container_width=True
        )

def digital_twin_matcher(patients_df, events_df):
    """Find similar patients (digital twins) based on input criteria."""
    
    st.header("👥 Digital Twin Matcher")
    st.markdown("Find patients similar to your current patient for prognosis and treatment guidance.")
    
    # Input patient characteristics
    st.subheader("📝 Enter Patient Characteristics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_age = st.number_input("Patient Age", min_value=18, max_value=90, value=39)
        input_gender = st.selectbox("Gender", options=['M', 'F'], index=0)
    
    with col2:
        available_cancers = sorted(patients_df['cancer_type'].unique())
        default_thyroid = 'Papillary Thyroid Carcinoma' if 'Papillary Thyroid Carcinoma' in available_cancers else available_cancers[0]
        input_cancer = st.selectbox("Cancer Type", options=available_cancers, 
                                   index=available_cancers.index(default_thyroid))
        
        available_stages = sorted(patients_df['stage'].unique())
        input_stage = st.selectbox("Stage", options=available_stages)
    
    with col3:
        age_tolerance = st.slider("Age Tolerance (±years)", min_value=1, max_value=15, value=5)
        min_matches = st.slider("Minimum Matches Required", min_value=5, max_value=50, value=10)
    
    # Find similar patients
    if st.button("🔍 Find Similar Patients", type="primary"):
        
        # Filter for similar patients
        similar_patients = patients_df[
            (patients_df['cancer_type'] == input_cancer) &
            (patients_df['gender'] == input_gender) &
            (patients_df['stage'] == input_stage) &
            (patients_df['age'] >= input_age - age_tolerance) &
            (patients_df['age'] <= input_age + age_tolerance)
        ].copy()
        
        if len(similar_patients) >= min_matches:
            st.success(f"✅ Found {len(similar_patients)} similar patients!")
            
            # Calculate similarity scores
            similar_patients['age_diff'] = abs(similar_patients['age'] - input_age)
            similar_patients = similar_patients.sort_values('age_diff')
            
            # Display top matches
            st.subheader("🎯 Top Digital Twin Matches")
            
            display_cols = ['subject_id', 'age', 'gender', 'cancer_type', 'stage', 'outcome', 'outcome_days']
            top_matches = similar_patients.head(10)[display_cols]
            
            st.dataframe(top_matches, use_container_width=True, hide_index=True)
            
            # Outcome analysis for similar patients
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Outcome Statistics for Similar Patients")
                
                outcome_stats = similar_patients['outcome'].value_counts()
                total_patients = len(similar_patients)
                
                for outcome, count in outcome_stats.items():
                    percentage = (count / total_patients) * 100
                    st.metric(f"{outcome}", f"{count} patients ({percentage:.1f}%)")
                
                # Time to outcome statistics
                st.metric("Median Time to Outcome", f"{similar_patients['outcome_days'].median():.0f} days")
                st.metric("Mean Age", f"{similar_patients['age'].mean():.1f} years")
            
            with col2:
                # Outcome distribution visualization
                fig = px.pie(
                    values=outcome_stats.values,
                    names=outcome_stats.index,
                    title="Outcome Distribution in Similar Patients"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Treatment pathway analysis for similar patients
            st.subheader("🏥 Common Treatment Pathways")
            
            similar_patient_ids = similar_patients['subject_id'].tolist()
            similar_events = events_df[events_df['subject_id'].isin(similar_patient_ids)]
            
            # Most common treatments
            treatment_events = similar_events[similar_events['event_type'].isin(['treatment', 'surgery'])]
            if len(treatment_events) > 0:
                treatment_counts = treatment_events['event_subtype'].value_counts().head(8)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Most Common Treatments:**")
                    for treatment, count in treatment_counts.items():
                        percentage = (count / len(similar_patients)) * 100
                        st.write(f"• {treatment}: {count} patients ({percentage:.1f}%)")
                
                with col2:
                    fig = px.bar(
                        x=treatment_counts.values,
                        y=treatment_counts.index,
                        orientation='h',
                        title="Treatment Frequency"
                    )
                    fig.update_layout(xaxis_title="Number of Patients")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Timeline analysis
            st.subheader("⏱️ Treatment Timeline Analysis")
            
            # Calculate typical timelines
            timeline_stats = []
            for patient_id in similar_patient_ids[:20]:  # Analyze top 20 matches
                patient_events = similar_events[similar_events['subject_id'] == patient_id].sort_values('timestamp')
                
                if len(patient_events) > 1:
                    first_treatment = patient_events[
                        patient_events['event_type'].isin(['treatment', 'surgery'])
                    ].head(1)
                    
                    if len(first_treatment) > 0:
                        days_to_treatment = first_treatment['days_from_start'].iloc[0]
                        timeline_stats.append(days_to_treatment)
            
            if timeline_stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Median Days to Treatment", f"{np.median(timeline_stats):.0f}")
                with col2:
                    st.metric("Mean Days to Treatment", f"{np.mean(timeline_stats):.0f}")
                with col3:
                    st.metric("Range", f"{min(timeline_stats)}-{max(timeline_stats)} days")
            
            # Clinical recommendations
            st.subheader("💡 Clinical Insights")
            
            good_outcomes = ['Complete Remission', 'Disease Free', 'No Evidence of Disease']
            good_outcome_rate = similar_patients['outcome'].isin(good_outcomes).mean() * 100
            
            if good_outcome_rate >= 80:
                st.success(f"🎉 Excellent prognosis: {good_outcome_rate:.1f}% of similar patients achieved good outcomes")
            elif good_outcome_rate >= 60:
                st.info(f"✅ Good prognosis: {good_outcome_rate:.1f}% of similar patients achieved good outcomes")
            else:
                st.warning(f"⚠️ Guarded prognosis: {good_outcome_rate:.1f}% of similar patients achieved good outcomes")
            
            # Export similar patients
            csv_data = similar_patients.to_csv(index=False)
            st.download_button(
                label="📥 Download Similar Patients Data",
                data=csv_data,
                file_name=f"similar_patients_{input_cancer.replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning(f"⚠️ Only found {len(similar_patients)} similar patients. Consider:")
            st.write("• Increasing age tolerance")
            st.write("• Reducing minimum matches required")
            st.write("• Expanding stage criteria")
            
            if len(similar_patients) > 0:
                st.write(f"\n**{len(similar_patients)} patients found with current criteria:**")
                st.dataframe(similar_patients[['subject_id', 'age', 'outcome']].head(), 
                           use_container_width=True, hide_index=True)
    
    # Example use case
    st.subheader("💡 Example: 39-year-old Male with Thyroid Cancer")
    st.write("""
    **How to use this tool:**
    1. Enter patient age: 39
    2. Select gender: Male
    3. Choose cancer type: Papillary Thyroid Carcinoma
    4. Select appropriate stage
    5. Click 'Find Similar Patients'
    
    **What you'll get:**
    - List of patients with similar characteristics
    - Outcome statistics and prognosis
    - Common treatment pathways
    - Timeline to treatment initiation
    - Clinical insights based on historical data
    """)

def outcome_comparator(patients_df, events_df):
    """Compare outcomes between different pathways or groups."""
    
    st.header("⚖️ Outcome Comparator")
    
    # Group selection
    st.subheader("Select Groups to Compare")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Group A**")
        group_a_cancer = st.selectbox("Cancer Type A", patients_df['cancer_type'].unique())
        group_a_stage = st.selectbox("Stage A", patients_df['stage'].unique())
    
    with col2:
        st.write("**Group B**")  
        group_b_cancer = st.selectbox("Cancer Type B", patients_df['cancer_type'].unique(), index=1)
        group_b_stage = st.selectbox("Stage B", patients_df['stage'].unique(), index=1)
    
    # Filter groups
    group_a = patients_df[
        (patients_df['cancer_type'] == group_a_cancer) & 
        (patients_df['stage'] == group_a_stage)
    ]
    
    group_b = patients_df[
        (patients_df['cancer_type'] == group_b_cancer) & 
        (patients_df['stage'] == group_b_stage)
    ]
    
    if len(group_a) > 0 and len(group_b) > 0:
        # Comparison results
        st.subheader("Comparison Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"Group A Patients", len(group_a))
            remission_a = (group_a['outcome'] == 'Remission').mean() * 100
            st.metric(f"Group A Remission Rate", f"{remission_a:.1f}%")
        
        with col2:
            st.metric(f"Group B Patients", len(group_b))
            remission_b = (group_b['outcome'] == 'Remission').mean() * 100
            st.metric(f"Group B Remission Rate", f"{remission_b:.1f}%")
        
        with col3:
            # Statistical test
            try:
                contingency_table = pd.crosstab(
                    pd.concat([group_a['outcome'], group_b['outcome']]),
                    ['Group A'] * len(group_a) + ['Group B'] * len(group_b)
                )
                
                if contingency_table.shape == (2, 2):
                    _, p_value = fisher_exact(contingency_table)
                    st.metric("Fisher's Exact p-value", f"{p_value:.4f}")
                else:
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    st.metric("Chi-square p-value", f"{p_value:.4f}")
            except:
                st.metric("Statistical Test", "N/A")
        
        # Visualization
        comparison_data = pd.DataFrame({
            'Group': ['A'] * len(group_a) + ['B'] * len(group_b),
            'Outcome': list(group_a['outcome']) + list(group_b['outcome']),
            'Outcome_Days': list(group_a['outcome_days']) + list(group_b['outcome_days'])
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Outcome comparison
            fig = px.histogram(comparison_data, x='Outcome', color='Group', 
                             barmode='group', title="Outcome Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time to outcome
            fig = px.box(comparison_data, x='Group', y='Outcome_Days',
                        title="Time to Outcome (Days)")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("One or both groups have no patients. Please adjust your selection.")

def survival_analysis(patients_df, events_df):
    """Perform survival analysis with Kaplan-Meier curves."""
    
    st.header("📈 Survival Analysis")
    
    # Stratification options
    strat_var = st.selectbox(
        "Stratify by:",
        options=['cancer_type', 'stage', 'gender']
    )
    
    # Prepare survival data
    survival_data = patients_df.copy()
    
    # Define event (1 for death, 0 for censored)
    survival_data['event'] = (survival_data['outcome'] == 'Death').astype(int)
    survival_data['duration'] = survival_data['outcome_days']
    
    # Kaplan-Meier analysis
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, group in enumerate(survival_data[strat_var].unique()):
        group_data = survival_data[survival_data[strat_var] == group]
        
        if len(group_data) > 5:  # Minimum group size
            kmf = KaplanMeierFitter()
            kmf.fit(group_data['duration'], group_data['event'], label=str(group))
            
            # Add survival curve
            fig.add_trace(go.Scatter(
                x=kmf.timeline,
                y=kmf.survival_function_[str(group)],
                mode='lines',
                name=f'{strat_var}: {group}',
                line=dict(color=colors[i % len(colors)])
            ))
    
    fig.update_layout(
        title=f"Kaplan-Meier Survival Curves by {strat_var.title()}",
        xaxis_title="Days",
        yaxis_title="Survival Probability",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("Survival Summary")
    
    summary_data = []
    for group in survival_data[strat_var].unique():
        group_data = survival_data[survival_data[strat_var] == group]
        
        if len(group_data) > 0:
            summary_data.append({
                'Group': group,
                'N': len(group_data),
                'Events': group_data['event'].sum(),
                'Median Survival': group_data['duration'].median(),
                'Event Rate': f"{group_data['event'].mean():.1%}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

# =============================================
# MAIN APPLICATION
# =============================================

def main():
    """Main application function."""
    
    # Title and description
    st.title("🏥 Comprehensive Oncology Pathway Mapping Engine")
    st.markdown("**Analyzing Clinical Pathways for Hematologic, Thyroid, and Gastrointestinal Cancers**")
    st.markdown("*Digital twin matching • Treatment pathway analysis • Outcome prediction*")
    
    # Load data
    with st.spinner("Loading comprehensive oncology dataset..."):
        patients_df, events_df = generate_sample_data()
    
    # Show dataset info
    with st.expander("📋 Dataset Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Hematologic Malignancies:**")
            st.write("• Hodgkin Lymphoma")
            st.write("• Non-Hodgkin Lymphoma") 
            st.write("• Multiple Myeloma")
            st.write("• Acute & Chronic Leukemias")
        
        with col2:
            st.write("**Thyroid Cancers:**")
            st.write("• Papillary Thyroid Carcinoma")
            st.write("• Follicular Thyroid Carcinoma")
            st.write("• Medullary Thyroid Carcinoma")
            st.write("• Anaplastic Thyroid Carcinoma")
        
        with col3:
            st.write("**Gastrointestinal Cancers:**")
            st.write("• Colorectal Cancer")
            st.write("• Gastric Cancer")
            st.write("• Pancreatic Cancer")
            st.write("• Hepatocellular Carcinoma")
            st.write("• Esophageal Cancer")
        
        st.info("💡 **New Feature**: Use the Digital Twin Matcher to find patients similar to your current case!")
    
    # Create sidebar filters
    filters = create_sidebar_filters(patients_df, events_df)
    
    # Apply filters
    filtered_patients, filtered_events = filter_data(patients_df, events_df, filters)
    
    # Navigation
    st.sidebar.header("🧭 Navigation")
    selected_page = st.sidebar.radio(
        "Select Page:",
        options=[
            "Dashboard Home",
            "Cohort Explorer", 
            "Pathway Flow",
            "Patient Timeline",
            "Digital Twin Matcher",
            "Outcome Comparison",
            "Survival Analysis"
        ]
    )
    
    # Display current cohort size and composition
    cohort_info = f"""
    **Current Cohort:** {len(filtered_patients)} patients  
    **Cancer Categories:** {', '.join(filtered_patients['cancer_category'].unique())}  
    **Date Range:** {len(filtered_patients)} patients from synthetic dataset
    """
    st.sidebar.info(cohort_info)
    
    # Route to selected page
    if selected_page == "Dashboard Home":
        dashboard_home(filtered_patients, filtered_events)
    elif selected_page == "Cohort Explorer":
        cohort_explorer(filtered_patients, filtered_events)
    elif selected_page == "Pathway Flow":
        pathway_flow_visualizer(filtered_patients, filtered_events)
    elif selected_page == "Patient Timeline":
        patient_timeline_viewer(filtered_patients, filtered_events)
    elif selected_page == "Digital Twin Matcher":
        digital_twin_matcher(filtered_patients, filtered_events)
    elif selected_page == "Outcome Comparison":
        outcome_comparator(filtered_patients, filtered_events)
    elif selected_page == "Survival Analysis":
        survival_analysis(filtered_patients, filtered_events)

if __name__ == "__main__":
    main()