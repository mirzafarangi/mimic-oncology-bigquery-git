# Real MIMIC-IV Oncology Pathway Mapping Dashboard
# Complete working Streamlit application with real data

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
import os
import logging

# Import real data extraction modules
from mimic_client import test_connection
from oncology_extractor import extract_oncology_cohort

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="MIMIC-IV Oncology Pathway Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# REAL DATA LOADING FROM MIMIC-IV - FIXED
# =============================================

def load_mimic_oncology_data_no_cache(project_id: str, limit: int = 100):
    """Load real oncology data from MIMIC-IV without caching conflicts."""
    
    logger.info(f"🔄 Loading MIMIC-IV oncology data (limit: {limit})")
    
    if not project_id or project_id.strip() == "":
        raise ValueError("Project ID is required")
    
    # Extract real oncology cohort directly
    try:
        patients_df, events_df, summary = extract_oncology_cohort(project_id, limit)
        
        if len(patients_df) == 0:
            raise ValueError("No oncology patients found in MIMIC-IV")
        
        # Add outcome information from events if not already present
        if 'outcome' not in patients_df.columns and len(events_df) > 0:
            outcomes = events_df[events_df['event_type'] == 'outcome'][['subject_id', 'event_subtype', 'days_from_start']]
            if len(outcomes) > 0:
                outcomes.columns = ['subject_id', 'outcome', 'outcome_days']
                patients_df = patients_df.merge(outcomes, on='subject_id', how='left')
        
        # Fill missing outcomes
        patients_df['outcome'] = patients_df['outcome'].fillna('Unknown')
        patients_df['outcome_days'] = patients_df['outcome_days'].fillna(365)
        
        logger.info(f"✅ Successfully loaded {len(patients_df)} patients with {len(events_df)} events")
        return patients_df, events_df, summary
        
    except Exception as e:
        logger.error(f"Error loading MIMIC-IV data: {str(e)}")
        raise e

# =============================================
# HELPER FUNCTIONS (OPTIMIZED)
# =============================================

def filter_data(patients_df, events_df, filters):
    """Apply filters to the dataset."""
    
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
    
    # Treatment filter
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
    if len(filtered_events) > 0:
        filtered_events = filtered_events[
            filtered_events['days_from_start'] <= filters['time_window']
        ]
    
    # Complications filter
    if not filters['include_complications'] and len(filtered_events) > 0:
        filtered_events = filtered_events[filtered_events['event_type'] != 'complication']
    
    # Pathway length filter
    if len(filtered_events) > 0:
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
    
    if len(events_df) == 0:
        return None
    
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
    
    if not transitions:
        return None
    
    # Count transitions
    from collections import Counter
    transition_counts = Counter(transitions)
    
    # Only include transitions with at least 2 patients for clarity
    filtered_transitions = {k: v for k, v in transition_counts.items() if v >= 2}
    
    if not filtered_transitions:
        return None
    
    # Prepare data for Sankey
    all_nodes = set()
    for source, target in filtered_transitions.keys():
        all_nodes.add(source)
        all_nodes.add(target)
    
    node_list = sorted(list(all_nodes))
    node_indices = {node: i for i, node in enumerate(node_list)}
    
    # Assign colors based on event type
    node_colors = []
    for node in node_list:
        if node.startswith('Diagnosis'):
            node_colors.append('rgba(255, 99, 132, 0.8)')
        elif node.startswith('Treatment'):
            node_colors.append('rgba(54, 162, 235, 0.8)')
        elif node.startswith('Complication'):
            node_colors.append('rgba(255, 206, 86, 0.8)')
        elif node.startswith('Outcome'):
            node_colors.append('rgba(75, 192, 192, 0.8)')
        else:
            node_colors.append('rgba(153, 102, 255, 0.8)')
    
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
    
    if len(events_df) == 0:
        return G, transitions
    
    for patient_id in events_df['subject_id'].unique():
        patient_events = events_df[events_df['subject_id'] == patient_id].sort_values('timestamp')
        
        for i in range(len(patient_events) - 1):
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
# SIDEBAR FILTERS (SAFE)
# =============================================

def create_sidebar_filters(patients_df: pd.DataFrame, events_df: pd.DataFrame):
    """Create sidebar filter controls."""
    
    if len(patients_df) == 0:
        return {
            'cancer_categories': [],
            'cancer_types': [],
            'age_range': (18, 90),
            'gender': 'All',
            'stages': [],
            'outcomes': [],
            'selected_treatments': [],
            'min_pathway_length': 2,
            'time_window': 365,
            'include_complications': True
        }
    
    st.sidebar.header("🎛️ Filters & Settings")
    
    # Cancer Category Filter
    st.sidebar.subheader("Cancer Category")
    cancer_categories = st.sidebar.multiselect(
        "Select Cancer Categories",
        options=sorted(patients_df['cancer_category'].unique()),
        default=sorted(patients_df['cancer_category'].unique())
    )
    
    # Filter cancer types based on selected categories
    available_cancer_types = patients_df[
        patients_df['cancer_category'].isin(cancer_categories)
    ]['cancer_type'].unique() if cancer_categories else patients_df['cancer_type'].unique()
    
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
    
    # Staging
    all_stages = sorted(patients_df['stage'].unique())
    stages = st.sidebar.multiselect(
        "Stage/TNM",
        options=all_stages,
        default=all_stages,
        help="Includes both hematologic stages and TNM staging for solid tumors"
    )
    
    # Outcomes
    all_outcomes = sorted(patients_df['outcome'].dropna().unique())
    outcomes = st.sidebar.multiselect(
        "Outcome",
        options=all_outcomes,
        default=all_outcomes
    )
    
    # Treatment Filters
    st.sidebar.subheader("Treatment Filters")
    
    treatment_events = events_df[events_df['event_type'].isin(['treatment', 'surgery'])] if len(events_df) > 0 else pd.DataFrame()
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
        max_value=10,
        value=2,
        help="Minimum number of events in patient pathway"
    )
    
    time_window = st.sidebar.slider(
        "Time Window (days)",
        min_value=30,
        max_value=1095,
        value=365,
        help="Include events within this timeframe from first diagnosis"
    )
    
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
# DASHBOARD PAGES (SAFE)
# =============================================

def dashboard_home(patients_df, events_df):
    """Dashboard home page with overview statistics."""
    
    st.header("🏠 Dashboard Overview - Real MIMIC-IV Data")
    
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
        good_outcomes = ['Complete Remission', 'Disease Free', 'No Evidence of Disease', 'Cure', 'Remission']
        good_outcome_rate = patients_df['outcome'].isin(good_outcomes).mean() * 100
        st.metric("Good Outcome Rate", f"{good_outcome_rate:.1f}%")
    
    # Enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Cancer category distribution
        category_counts = patients_df['cancer_category'].value_counts()
        fig = px.pie(
            values=category_counts.values, 
            names=category_counts.index, 
            title="Distribution by Cancer Category (Real MIMIC-IV Data)",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Detailed cancer type frequencies (top 10)
        cancer_counts = patients_df['cancer_type'].value_counts().head(10)
        fig = px.bar(
            x=cancer_counts.values, 
            y=cancer_counts.index,
            orientation='h',
            title="Top 10 Specific Cancer Types (Real Data)"
        )
        fig.update_layout(
            xaxis_title="Number of Patients", 
            yaxis_title="Cancer Type",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics table
    st.subheader("📊 MIMIC-IV Cancer Category Summary")
    
    summary_stats = []
    good_outcomes = ['Complete Remission', 'Disease Free', 'No Evidence of Disease', 'Cure', 'Remission']
    
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

def cohort_explorer(patients_df, events_df):
    """Cohort explorer with patient table and search."""
    
    st.header("👥 MIMIC-IV Cohort Explorer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_id = st.text_input("Search by Patient ID", placeholder="e.g., P0123456")
    
    with col2:
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
    if 'first_admission' in display_df.columns:
        display_df['first_admission'] = pd.to_datetime(display_df['first_admission'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # Safe column selection
    available_cols = ['subject_id', 'age', 'gender', 'cancer_category', 'cancer_type', 'stage', 'outcome', 'outcome_days']
    display_cols = [col for col in available_cols if col in display_df.columns]
    
    if 'first_admission' in display_df.columns:
        display_cols.append('first_admission')
    
    st.dataframe(
        display_df[display_cols],
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
            good_outcomes = ['Complete Remission', 'Disease Free', 'No Evidence of Disease', 'Cure', 'Remission']
            good_rate = display_df['outcome'].isin(good_outcomes).mean() * 100
            st.metric("Good Outcome Rate", f"{good_rate:.1f}%")
    
    # Export functionality
    csv_data = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Current View as CSV",
        data=csv_data,
        file_name="mimic_oncology_cohort.csv",
        mime="text/csv"
    )

def pathway_flow_visualizer(patients_df, events_df):
    """Visualize aggregate pathways using Sankey and network graphs."""
    
    st.header("🌊 Real Pathway Flow Visualizer")
    
    if len(events_df) == 0:
        st.warning("⚠️ No clinical events found. This may be normal for the current data extraction.")
        return
    
    # Sankey Diagram
    st.subheader("Treatment Flow (Sankey Diagram) - Real MIMIC-IV Data")
    
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
            title_text="Real Patient Flow Through Treatment Pathways (MIMIC-IV)", 
            font_size=12,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        total_patients = len(events_df['subject_id'].unique())
        total_transitions = len(sankey_data['links'])
        
        st.info(f"📊 Showing {total_transitions} common transitions for {total_patients} real MIMIC-IV patients. "
               f"Only transitions with ≥2 patients are displayed for clarity.")
        
    else:
        st.info("ℹ️ No common treatment pathways found. This is normal for small datasets or filtered views.")
    
    # Transition frequency table
    st.subheader("Real Event Transition Frequencies")
    
    G, transitions = build_transition_graph(events_df)
    
    if transitions:
        transition_data = []
        for (source, target), count in sorted(transitions.items(), key=lambda x: x[1], reverse=True):
            source_clean = source.replace(':', ': ').replace('_', ' ').title()
            target_clean = target.replace(':', ': ').replace('_', ' ').title()
            
            total_patients = len(events_df['subject_id'].unique())
            percentage = (count / total_patients) * 100
            
            transition_data.append({
                'From': source_clean,
                'To': target_clean,
                'Patient Count': count,
                'Percentage of Cohort': f"{percentage:.1f}%"
            })
        
        transition_df = pd.DataFrame(transition_data)
        st.dataframe(transition_df.head(15), use_container_width=True, hide_index=True)
        
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

def patient_timeline_viewer(patients_df, events_df):
    """View individual patient timelines."""
    
    st.header("📅 Real Patient Timeline Viewer")
    
    patient_ids = sorted(patients_df['subject_id'].unique())
    selected_patient = st.selectbox("Select Patient ID", patient_ids)
    
    if selected_patient:
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
        if len(patient_events) > 0:
            fig = go.Figure()
            
            colors = {'diagnosis': 'red', 'treatment': 'blue', 'complication': 'orange', 'outcome': 'green', 'surgery': 'purple', 'diagnostic': 'gray'}
            
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
                title=f"Real Timeline for MIMIC-IV Patient {selected_patient}",
                xaxis_title="Days from First Event",
                yaxis_title="Event Type",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Event details table
            st.subheader("Real Event Details")
            display_events = patient_events.copy()
            display_events['timestamp'] = display_events['timestamp'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                display_events[['event_type', 'event_subtype', 'timestamp', 'days_from_start']],
                use_container_width=True
            )
        else:
            st.info("No clinical events recorded for this patient.")

def digital_twin_matcher(patients_df, events_df):
    """Find similar patients (digital twins) based on input criteria using real data."""
    
    st.header("👥 MIMIC-IV Digital Twin Matcher")
    st.markdown("Find real patients similar to your current patient using actual MIMIC-IV data.")
    
    # Input patient characteristics
    st.subheader("📝 Enter Patient Characteristics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_age = st.number_input("Patient Age", min_value=18, max_value=90, value=65)
        input_gender = st.selectbox("Gender", options=['M', 'F'], index=0)
    
    with col2:
        available_cancers = sorted(patients_df['cancer_type'].unique())
        input_cancer = st.selectbox("Cancer Type", options=available_cancers)
        
        available_stages = sorted(patients_df['stage'].unique())
        input_stage = st.selectbox("Stage", options=available_stages)
    
    with col3:
        age_tolerance = st.slider("Age Tolerance (±years)", min_value=1, max_value=15, value=10)
        min_matches = st.slider("Minimum Matches Required", min_value=1, max_value=10, value=3)
    
    # Find similar patients
    if st.button("🔍 Find Similar MIMIC-IV Patients", type="primary"):
        
        similar_patients = patients_df[
            (patients_df['cancer_type'] == input_cancer) &
            (patients_df['gender'] == input_gender) &
            (patients_df['stage'] == input_stage) &
            (patients_df['age'] >= input_age - age_tolerance) &
            (patients_df['age'] <= input_age + age_tolerance)
        ].copy()
        
        if len(similar_patients) >= min_matches:
            st.success(f"✅ Found {len(similar_patients)} similar real MIMIC-IV patients!")
            
            similar_patients['age_diff'] = abs(similar_patients['age'] - input_age)
            similar_patients = similar_patients.sort_values('age_diff')
            
            # Display top matches
            st.subheader("🎯 Top Real Digital Twin Matches")
            
            display_cols = ['subject_id', 'age', 'gender', 'cancer_type', 'stage', 'outcome', 'outcome_days']
            display_cols = [col for col in display_cols if col in similar_patients.columns]
            top_matches = similar_patients.head(10)[display_cols]
            
            st.dataframe(top_matches, use_container_width=True, hide_index=True)
            
            # Outcome analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Real Outcome Statistics")
                
                outcome_stats = similar_patients['outcome'].value_counts()
                total_patients = len(similar_patients)
                
                for outcome, count in outcome_stats.items():
                    percentage = (count / total_patients) * 100
                    st.metric(f"{outcome}", f"{count} patients ({percentage:.1f}%)")
                
                st.metric("Median Time to Outcome", f"{similar_patients['outcome_days'].median():.0f} days")
                st.metric("Mean Age", f"{similar_patients['age'].mean():.1f} years")
            
            with col2:
                fig = px.pie(
                    values=outcome_stats.values,
                    names=outcome_stats.index,
                    title="Real Outcome Distribution in Similar Patients"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Export similar patients
            csv_data = similar_patients.to_csv(index=False)
            st.download_button(
                label="📥 Download Similar MIMIC-IV Patients Data",
                data=csv_data,
                file_name=f"similar_mimic_patients_{input_cancer.replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning(f"⚠️ Only found {len(similar_patients)} similar patients in MIMIC-IV. Consider expanding criteria.")
            
            if len(similar_patients) > 0:
                st.write(f"\n**{len(similar_patients)} patients found with current criteria:**")
                display_cols = ['subject_id', 'age', 'outcome']
                display_cols = [col for col in display_cols if col in similar_patients.columns]
                st.dataframe(similar_patients[display_cols].head(), 
                           use_container_width=True, hide_index=True)

# =============================================
# MAIN APPLICATION (FIXED)
# =============================================

def main():
    """Main application function using real MIMIC-IV data."""
    
    # Title and description
    st.title("🏥 Real MIMIC-IV Oncology Pathway Mapping Engine")
    st.markdown("**Analyzing Real Clinical Pathways from MIMIC-IV Dataset**")
    st.markdown("*Real patient data • Actual treatment patterns • Evidence-based outcomes*")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.patients_df = None
        st.session_state.events_df = None
        st.session_state.summary = None
    
    # Configuration section
    st.sidebar.header("⚙️ Configuration")
    
    # Project ID input
    project_id = st.sidebar.text_input(
        "Google Cloud Project ID",
        value="mimic-oncology-pathways",
        help="Your Google Cloud Project ID with BigQuery access to MIMIC-IV"
    )
    
    # Data limit
    data_limit = st.sidebar.slider(
        "Max Patients to Load",
        min_value=5,
        max_value=100,
        value=10,
        help="Number of oncology patients to extract from MIMIC-IV"
    )
    
    # Load data button
    if st.sidebar.button("🔄 Load MIMIC-IV Data", type="primary"):
        
        if not project_id or project_id.strip() == "":
            st.sidebar.error("Please enter a valid Project ID")
        else:
            # Clear previous data
            st.session_state.data_loaded = False
            st.session_state.patients_df = None
            st.session_state.events_df = None
            st.session_state.summary = None
            
            with st.spinner(f"🔄 Loading {data_limit} oncology patients from MIMIC-IV..."):
                try:
                    # Test connection first
                    st.info(f"🔗 Testing connection to project: {project_id}")
                    if not test_connection(project_id):
                        st.error("❌ Cannot connect to MIMIC-IV BigQuery. Please check your setup.")
                        st.stop()
                    
                    st.success("✅ Connected to BigQuery successfully!")
                    st.info("🔍 Extracting oncology patients from MIMIC-IV...")
                    
                    # Load data
                    patients_df, events_df, summary = load_mimic_oncology_data_no_cache(project_id, data_limit)
                    
                    # Store in session state
                    st.session_state.patients_df = patients_df
                    st.session_state.events_df = events_df
                    st.session_state.summary = summary
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ Successfully loaded {len(patients_df)} patients with {len(events_df)} events")
                    st.rerun()
                    
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Error loading MIMIC-IV data: {error_msg}")
                    
                    # Provide specific help based on error type
                    if "403" in error_msg or "Access Denied" in error_msg:
                        st.markdown("""
                        **403 Access Denied Error:**
                        
                        1. **Check MIMIC-IV access:**
                        - Complete training at: https://physionet.org/
                        - Request access to MIMIC-IV on Google Cloud
                        - Wait for approval (can take 1-2 days)
                        
                        2. **Verify authentication:**
                        ```bash
                        gcloud auth application-default login
                        ```
                        """)
                    elif "404" in error_msg or "Not found" in error_msg:
                        st.markdown("""
                        **404 Not Found Error:**
                        
                        - Check your project ID is correct
                        - Verify the dataset names exist in your project
                        - Make sure you have the right MIMIC-IV version
                        """)
    
    # Show authentication help
    with st.sidebar.expander("🆘 Authentication Help"):
        st.markdown("""
        **If connection fails:**
        
        1. **Run authentication setup:**
        ```bash
        python setup_auth.py
        ```
        
        2. **Or manually set up gcloud:**
        ```bash
        gcloud config set project mimic-oncology-pathways
        gcloud auth application-default login
        ```
        
        3. **Or use service account:**
        ```bash
        export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"
        ```
        """)
    
    # Main application logic
    if not st.session_state.data_loaded:
        st.info("👆 Please load MIMIC-IV data using the sidebar to get started.")
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Enter your Google Cloud Project ID** in the sidebar
        2. **Set the number of patients** to load (start with 10-20)
        3. **Click "Load MIMIC-IV Data"** to extract real patient data
        4. **Explore the dashboard** once data is loaded
        
        ### 📋 What You'll Get
        
        - **Real patient demographics** from MIMIC-IV
        - **Actual cancer diagnoses** based on ICD codes
        - **Clinical pathways** from real hospital data
        - **Treatment patterns** and outcomes
        """)
        st.stop()
    
    # Get data from session state
    patients_df = st.session_state.patients_df
    events_df = st.session_state.events_df
    summary = st.session_state.summary
    
    # Show dataset info
    with st.expander("📋 Real MIMIC-IV Dataset Information", expanded=True):
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Real Patients Loaded", summary['total_patients'])
            st.metric("Cancer Categories", len(summary['cancer_categories']))
        
        with col2:
            st.metric("Cancer Types Found", len(summary['cancer_types']))
            st.metric("Clinical Events", len(events_df))
        
        with col3:
            st.write("**Gender Distribution:**")
            for gender, count in summary['demographics']['gender'].items():
                st.write(f"• {gender}: {count}")
        
        with col4:
            st.write("**Age Statistics:**")
            st.write(f"• Mean: {summary['demographics']['age_mean']:.1f}")
            st.write(f"• Std: {summary['demographics']['age_std']:.1f}")
        
        # Show cancer types found
        st.subheader("🔬 Cancer Types Found in MIMIC-IV")
        
        cancer_cols = st.columns(3)
        cancer_types = list(summary['cancer_types'].keys())
        
        for i, cancer_type in enumerate(cancer_types):
            with cancer_cols[i % 3]:
                st.write(f"• **{cancer_type}**: {summary['cancer_types'][cancer_type]} patients")
        
        st.success("✅ Successfully loaded real MIMIC-IV oncology data!")
        st.info("💡 **This is real patient data from MIMIC-IV** - all analyses use actual clinical records, treatment patterns, and outcomes.")
    
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
            "Digital Twin Matcher"
        ]
    )
    
    # Display current cohort info
    cohort_info = f"""
    **Current Cohort:** {len(filtered_patients)} real patients  
    **Cancer Categories:** {', '.join(filtered_patients['cancer_category'].unique())}  
    **Data Source:** MIMIC-IV Real Hospital Data  
    **Events:** {len(filtered_events)} real clinical events
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

if __name__ == "__main__":
    main()