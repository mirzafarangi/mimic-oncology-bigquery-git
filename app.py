# Comprehensive Oncology Pathway Mapping Dashboard
# Updated version with modular cancer registry system

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

# Import the cancer registry system
from cancer_registry import CancerRegistryManager, CANCER_REGISTRY

# Set page configuration
st.set_page_config(
    page_title="Comprehensive Oncology Pathway Mapping Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# DATA GENERATION
# =============================================

@st.cache_data
def generate_sample_data(n_patients: int = 500):
    """Generate comprehensive oncology patient data using modular cancer system."""
    
    np.random.seed(42)
    random.seed(42)
    
    # Initialize cancer registry manager
    manager = CancerRegistryManager()
    
    # Generate patient cohort
    patients, all_events = manager.generate_patient_cohort(n_patients)
    
    # Convert to DataFrames
    patients_df = pd.DataFrame(patients)
    events_df = pd.DataFrame(all_events)
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    
    # Merge patient outcomes back to patients_df
    outcomes = events_df[events_df['event_type'] == 'outcome'][['subject_id', 'event_subtype', 'days_from_start']]
    outcomes.columns = ['subject_id', 'outcome', 'outcome_days']
    patients_df = patients_df.merge(outcomes, on='subject_id', how='left')
    
    # Fill missing outcomes
    patients_df['outcome'] = patients_df['outcome'].fillna('Unknown')
    patients_df['outcome_days'] = patients_df['outcome_days'].fillna(365)
    
    return patients_df, events_df

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
# SIDEBAR FILTERS
# =============================================

def create_sidebar_filters(patients_df: pd.DataFrame, events_df: pd.DataFrame):
    """Create sidebar filter controls with expanded cancer types."""
    
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
        max_value=1095,
        value=730,
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
# DASHBOARD PAGES
# =============================================

def dashboard_home(patients_df, events_df):
    """Dashboard home page with overview statistics."""
    
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
            title="Distribution by Cancer Category",
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

def cohort_explorer(patients_df, events_df):
    """Cohort explorer with patient table and search."""
    
    st.header("👥 Cohort Explorer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_id = st.text_input("Search by Patient ID", placeholder="e.g., P0001")
    
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
    display_df['first_admission'] = pd.to_datetime(display_df['first_admission']).dt.strftime('%Y-%m-%d')
    
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
            good_outcomes = ['Complete Remission', 'Disease Free', 'No Evidence of Disease', 'Cure', 'Remission']
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
        
        total_patients = len(events_df['subject_id'].unique())
        total_transitions = len(sankey_data['links'])
        
        st.info(f"📊 Showing {total_transitions} common transitions for {total_patients} patients. "
               f"Only transitions with ≥3 patients are displayed for clarity.")
        
    else:
        st.warning("⚠️ No common treatment pathways found with current filters.")
    
    # Transition frequency table
    st.subheader("Event Transition Frequencies")
    
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
    
    st.header("📅 Patient Timeline Viewer")
    
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
        input_age = st.number_input("Patient Age", min_value=18, max_value=90, value=45)
        input_gender = st.selectbox("Gender", options=['M', 'F'], index=1)
    
    with col2:
        available_cancers = sorted(patients_df['cancer_type'].unique())
        default_cancer = 'Papillary Thyroid Carcinoma' if 'Papillary Thyroid Carcinoma' in available_cancers else available_cancers[0]
        input_cancer = st.selectbox("Cancer Type", options=available_cancers, 
                                   index=available_cancers.index(default_cancer))
        
        available_stages = sorted(patients_df['stage'].unique())
        input_stage = st.selectbox("Stage", options=available_stages)
    
    with col3:
        age_tolerance = st.slider("Age Tolerance (±years)", min_value=1, max_value=15, value=10)
        min_matches = st.slider("Minimum Matches Required", min_value=5, max_value=50, value=10)
    
    # Find similar patients
    if st.button("🔍 Find Similar Patients", type="primary"):
        
        similar_patients = patients_df[
            (patients_df['cancer_type'] == input_cancer) &
            (patients_df['gender'] == input_gender) &
            (patients_df['stage'] == input_stage) &
            (patients_df['age'] >= input_age - age_tolerance) &
            (patients_df['age'] <= input_age + age_tolerance)
        ].copy()
        
        if len(similar_patients) >= min_matches:
            st.success(f"✅ Found {len(similar_patients)} similar patients!")
            
            similar_patients['age_diff'] = abs(similar_patients['age'] - input_age)
            similar_patients = similar_patients.sort_values('age_diff')
            
            # Display top matches
            st.subheader("🎯 Top Digital Twin Matches")
            
            display_cols = ['subject_id', 'age', 'gender', 'cancer_type', 'stage', 'outcome', 'outcome_days']
            top_matches = similar_patients.head(10)[display_cols]
            
            st.dataframe(top_matches, use_container_width=True, hide_index=True)
            
            # Outcome analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Outcome Statistics for Similar Patients")
                
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
                    title="Outcome Distribution in Similar Patients"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Treatment pathway analysis
            st.subheader("🏥 Common Treatment Pathways")
            
            similar_patient_ids = similar_patients['subject_id'].tolist()
            similar_events = events_df[events_df['subject_id'].isin(similar_patient_ids)]
            
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
            
            # Clinical insights
            st.subheader("💡 Clinical Insights")
            
            good_outcomes = ['Complete Remission', 'Disease Free', 'No Evidence of Disease', 'Cure', 'Remission']
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
            st.warning(f"⚠️ Only found {len(similar_patients)} similar patients. Consider expanding criteria.")
            
            if len(similar_patients) > 0:
                st.write(f"\n**{len(similar_patients)} patients found with current criteria:**")
                st.dataframe(similar_patients[['subject_id', 'age', 'outcome']].head(), 
                           use_container_width=True, hide_index=True)

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
            good_outcomes = ['Complete Remission', 'Disease Free', 'No Evidence of Disease', 'Cure', 'Remission']
            good_a = group_a['outcome'].isin(good_outcomes).mean() * 100
            st.metric(f"Group A Good Outcome Rate", f"{good_a:.1f}%")
        
        with col2:
            st.metric(f"Group B Patients", len(group_b))
            good_b = group_b['outcome'].isin(good_outcomes).mean() * 100
            st.metric(f"Group B Good Outcome Rate", f"{good_b:.1f}%")
        
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
            fig = px.histogram(comparison_data, x='Outcome', color='Group', 
                             barmode='group', title="Outcome Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
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
        options=['cancer_type', 'stage', 'gender', 'cancer_category']
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
    st.markdown("**Analyzing Clinical Pathways Across 25+ Cancer Types and 8 Major Oncology Categories**")
    st.markdown("*Digital twin matching • Treatment pathway analysis • Outcome prediction • Comprehensive cancer registry*")
    
    # Load data
    with st.spinner("Loading comprehensive oncology dataset..."):
        patients_df, events_df = generate_sample_data()
    
    # Show dataset info
    with st.expander("📋 Comprehensive Cancer Registry Information", expanded=False):
        
        # Initialize cancer registry manager to get statistics
        manager = CancerRegistryManager()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**Hematologic Malignancies:**")
            st.write("• Hodgkin Lymphoma")
            st.write("• Non-Hodgkin Lymphoma") 
        
        with col2:
            st.write("**Genitourinary Cancers:**")
            st.write("• Prostate Cancer")
            st.write("• Renal Cell Carcinoma")
            st.write("• Bladder Cancer")
            st.write("• Testicular Cancer")
        
        with col3:
            st.write("**Gynecologic Cancers:**")
            st.write("• Ovarian Cancer")
            st.write("• Endometrial Cancer")
            st.write("• Cervical Cancer")
        
        with col4:
            st.write("**Thyroid Cancers:**")
            st.write("• Papillary Thyroid CA")
            st.write("• Anaplastic Thyroid CA")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**Thoracic Cancers:**")
            st.write("• Lung Adenocarcinoma")
            st.write("• Lung Squamous Cell")
            st.write("• Small Cell Lung CA")
            st.write("• Pleural Mesothelioma")
        
        with col2:
            st.write("**CNS Tumors:**")
            st.write("• Glioblastoma")
            st.write("• Meningioma")
            st.write("• Pituitary Adenoma")
        
        with col3:
            st.write("**GI Cancers:**")
            st.write("• Colorectal Cancer")
            st.write("• Pancreatic Cancer")
            st.write("• Gastric Cancer")
            st.write("• Esophageal Cancer")
            st.write("• Hepatocellular CA")
            st.write("• Cholangiocarcinoma")
        
        with col4:
            st.write("**Other Categories:**")
            st.write("• Melanoma")
            st.write("• Basal Cell Carcinoma")
            st.write("• Head & Neck SCC")
            st.write("• Osteosarcoma")
            st.write("• Soft Tissue Sarcoma")
        
        st.success("💡 **Enhanced Features**: Digital Twin Matcher finds patients similar to your current case across all 25+ cancer types!")
        
        # Registry statistics
        st.subheader("🔧 Cancer Registry Statistics")
        
        total_cancers = len(CANCER_REGISTRY)
        categories = set(config.category for config in CANCER_REGISTRY.values())
        total_rate = sum(config.incidence_rate for config in CANCER_REGISTRY.values())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cancer Types", total_cancers)
        with col2:
            st.metric("Cancer Categories", len(categories))
        with col3:
            st.metric("Patients Generated", len(patients_df))
        with col4:
            st.metric("Registry Completeness", f"{total_rate:.3f}" if abs(total_rate - 1.0) < 0.05 else "⚠️ Check rates")
        
        # Show detailed registry
        if st.button("📊 View Complete Cancer Registry Details"):
            registry_data = []
            for key, config in CANCER_REGISTRY.items():
                registry_data.append({
                    'Cancer Type': config.name,
                    'Category': config.category,
                    'Incidence Rate': f"{config.incidence_rate:.1%}",
                    'Staging System': config.staging_system,
                    'Number of Stages': len(config.stages),
                    'Number of Treatments': len(config.treatments),
                    'Mean Age': f"{config.age_distribution['mean']:.0f}",
                    'Gender Ratio (M:F)': f"{config.gender_distribution['M']:.0%}:{config.gender_distribution['F']:.0%}"
                })
            
            registry_df = pd.DataFrame(registry_data)
            st.dataframe(registry_df, use_container_width=True, hide_index=True)
    
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
    
    # Display current cohort info
    cohort_info = f"""
    **Current Cohort:** {len(filtered_patients)} patients  
    **Cancer Categories:** {', '.join(filtered_patients['cancer_category'].unique())}  
    **Cancer Types:** {len(filtered_patients['cancer_type'].unique())} different types  
    **Events:** {len(filtered_events)} total events
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