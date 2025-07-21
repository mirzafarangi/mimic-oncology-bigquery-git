"""
Analyze clinical pathways from extracted cohort.
"""

import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

class PathwayAnalyzer:
    """Analyze clinical pathways and transitions."""
    
    def __init__(self, patients: pd.DataFrame, diagnoses: pd.DataFrame):
        self.patients = patients
        self.diagnoses = diagnoses
        self.pathways = {}
        
    def extract_clinical_events(self) -> pd.DataFrame:
        """Extract and normalize clinical events."""
        
        print("🔄 Extracting clinical events...")
        
        # Convert diagnoses to events
        events = []
        
        for _, diagnosis in self.diagnoses.iterrows():
            if pd.notna(diagnosis['admittime']):
                events.append({
                    'patient_id': diagnosis['subject_id'],
                    'event_type': 'diagnosis',
                    'event_subtype': diagnosis['malignancy_type'],
                    'event_code': diagnosis['icd_code'],
                    'timestamp': pd.to_datetime(diagnosis['admittime']),
                    'description': diagnosis['diagnosis_name']
                })
        
        # Convert to DataFrame and sort
        events_df = pd.DataFrame(events)
        if len(events_df) > 0:
            events_df = events_df.sort_values(['patient_id', 'timestamp'])
            events_df['event_sequence'] = events_df.groupby('patient_id').cumcount() + 1
        
        print(f"📊 Extracted {len(events_df):,} clinical events")
        return events_df
    
    def build_patient_pathways(self, events_df: pd.DataFrame) -> Dict:
        """Build patient pathway sequences."""
        
        pathways = {}
        
        for patient_id in events_df['patient_id'].unique():
            patient_events = events_df[events_df['patient_id'] == patient_id].sort_values('timestamp')
            
            pathway = []
            for _, event in patient_events.iterrows():
                days_from_start = 0
                if len(pathway) > 0:
                    days_from_start = (event['timestamp'] - patient_events.iloc[0]['timestamp']).days
                
                pathway.append({
                    'event_type': event['event_type'],
                    'event_subtype': event['event_subtype'],
                    'timestamp': event['timestamp'],
                    'days_from_start': days_from_start
                })
            
            pathways[str(patient_id)] = pathway
        
        self.pathways = pathways
        print(f"🛤️  Built pathways for {len(pathways)} patients")
        
        return pathways
    
    def analyze_pathway_patterns(self) -> Dict:
        """Analyze common pathway patterns."""
        
        analysis = {
            'pathway_lengths': [],
            'common_transitions': Counter(),
            'malignancy_sequences': Counter()
        }
        
        for patient_id, pathway in self.pathways.items():
            # Pathway length
            analysis['pathway_lengths'].append(len(pathway))
            
            # Transitions (if multiple events)
            if len(pathway) > 1:
                for i in range(len(pathway) - 1):
                    transition = f"{pathway[i]['event_subtype']} → {pathway[i+1]['event_subtype']}"
                    analysis['common_transitions'][transition] += 1
            
            # Malignancy sequences
            sequence = ' → '.join([event['event_subtype'] for event in pathway])
            analysis['malignancy_sequences'][sequence] += 1
        
        # Convert to summary statistics
        lengths = analysis['pathway_lengths']
        summary = {
            'total_pathways': len(self.pathways),
            'pathway_length_stats': {
                'mean': sum(lengths) / len(lengths) if lengths else 0,
                'min': min(lengths) if lengths else 0,
                'max': max(lengths) if lengths else 0
            },
            'top_transitions': analysis['common_transitions'].most_common(10),
            'top_sequences': analysis['malignancy_sequences'].most_common(10)
        }
        
        return summary
    
    def visualize_cohort(self) -> None:
        """Create visualizations of the cohort."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Age distribution
        ages = self.patients['anchor_age'].dropna()
        axes[0,0].hist(ages, bins=20, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Age Distribution')
        axes[0,0].set_xlabel('Age')
        axes[0,0].set_ylabel('Frequency')
        
        # Gender distribution
        gender_counts = self.patients['gender'].value_counts()
        axes[0,1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('Gender Distribution')
        
        # Malignancy types
        all_malignancies = []
        for types_str in self.patients['malignancy_types'].dropna():
            all_malignancies.extend([t.strip() for t in types_str.split(';')])
        
        malignancy_counts = pd.Series(all_malignancies).value_counts()
        axes[1,0].barh(malignancy_counts.index, malignancy_counts.values, color='lightcoral')
        axes[1,0].set_title('Malignancy Types')
        axes[1,0].set_xlabel('Count')
        
        # Pathway lengths (if available)
        if self.pathways:
            pathway_lengths = [len(pathway) for pathway in self.pathways.values()]
            axes[1,1].hist(pathway_lengths, bins=10, color='gold', alpha=0.7)
            axes[1,1].set_title('Events per Patient')
            axes[1,1].set_xlabel('Number of Events')
            axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def save_analysis(self, output_dir: str = "outputs/") -> str:
        """Save pathway analysis results."""
        
        import os, json
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare analysis data
        analysis_data = {
            'cohort_summary': {
                'total_patients': len(self.patients),
                'total_pathways': len(self.pathways),
                'avg_pathway_length': sum(len(p) for p in self.pathways.values()) / len(self.pathways) if self.pathways else 0
            },
            'pathways': {pid: pathway for pid, pathway in list(self.pathways.items())[:10]}  # Sample pathways
        }
        
        # Save to JSON
        output_file = f"{output_dir}/pathway_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        print(f"💾 Analysis saved to: {output_file}")
        return output_file


def analyze_oncology_pathways(patients: pd.DataFrame, diagnoses: pd.DataFrame) -> Dict:
    """Main function to analyze oncology pathways."""
    
    analyzer = PathwayAnalyzer(patients, diagnoses)
    
    # Extract events and build pathways
    events = analyzer.extract_clinical_events()
    pathways = analyzer.build_patient_pathways(events)
    
    # Analyze patterns
    patterns = analyzer.analyze_pathway_patterns()
    
    # Create visualizations
    analyzer.visualize_cohort()
    
    # Save results
    analyzer.save_analysis()
    
    return patterns