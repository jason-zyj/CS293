import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import sys

def load_annotator_data(file_path, sheet_name):
    """Load annotations from a specific sheet"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

def calculate_irr_for_pair(df1, df2, annotator1_name, annotator2_name):
    """Calculate IRR metrics for a pair of annotators on overlapping rows"""
    
    # Merge on target_comb_idx to get overlapping annotations
    merged = pd.merge(df1, df2, on='target_comb_idx', suffixes=('_1', '_2'))
    
    if len(merged) == 0:
        print(f"No overlapping annotations found for {annotator1_name} and {annotator2_name}")
        return None
    
    print(f"\n{'='*60}")
    print(f"IRR Analysis: {annotator1_name} vs {annotator2_name}")
    print(f"{'='*60}")
    print(f"Number of overlapping annotations: {len(merged)}")
    
    # Binary variables to analyze
    binary_vars = ['R1: References prior student content', 
                   'R2: Builds on student content', 
                   'R3: Invites further student thinking']
    
    results = {}
    
    for var in binary_vars:
        var1 = f"{var}_1"
        var2 = f"{var}_2"
        
        # Handle missing values - treat NaN as FALSE
        annotations1 = merged[var1].fillna(False).astype(bool)
        annotations2 = merged[var2].fillna(False).astype(bool)
        
        # Calculate agreement metrics
        agreement = (annotations1 == annotations2).sum()
        percent_agreement = (agreement / len(merged)) * 100
        
        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(annotations1, annotations2)
        
        # Calculate prevalence for each coder
        prev1 = annotations1.sum() / len(annotations1) * 100
        prev2 = annotations2.sum() / len(annotations2) * 100
        
        results[var] = {
            'percent_agreement': percent_agreement,
            'cohen_kappa': kappa,
            'prevalence_annotator1': prev1,
            'prevalence_annotator2': prev2,
            'n_overlap': len(merged)
        }
        
        print(f"\n{var}:")
        print(f"  Percent Agreement: {percent_agreement:.2f}%")
        print(f"  Cohen's Kappa: {kappa:.3f}")
        print(f"  {annotator1_name} prevalence: {prev1:.1f}%")
        print(f"  {annotator2_name} prevalence: {prev2:.1f}%")
    
    # Calculate average kappa across all variables
    avg_kappa = np.mean([r['cohen_kappa'] for r in results.values()])
    print(f"\nAverage Cohen's Kappa across all variables: {avg_kappa:.3f}")
    
    return results

def main(file_path):
    """Main function to calculate IRR for all annotator pairs"""
    
    # Define the annotator pairs (without predefined row ranges)
    pairs = [
        ('Jason', 'Evelyn'),
        ('Evelyn', 'Elisabeth'),
        ('Elisabeth', 'Mrs. Cousins'),
        ('Mrs. Cousins', 'Jason')
    ]
    
    all_results = {}
    overlap_summary = []
    
    for annotator1, annotator2 in pairs:
        try:
            # Load data for each annotator
            df1 = load_annotator_data(file_path, annotator1)
            df2 = load_annotator_data(file_path, annotator2)
            
            # Find actual overlap
            overlap_ids = set(df1['target_comb_idx']) & set(df2['target_comb_idx'])
            n_overlap = len(overlap_ids)
            
            overlap_summary.append({
                'pair': f"{annotator1} & {annotator2}",
                'annotator1_total': len(df1),
                'annotator2_total': len(df2),
                'overlap': n_overlap
            })
            
            # Calculate IRR for this pair
            results = calculate_irr_for_pair(df1, df2, annotator1, annotator2)
            all_results[f"{annotator1}_{annotator2}"] = results
            
        except Exception as e:
            print(f"\nError processing {annotator1} vs {annotator2}: {str(e)}")
    
    # Print overlap summary first
    print(f"\n{'='*60}")
    print("OVERLAP SUMMARY FOR ALL PAIRS")
    print(f"{'='*60}")
    for summary in overlap_summary:
        print(f"\n{summary['pair']}:")
        print(f"  {summary['pair'].split(' & ')[0]} annotated: {summary['annotator1_total']} rows")
        print(f"  {summary['pair'].split(' & ')[1]} annotated: {summary['annotator2_total']} rows")
        print(f"  Overlapping rows: {summary['overlap']}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    for pair_name, results in all_results.items():
        if results:
            avg_kappa = np.mean([r['cohen_kappa'] for r in results.values()])
            print(f"{pair_name}: Average Kappa = {avg_kappa:.3f}")
    
    return all_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calculate_irr.py <path_to_excel_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    results = main(file_path)