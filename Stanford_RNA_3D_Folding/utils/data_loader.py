import pandas as pd
import numpy as np
import os
from Bio import SeqIO
import io
import time

def load_sequences(file_path):
    """Load RNA sequences from CSV file."""
    print(f"Loading sequences from {file_path}")
    start_time = time.time()
    
    df = pd.read_csv(file_path)
    
    duration = time.time() - start_time
    print(f"Loaded {len(df)} sequences in {duration:.2f} seconds")
    return df

def load_labels(file_path):
    """Load 3D structure labels from CSV file."""
    print(f"Loading structure labels from {file_path}")
    start_time = time.time()
    
    df = pd.read_csv(file_path)
    
    duration = time.time() - start_time
    print(f"Loaded {len(df)} label entries in {duration:.2f} seconds")
    return df

def load_msa(target_id, msa_dir):
    """Load multiple sequence alignment for a target."""
    msa_path = os.path.join(msa_dir, f"{target_id}.MSA.fasta")
    print(f"Attempting to load MSA for {target_id}")
    
    if os.path.exists(msa_path):
        with open(msa_path, 'r') as f:
            content = f.read()
        
        # Log the number of sequences in the MSA file
        seq_count = content.count('>')
        print(f"Loaded MSA for {target_id} with {seq_count} sequences")
        return content
    else:
        print(f"MSA file for {target_id} not found: {msa_path}")
        return None

def extract_coordinates(labels_df, target_id):
    """Extract 3D coordinates for a specific target."""
    print(f"Extracting coordinates for target {target_id}")
    
    # Filter for this target
    target_rows = labels_df[labels_df['ID'].str.startswith(f"{target_id}_")]
    
    if len(target_rows) == 0:
        print(f"No label entries found for target {target_id}")
        return {}
    
    # Sort by residue number to ensure correct order
    target_rows = target_rows.sort_values(by='resid')
    print(f"Found {len(target_rows)} residues for target {target_id}")
    
    # Determine how many structures we have (x_1, x_2, etc.)
    structure_indices = []
    for col in labels_df.columns:
        if col.startswith('x_'):
            structure_indices.append(col.split('_')[1])
    
    print(f"Found {len(structure_indices)} different structures for target {target_id}")
    
    # Extract coordinates for each structure
    structures = {}
    for idx in structure_indices:
        # Get columns for this structure
        x_col = f'x_{idx}'
        y_col = f'y_{idx}'
        z_col = f'z_{idx}'
        
        # Skip if any column doesn't exist
        if not all(col in target_rows.columns for col in [x_col, y_col, z_col]):
            print(f"Missing coordinate columns for structure {idx} of target {target_id}")
            continue
        
        # Extract coordinates
        coords = target_rows[[x_col, y_col, z_col]].values
        structures[idx] = coords
    
    print(f"Successfully extracted coordinates for {len(structures)} structures of target {target_id}")
    return structures

def parse_fasta(fasta_content):
    """Parse FASTA content to a list of SeqRecord objects."""
    if not fasta_content:
        print("Empty FASTA content provided")
        return []
    
    try:
        fasta_io = io.StringIO(fasta_content)
        records = list(SeqIO.parse(fasta_io, "fasta"))
        print(f"Parsed {len(records)} sequences from FASTA content")
        return records
    except Exception as e:
        print(f"Error parsing FASTA content: {str(e)}")
        return []

def get_target_sequences(target_id, sequences_df):
    """Get all relevant sequence information for a target."""
    print(f"Retrieving sequence info for target {target_id}")
    
    row = sequences_df[sequences_df['target_id'] == target_id]
    if row.empty:
        print(f"No sequence information found for target {target_id}")
        return None
    
    result = row.iloc[0].to_dict()
    seq_length = len(result['sequence']) if 'sequence' in result else 0
    print(f"Found sequence of length {seq_length} for target {target_id}")
    return result

def prepare_for_submission(predictions, target_ids, sequences_df):
    """Format model predictions into competition submission format."""
    print(f"Preparing submission for {len(target_ids)} targets")
    start_time = time.time()
    
    rows = []
    missing_targets = 0
    missing_structures = 0
    
    for target_id in target_ids:
        # Get sequence for this target
        seq_info = get_target_sequences(target_id, sequences_df)
        if seq_info is None:
            missing_targets += 1
            continue
            
        sequence = seq_info['sequence']
        
        # For each residue
        for i, base in enumerate(sequence):
            resid = i + 1  # 1-based indexing
            row = {
                'ID': f"{target_id}_{resid}",
                'resname': base,
                'resid': resid
            }
            
            # Add coordinates for each of the 5 predicted structures
            for struct_idx in range(5):
                if target_id in predictions and len(predictions[target_id]) > struct_idx:
                    pred_struct = predictions[target_id][struct_idx]
                    if i < len(pred_struct):
                        coords = pred_struct[i]
                        row[f'x_{struct_idx+1}'] = coords[0]
                        row[f'y_{struct_idx+1}'] = coords[1]
                        row[f'z_{struct_idx+1}'] = coords[2]
                    else:
                        # Handle case where prediction is missing residues
                        missing_structures += 1
                        row[f'x_{struct_idx+1}'] = np.nan
                        row[f'y_{struct_idx+1}'] = np.nan
                        row[f'z_{struct_idx+1}'] = np.nan
                else:
                    # Handle case where prediction is missing
                    missing_structures += 1
                    row[f'x_{struct_idx+1}'] = np.nan
                    row[f'y_{struct_idx+1}'] = np.nan
                    row[f'z_{struct_idx+1}'] = np.nan
            
            rows.append(row)
    
    # Create DataFrame and ensure all required columns
    submission_df = pd.DataFrame(rows)
    
    # Make sure columns are in the right order
    col_order = ['ID', 'resname', 'resid']
    for i in range(1, 6):
        col_order.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    # Reorder and select only necessary columns
    submission_df = submission_df[col_order]
    
    duration = time.time() - start_time
    print(f"Submission preparation completed in {duration:.2f} seconds")
    print(f"Created submission with {len(submission_df)} rows")
    print(f"Missing targets: {missing_targets}, Missing structures: {missing_structures}")
    
    return submission_df