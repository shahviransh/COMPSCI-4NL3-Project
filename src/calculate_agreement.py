"""
Calculate inter-annotator agreement for 20 Newsgroups annotations

This script calculates Cohen's Kappa, Fleiss' Kappa, or Krippendorff's Alpha
depending on the annotation setup.
"""

import json
from collections import defaultdict
from pathlib import Path
import krippendorff
import numpy as np

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent


def load_all_annotations():
    annotation_files = list(SCRIPT_DIR.glob('annotations_*.json'))
    annotation_files = [f for f in annotation_files if 'backup' not in f.name and 'reannotate_' not in f.name]
    
    if len(annotation_files) == 0:
        print("ERROR: No annotation files found")
        return None
    
    print(f"Found {len(annotation_files)} annotation file(s):")
    for f in annotation_files:
        print(f"  - {f.name}")
    
    all_data = {}
    for file in annotation_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            annotator = data['annotator']
            
            # Combine regular annotations and re-annotations
            annotations = data['annotations']
            
            # Check for re-annotation file
            reannotation_file = SCRIPT_DIR / f"{annotator}_reannotation.json"
            if reannotation_file.exists():
                with open(reannotation_file, 'r', encoding='utf-8') as rf:
                    reann_data = json.load(rf)
                    annotations.extend(reann_data['annotations'])
                    print(f"    + {len(reann_data['annotations'])} re-annotations")
            
            all_data[annotator] = annotations
    
    return all_data


def find_overlapping_annotations(all_data):
    # Group by document_id
    doc_annotations = defaultdict(list)
    
    for annotator, annotations in all_data.items():
        for ann in annotations:
            doc_id = ann['document_id']
            category = ann['category_number']
            doc_annotations[doc_id].append((annotator, category))
    
    # Find documents with multiple annotations
    overlapping = {}
    for doc_id, anns in doc_annotations.items():
        if len(anns) >= 2:
            overlapping[doc_id] = anns
    
    return overlapping

def calculate_krippendorffs_alpha(overlapping_annotations, all_data):
    try:
               
        # Build reliability matrix for only overlapping documents
        # Rows = annotators, Columns = overlapping documents
        annotators = list(all_data.keys())
        overlapping_doc_ids = sorted(overlapping_annotations.keys())
        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(overlapping_doc_ids)}

        # Create matrix (annotators x overlapping documents)
        # Use np.nan for missing values
        matrix = np.full((len(annotators), len(overlapping_doc_ids)), np.nan)

        for annotator_idx, annotator in enumerate(annotators):
            for ann in all_data[annotator]:
                doc_id = ann['document_id']
                if doc_id in doc_id_to_idx:
                    doc_idx = doc_id_to_idx[doc_id]
                    category = ann['category_number']
                    matrix[annotator_idx, doc_idx] = category

        print(f"\nCalculating Krippendorff's Alpha for {len(annotators)} annotators:")
        print(f"  - Overlapping documents: {len(overlapping_doc_ids)}")
        print(f"  - Reliability matrix shape: {matrix.shape}")

        # Calculate alpha
        alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement='nominal')

        print(f"\n  Krippendorff's Alpha: {alpha:.4f}")

        return alpha
        
    except ImportError:
        print("\nWARNING: krippendorff library not installed.")
        print("   Install with: pip install krippendorff")
        print("   (Note: krippendorff is only available via pip, not conda)")
        print("\n   Falling back to simpler calculation...")
        return None


def interpret_agreement(score, metric_name):
    print(f"\nInterpretation of {metric_name} = {score:.4f}:")
    
    if score < 0:
        interpretation = "Poor (less than chance agreement)"
    elif score < 0.20:
        interpretation = "Slight agreement"
    elif score < 0.40:
        interpretation = "Fair agreement"
    elif score < 0.60:
        interpretation = "Moderate agreement"
    elif score < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    
    print(f"  {interpretation}")
    print("\nGeneral guidelines (Landis & Koch, 1977):")
    print("  < 0.00: Poor")
    print("  0.00-0.20: Slight")
    print("  0.21-0.40: Fair")
    print("  0.41-0.60: Moderate")
    print("  0.61-0.80: Substantial")
    print("  0.81-1.00: Almost Perfect")


def main():
    print("INTER-ANNOTATOR AGREEMENT CALCULATOR")
    print("COMPSCI 4NL3 - Group 37")
    
    # Load all annotations
    all_data = load_all_annotations()
    if all_data is None:
        return
    
    # Find overlapping annotations
    overlapping = find_overlapping_annotations(all_data)
    
    print(f"\n{len(overlapping)} documents annotated by multiple people")
    
    if len(overlapping) == 0:
        print("\nWARNING: No overlapping annotations found.")
        print("   For agreement calculation, you need documents annotated by multiple people.")
        return
    
    # Category mappings
    CATEGORIES = {
        1: "comp.graphics",
        2: "comp.os.ms-windows.misc",
        3: "comp.sys.ibm.pc.hardware",
        4: "comp.sys.mac.hardware",
        5: "comp.windows.x",
        6: "rec.autos",
        7: "rec.motorcycles",
        8: "rec.sport.baseball",
        9: "rec.sport.hockey",
        10: "sci.crypt",
        11: "sci.electronics",
        12: "sci.med",
        13: "sci.space",
        14: "misc.forsale",
        15: "talk.politics.guns",
        16: "talk.politics.mideast",
        17: "talk.politics.misc",
        18: "talk.religion.misc",
        19: "alt.atheism",
        20: "soc.religion.christian"
    }
    
    # Try Krippendorff's Alpha
    alpha = calculate_krippendorffs_alpha(overlapping, all_data)
    if alpha is not None:
        interpret_agreement(alpha, "Krippendorff's Alpha")

        # Diagnostic: Print disagreements for overlapping documents
        print("\n--- Overlapping Document Disagreements ---")
        disagreement_count = 0
        for doc_id, anns in overlapping.items():
            categories = [cat for _, cat in anns]
            if len(set(categories)) > 1:
                disagreement_count += 1
                print(f"Document ID {doc_id}:")
                # Try to find the original label (first annotation for this doc_id in any annotator)
                original_label = None
                for annotator, annotations in all_data.items():
                    for ann in annotations:
                        if ann['document_id'] == doc_id:
                            original_label = ann.get('original_label', None)
                            break
                    if original_label is not None:
                        break
                print(f"  Original label: {original_label}")
                for annotator, cat in anns:
                    print(f"  {annotator}: {CATEGORIES.get(cat, f'Unknown category {cat}')}")
        print(f"\nTotal overlapping documents: {len(overlapping)}")
        print(f"Documents with disagreement: {disagreement_count}")
        print(f"Documents with full agreement: {len(overlapping) - disagreement_count}")


if __name__ == "__main__":
    main()
