"""
Prepare documents for re-annotation (overlap annotation)

This script helps you select documents that have been annotated by one person
to be re-annotated by another person for calculating inter-annotator agreement.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Get the directory where this script  is located
SCRIPT_DIR = Path(__file__).parent


def load_all_annotations():
    annotation_files = list(SCRIPT_DIR.glob('annotations_*.json'))
    annotation_files = [f for f in annotation_files if 'backup' not in f.name]
    
    if len(annotation_files) == 0:
        print("ERROR: No annotation files found")
        return None
    
    print(f"Found {len(annotation_files)} annotation file(s):")
    all_data = {}
    for file in annotation_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            annotator = data['annotator']
            all_data[annotator] = data
            num_annotations = len(data['annotations'])
            print(f"  - {file.name}: {num_annotations} annotations by {annotator}")
    
    return all_data


def select_overlap_documents(all_data, percentage=15, seed=42):
    random.seed(seed)
    
    # Get all annotated document IDs per annotator
    annotator_docs = {}
    for annotator, data in all_data.items():
        doc_ids = [ann['document_id'] for ann in data['annotations']]
        annotator_docs[annotator] = doc_ids
        print(f"{annotator}: {len(doc_ids)} documents annotated")
    
    # Calculate total and overlap size
    total_docs = sum(len(docs) for docs in annotator_docs.values())
    overlap_size = int(total_docs * (percentage / 100))
    
    print(f"\nTotal documents annotated: {total_docs}")
    print(f"Target overlap size ({percentage}%): {overlap_size}")
    
    # Distribute overlap among annotators
    num_annotators = len(annotator_docs)
    docs_per_annotator = overlap_size // num_annotators
    
    print(f"Documents per annotator to re-annotate: {docs_per_annotator}")
    
    # Create assignment: each annotator re-annotates some docs from others
    assignments = defaultdict(list)
    annotators = list(annotator_docs.keys())
    
    for i, current_annotator in enumerate(annotators):
        # This annotator will re-annotate docs from the next annotator (circular)
        next_annotator = annotators[(i + 1) % num_annotators]
        
        # Select random documents from next_annotator
        available_docs = annotator_docs[next_annotator]
        if len(available_docs) >= docs_per_annotator:
            selected = random.sample(available_docs, docs_per_annotator)
        else:
            selected = available_docs
            print(f"WARNING: Not enough documents from {next_annotator} for {current_annotator}")
        
        assignments[current_annotator] = {
            'reannotate_from': next_annotator,
            'document_ids': selected
        }
    
    return assignments


def save_reannotation_assignments(assignments):
    filename = SCRIPT_DIR / "reannotation_assignments.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(assignments, f, indent=2)
    
    print(f"\nAssignments saved to {filename}")


def create_reannotation_files(assignments, all_data):
    for annotator, assignment in assignments.items():
        filename = SCRIPT_DIR / f"reannotate_{annotator}.json"
        
        # Get the documents to re-annotate
        doc_ids = assignment['document_ids']
        source_annotator = assignment['reannotate_from']
        
        # Find the actual document info
        source_annotations = all_data[source_annotator]['annotations']
        docs_to_annotate = [ann for ann in source_annotations if ann['document_id'] in doc_ids]
        
        data = {
            'annotator': annotator,
            'mode': 'reannotation',
            'reannotating_from': source_annotator,
            'document_ids': doc_ids,
            'num_documents': len(doc_ids)
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Created {filename} for {annotator} ({len(doc_ids)} docs)")


def display_statistics(all_data, assignments):
    print("\nANNOTATION STATISTICS")
    
    # Per annotator stats
    for annotator, data in all_data.items():
        annotations = data['annotations']
        num_total = len(annotations)
        num_overlap = len(assignments[annotator]['document_ids'])
        
        print(f"\n{annotator}:")
        print(f"  - Total annotations: {num_total}")
        print(f"  - Will re-annotate: {num_overlap} docs from {assignments[annotator]['reannotate_from']}")
        print(f"  - New total after re-annotation: {num_total + num_overlap}")
    
    # Overall stats
    total_unique = sum(len(data['annotations']) for data in all_data.values())
    total_overlap = sum(len(a['document_ids']) for a in assignments.values())
    total_with_overlap = total_unique + total_overlap
    overlap_percentage = (total_overlap / total_with_overlap) * 100
    
    print(f"\nOVERALL:")
    print(f"  - Unique documents: {total_unique}")
    print(f"  - Overlap annotations: {total_overlap}")
    print(f"  - Total annotations: {total_with_overlap}")
    print(f"  - Overlap percentage: {overlap_percentage:.1f}%")


def main():
    print("RE-ANNOTATION SETUP TOOL")
    print("COMPSCI 4NL3 - Group 37")
    print()
    
    # Load annotations
    all_data = load_all_annotations()
    if all_data is None:
        return
        
    # Get parameters
    percentage = input("\nEnter overlap percentage (default 15%): ").strip()
    if not percentage:
        percentage = 15
    else:
        try:
            percentage = float(percentage)
        except ValueError:
            print("Invalid percentage, using default 15%")
            percentage = 15
    
    seed = input("Enter random seed for reproducibility (default 42): ").strip()
    if not seed:
        seed = 42
    else:
        try:
            seed = int(seed)
        except ValueError:
            print("Invalid seed, using default 42")
            seed = 42
    
    print("\nSELECTING DOCUMENTS FOR OVERLAP")
    
    # Select documents
    assignments = select_overlap_documents(all_data, percentage, seed)
    
    # Save assignments
    save_reannotation_assignments(assignments)
    create_reannotation_files(assignments, all_data)
    
    # Display statistics
    display_statistics(all_data, assignments)
    
    print("\nNEXT STEPS")
    print("\n1. Each team member should check their reannotate_<name>.json file")
    print("2. Run main.py again and choose mode 2 (Re-annotation)")
    print("3. The tool will load the documents you need to re-annotate")
    print("4. After everyone completes re-annotation, run calculate_agreement.py")


if __name__ == "__main__":
    main()
