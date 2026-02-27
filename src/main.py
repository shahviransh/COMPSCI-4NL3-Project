"""
20 Newsgroups Annotation Interface
COMPSCI 4NL3 - Group 37
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent


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


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def display_categories():
    print("\nCATEGORIES:")
    for num, name in CATEGORIES.items():
        print(f"  {num}. {name}")


def display_document(doc_id, text, subject, total_docs, annotated_count):
    clear_screen()
    print(f"20 NEWSGROUPS ANNOTATION TOOL - Document {doc_id} of {total_docs}")
    print(f"Progress: {annotated_count}/{total_docs} annotated ({annotated_count/total_docs*100:.1f}%)")
    
    # Display subject
    print(f"\nSUBJECT: {subject}")
    
    # Display document text
    print("\nDOCUMENT TEXT:")
    print(text[:2000])  # Limit display to first 2000 characters
    if len(text) > 2000:
        print("\n... (text truncated for display) ...")


def get_annotation():
    display_categories()
    
    print("\nCOMMANDS:")
    print("  Enter 1-20: Assign category")
    print("  's': Skip this document")
    print("  'q': Save and quit")
    print("  'h': Show categories again")
    
    while True:
        response = input("\nYour choice: ").strip().lower()
        
        if response == 'q':
            return 'quit'
        elif response == 's':
            return 'skip'
        elif response == 'h':
            display_categories()
            continue
        
        try:
            choice = int(response)
            if 1 <= choice <= 20:
                return choice
            else:
                print("ERROR: Please enter a number between 1 and 20.")
        except ValueError:
            print("ERROR: Invalid input. Please enter a number (1-20), 's' to skip, or 'q' to quit.")


def load_data():
    try:
        
        print("Loading 20 Newsgroups dataset...")
        
        # Load with headers to extract subjects
        train_data_with_headers = fetch_20newsgroups(subset='train', remove=('footers', 'quotes'))
        test_data_with_headers = fetch_20newsgroups(subset='test', remove=('footers', 'quotes'))
        
        # Load without headers for clean text
        train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
        
        # Combine data
        all_texts = list(train_data.data) + list(test_data.data)
        all_texts_with_headers = list(train_data_with_headers.data) + list(test_data_with_headers.data)
        all_labels = list(train_data.target) + list(test_data.target)
        
        # Create document entries
        documents = []
        for i, (text, text_with_headers, label) in enumerate(zip(all_texts, all_texts_with_headers, all_labels)):
            # Extract subject from headers
            subject = "(No Subject)"
            for line in text_with_headers.split('\n')[:20]:  # Check first 20 lines
                if line.startswith('Subject: '):
                    subject = line[9:].strip()
                    break
            
            documents.append({
                'id': i,
                'text': text,
                'subject': subject,
                'original_label': train_data.target_names[label],
                'source': 'train' if i < len(train_data.data) else 'test'
            })
        
        print(f"Loaded {len(documents)} documents")
        return documents
        
    except ImportError:
        print("ERROR: scikit-learn is not installed. Please install it with:")
        print("   conda install scikit-learn")
        print("   or activate your environment: conda activate newsgroups-annotation")
        sys.exit(1)


def load_annotations(annotator_name):
    filename = SCRIPT_DIR / f"annotations_{annotator_name}.json"
    if filename.exists():
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded {len(data.get('annotations', []))} existing annotations")
            return data
    return {
        'annotator': annotator_name,
        'created_at': datetime.now().isoformat(),
        'annotations': []
    }


def load_all_annotated_ids():
    """Load document IDs that have been annotated by ANY annotator."""
    all_annotated_ids = set()
    
    # Find all annotation files
    for annotation_file in SCRIPT_DIR.glob("annotations_*.json"):
        # Skip backup and reannotation files
        if 'backup' in annotation_file.name or 'reannotation' in annotation_file.name:
            continue
        
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for ann in data.get('annotations', []):
                    all_annotated_ids.add(ann['document_id'])
        except Exception as e:
            print(f"Error loading annotations from {annotation_file}: {e}")
            continue
    
    return all_annotated_ids


def save_annotations(annotation_data, annotator_name, mode='new'):
    suffix = '_reannotation' if mode == 'reannotate' else ''
    prefix = '' if mode == 'reannotate' else "annotations_"
    filename = SCRIPT_DIR / f"{prefix}{annotator_name}{suffix}.json"
    annotation_data['last_updated'] = datetime.now().isoformat()
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(annotation_data, f, indent=2, ensure_ascii=False)
    
    # Also save a backup
    backup_filename = SCRIPT_DIR / f"{prefix}{annotator_name}{suffix}_backup.json"
    with open(backup_filename, 'w', encoding='utf-8') as f:
        json.dump(annotation_data, f, indent=2, ensure_ascii=False)


def load_reannotation_assignment(annotator_name):
    filename = SCRIPT_DIR / f"reannotate_{annotator_name}.json"
    if filename.exists():
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def get_next_document(documents, annotation_data, mode='new', reannotation_assignment=None, skipped_ids=None, all_annotated_ids=None):
    if skipped_ids is None:
        skipped_ids = set()
    
    annotated_ids = {ann['document_id'] for ann in annotation_data['annotations']}
    excluded_ids = annotated_ids | skipped_ids
    
    if mode == 'new':
        # For new annotations, also exclude documents annotated by ANYONE
        if all_annotated_ids is not None:
            excluded_ids = excluded_ids | all_annotated_ids
        
        # Find first unannotated and non-skipped document
        for doc in documents:
            if doc['id'] not in excluded_ids:
                return doc
    elif mode == 'reannotate':
        # For re-annotation mode, get documents from assignment
        if reannotation_assignment is None:
            return None
        
        target_ids = set(reannotation_assignment['document_ids'])
        
        # Find next document in assignment that hasn't been re-annotated or skipped yet
        for doc in documents:
            if doc['id'] in target_ids and doc['id'] not in excluded_ids:
                return doc
    
    return None

def main():
    clear_screen()
    print("20 NEWSGROUPS ANNOTATION TOOL")
    print("COMPSCI 4NL3 - Group 37")
    
    # Get annotator name
    annotator_name = input("\nEnter your name: ").strip().lower()
    if not annotator_name:
        print("Name cannot be empty")
        return
    
    # Choose mode
    print("\nMODE:")
    print("  1. New annotation (annotate new documents)")
    print("  2. Re-annotation (annotate documents already labeled by others)")
    mode_choice = input("Choose mode (1 or 2): ").strip()
    
    mode = 'new' if mode_choice == '1' else 'reannotate'
    
    # Load re-annotation assignment if needed
    reannotation_assignment = None
    if mode == 'reannotate':
        reannotation_assignment = load_reannotation_assignment(annotator_name)
        if reannotation_assignment is None:
            print(f"\nERROR: No re-annotation assignment found for {annotator_name}")
            print("   Run prepare_reannotation.py first to create assignments.")
            return
        
        print("\nRe-annotation assignment loaded:")
        print(f"   - Re-annotating documents from: {reannotation_assignment['reannotating_from']}")
        print(f"   - Number of documents: {reannotation_assignment['num_documents']}")
    
    # Load data
    documents = load_data()
    annotation_data = load_annotations(annotator_name)
    
    # For new annotations, load all annotated IDs to avoid duplicates
    all_annotated_ids = None
    if mode == 'new':
        all_annotated_ids = load_all_annotated_ids()
        if all_annotated_ids:
            print(f"Found {len(all_annotated_ids)} documents already annotated by all annotators")
            print("Will skip these to avoid duplicates")
    
    # For re-annotation, use a separate file
    if mode == 'reannotate':
        reannotation_file = SCRIPT_DIR / f"{annotator_name}_reannotation.json"
        if reannotation_file.exists():
            with open(reannotation_file, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
                print(f"Loaded existing re-annotations: {len(annotation_data['annotations'])}")
        else:
            annotation_data = {
                'annotator': annotator_name,
                'mode': 'reannotation',
                'reannotating_from': reannotation_assignment['reannotating_from'],
                'created_at': datetime.now().isoformat(),
                'annotations': []
            }
    
    annotated_count = len(annotation_data['annotations'])
    skipped_ids = set()  # Track skipped documents in this session
    print(f"\nReady to annotate! Currently have {annotated_count} annotations.")
    input("\nPress Enter to start annotating...")
    
    # Annotation loop
    try:
        while True:
            # Get next document
            doc = get_next_document(documents, annotation_data, mode, reannotation_assignment, skipped_ids, all_annotated_ids)
            
            if doc is None:
                if mode == 'reannotate':
                    print("\nAll re-annotation documents have been completed!")
                else:
                    print("\nAll documents have been annotated!")
                break
            
            # Display document
            total_docs = reannotation_assignment['num_documents'] if mode == 'reannotate' else len(documents)
            
            display_document(doc['id'], doc['text'], doc['subject'], total_docs, annotated_count)
            
            result = get_annotation()
            
            if result == 'quit':
                break
            elif result == 'skip':
                skipped_ids.add(doc['id'])
                print("\nDocument skipped.")
                continue
            else:
                # Save annotation
                annotation = {
                    'document_id': doc['id'],
                    'category_number': result,
                    'category_name': CATEGORIES[result],
                    'original_label': doc['original_label'],
                    'timestamp': datetime.now().isoformat()
                }
                annotation_data['annotations'].append(annotation)
                annotated_count += 1
                
                # Save immediately
                save_annotations(annotation_data, annotator_name, mode)
                
                print(f"\nSaved! ({annotated_count} documents annotated)")
                input("Press Enter to continue...")
    
    except KeyboardInterrupt:
        print("\n\nWARNING: Interrupted by user")
    
    finally:
        # Final save
        save_annotations(annotation_data, annotator_name, mode)
        
        suffix = '_reannotation' if mode == 'reannotate' else ''
        prefix = '' if mode == 'reannotate' else "annotations_"
        filename_base = f"{prefix}{annotator_name}{suffix}"
        
        print(f"Total annotations: {len(annotation_data['annotations'])}")
        print(f"Data saved to: {filename_base}.json")


if __name__ == "__main__":
    main()
