"""
Export all annotation and re-annotation data to a single Excel file with multiple sheets.

- Sheet 'original_data': document_id, subject, document_text, original_label
- For each annotator: sheet 'annotations_<name>': document_id, subject, document_text, assigned_label
- For each re-annotation: sheet 'reannotation_by_<annotator>_for_<source>': document_id, subject, document_text, assigned_label
"""

import json
from pathlib import Path
import pandas as pd
from main import load_data, CATEGORIES

SCRIPT_DIR = Path(__file__).parent


def get_all_annotators():
    files = list(SCRIPT_DIR.glob('annotations_*.json'))
    annotators = []
    for f in files:
        if 'backup' not in f.name and 'reannotation' not in f.name:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                annotators.append(data['annotator'])
    return annotators


def get_all_reannotators():
    files = list(SCRIPT_DIR.glob('*_reannotation.json'))
    reannotators = []
    for f in files:
        if 'backup' not in f.name:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                reannotators.append((data['annotator'], data.get('reannotating_from', 'unknown')))
    return reannotators


def main():
    documents = load_data()
    doc_map = {doc['id']: doc for doc in documents}
    
    # Sheet 1: original data
    original_rows = []
    for doc in documents:
        original_rows.append({
            'document_id': doc['id'],
            'subject': doc['subject'],
            'document_text': doc['text'],
            'original_label': doc['original_label']
        })
    df_original = pd.DataFrame(original_rows)
    
    # Annotation sheets
    annotators = get_all_annotators()
    annotation_dfs = {}
    for annotator in annotators:
        filename = SCRIPT_DIR / f"annotations_{annotator}.json"
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            rows = []
            for ann in data['annotations']:
                doc = doc_map.get(ann['document_id'])
                if doc:
                    rows.append({
                        'document_id': ann['document_id'],
                        'subject': doc['subject'],
                        'document_text': doc['text'],
                        'assigned_label': CATEGORIES.get(ann['category_number'], ann.get('category_name', ''))
                    })
            df = pd.DataFrame(rows)
            annotation_dfs[f'annotations_{annotator}'] = df
    
    # Re-annotation sheets
    reannotators = get_all_reannotators()
    reannotation_dfs = {}
    for annotator, source in reannotators:
        filename = SCRIPT_DIR / f"{annotator}_reannotation.json"
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            rows = []
            for ann in data.get('annotations', []):
                doc = doc_map.get(ann['document_id'])
                if doc:
                    rows.append({
                        'document_id': ann['document_id'],
                        'subject': doc['subject'],
                        'document_text': doc['text'],
                        'assigned_label': CATEGORIES.get(ann['category_number'], ann.get('category_name', ''))
                    })
            df = pd.DataFrame(rows)
            sheet_name = f"reannotate_{annotator}_for_{source}"
            reannotation_dfs[sheet_name] = df
    
    # Write to Excel
    with pd.ExcelWriter(SCRIPT_DIR / 'all_annotations.xlsx', engine='xlsxwriter') as writer:
        df_original.to_excel(writer, sheet_name='Original Data', index=False)
        for sheet, df in annotation_dfs.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
        for sheet, df in reannotation_dfs.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    print("Exported all annotation data to all_annotations.xlsx")

if __name__ == "__main__":
    main()
