"""
Generate Annotation Report
Calculates total instances annotated and average time per instance

Takes into account breaks: if time between annotations > 10 minutes,
treats it as a separate session (not counted in average time)
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# Maximum time between annotations before considering it a break (in minutes)
BREAK_THRESHOLD_MINUTES = 10


def parse_timestamp(timestamp_str):
    return datetime.fromisoformat(timestamp_str)


def load_all_annotations():
    annotation_files = list(SCRIPT_DIR.glob('annotations_*.json'))
    annotation_files = [f for f in annotation_files 
                        if 'backup' not in f.name and 'reannotation' not in f.name]
    
    if len(annotation_files) == 0:
        print("ERROR: No annotation files found")
        return None
    
    all_data = {}
    for file in annotation_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            annotator = data['annotator']
            all_data[annotator] = data
    
    return all_data


def calculate_annotation_times(annotations):
    if len(annotations) < 2:
        return []
    
    # Sort by timestamp
    sorted_annotations = sorted(annotations, key=lambda x: parse_timestamp(x['timestamp']))
    
    annotation_times = []
    
    for i in range(1, len(sorted_annotations)):
        prev_time = parse_timestamp(sorted_annotations[i-1]['timestamp'])
        curr_time = parse_timestamp(sorted_annotations[i]['timestamp'])
        
        time_diff = (curr_time - prev_time).total_seconds()
        time_diff_minutes = time_diff / 60
        
        # Only count if less than break threshold (10 minutes)
        if time_diff_minutes <= BREAK_THRESHOLD_MINUTES:
            annotation_times.append(time_diff)
    
    return annotation_times


def generate_report(all_data):
    print("ANNOTATION REPORT")
    print("COMPSCI 4NL3 - Group 37")
    
    # Per-annotator statistics
    annotator_stats = {}
    all_times = []
    total_instances = 0
    total_valid_times = 0
    total_breaks = 0
    
    for annotator, data in all_data.items():
        annotations = data['annotations']
        num_annotations = len(annotations)
        total_instances += num_annotations
        
        # Calculate annotation times
        times = calculate_annotation_times(annotations)
        all_times.extend(times)
        
        # Calculate breaks (gaps > 10 minutes)
        sorted_annotations = sorted(annotations, key=lambda x: parse_timestamp(x['timestamp']))
        breaks = 0
        for i in range(1, len(sorted_annotations)):
            prev_time = parse_timestamp(sorted_annotations[i-1]['timestamp'])
            curr_time = parse_timestamp(sorted_annotations[i]['timestamp'])
            time_diff_minutes = (curr_time - prev_time).total_seconds() / 60
            if time_diff_minutes > BREAK_THRESHOLD_MINUTES:
                breaks += 1
        
        total_breaks += breaks
        total_valid_times += len(times)
        
        # Calculate average for this annotator
        avg_time = sum(times) / len(times) if times else 0
        
        annotator_stats[annotator] = {
            'num_annotations': num_annotations,
            'valid_times': len(times),
            'breaks': breaks,
            'avg_time_seconds': avg_time,
            'avg_time_minutes': avg_time / 60,
            'total_time_hours': sum(times) / 3600 if times else 0
        }
    
    # Overall statistics
    overall_avg_seconds = sum(all_times) / len(all_times) if all_times else 0
    overall_avg_minutes = overall_avg_seconds / 60
    
    print("\nPER-ANNOTATOR STATISTICS:")
    
    for annotator, stats in sorted(annotator_stats.items()):
        print(f"\n{annotator.upper()}:")
        print(f"  Total instances annotated:     {stats['num_annotations']}")
        print(f"  Valid timing pairs:            {stats['valid_times']}")
        print(f"  Breaks detected (>10 min):     {stats['breaks']}")
        print(f"  Average time per instance:     {stats['avg_time_minutes']:.2f} minutes ({stats['avg_time_seconds']:.1f} seconds)")
        print(f"  Total annotation time:         {stats['total_time_hours']:.2f} hours")
    
    print("\nOVERALL STATISTICS:")
    print(f"Total annotators:                {len(all_data)}")
    print(f"Total instances annotated:       {total_instances}")
    print(f"Valid timing pairs:              {total_valid_times}")
    print(f"Total breaks detected:           {total_breaks}")
    print("\nAverage time per instance (across all annotators):")
    print(f"  {overall_avg_minutes:.2f} minutes")
    print(f"  ({overall_avg_seconds:.1f} seconds)")
    
    # Category distribution
    print("\nCATEGORY DISTRIBUTION:")
    
    category_counts = defaultdict(int)
    for annotator, data in all_data.items():
        for ann in data['annotations']:
            category_counts[ann['category_name']] += 1
    
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_instances) * 100
        print(f"  {category:30s}  {count:4d}  ({percentage:5.1f}%)")
    
    # Time analysis
    if all_times:
        print("\nTIME DISTRIBUTION ANALYSIS:")

        times_minutes = [t / 60 for t in all_times]
        times_minutes.sort()
        
        # Calculate percentiles
        def percentile(data, p):
            index = int(len(data) * p / 100)
            return data[min(index, len(data) - 1)]
        
        print(f"  Minimum time:      {min(times_minutes):.2f} minutes")
        print(f"  25th percentile:   {percentile(times_minutes, 25):.2f} minutes")
        print(f"  Median (50th):     {percentile(times_minutes, 50):.2f} minutes")
        print(f"  75th percentile:   {percentile(times_minutes, 75):.2f} minutes")
        print(f"  90th percentile:   {percentile(times_minutes, 90):.2f} minutes")
        print(f"  Maximum time:      {max(times_minutes):.2f} minutes")

def main():
    print("Loading annotation data...")
    
    all_data = load_all_annotations()
    if all_data is None:
        return
    
    print(f"Found {len(all_data)} annotator(s)\n")
    
    generate_report(all_data)


if __name__ == "__main__":
    main()
