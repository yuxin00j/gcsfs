import sys
import argparse

def analyze_logs(file_handle):
    info_hits = 0
    info_misses = 0
    data_hits = 0
    data_misses = 0
    
    for line in file_handle:
        if "InfoCache hit" in line:
            info_hits += 1
        elif "InfoCache miss" in line:
            info_misses += 1
        elif "Multi-process cache hit" in line:
            data_hits += 1
        elif "Multi-process cache miss" in line:
            data_misses += 1

    print("-" * 40)
    print("Cache Analysis Results")
    print("-" * 40)
    
    total_info = info_hits + info_misses
    if total_info > 0:
        print(f"InfoCache Hit Rate: {info_hits / total_info * 100:.2f}% ({info_hits} hits / {total_info} total requests)")
    else:
        print("InfoCache: No requests found.")
        
    total_data = data_hits + data_misses
    if total_data > 0:
        print(f"Data Cache Hit Rate: {data_hits / total_data * 100:.2f}% ({data_hits} hits / {total_data} total requests)")
    else:
        print("Data Cache: No requests found.")

def main():
    parser = argparse.ArgumentParser(description="Filter and analyze cache hit/miss logs from a local file.")
    parser.add_argument(
        "logfile", 
        nargs="?", 
        type=argparse.FileType("r"), 
        default=sys.stdin,
        help="Path to the log file (JSON or raw text) downloaded from GCP. Defaults to standard input."
    )
    
    args = parser.parse_args()
    
    print(f"Analyzing logs from: {args.logfile.name}...")
    analyze_logs(args.logfile)

if __name__ == "__main__":
    main()
