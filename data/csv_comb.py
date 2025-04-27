import glob
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading
import csv

# Set the directory where your CSVs are stored
csv_dir = '/Users/namanbajpai/lpm/data/temp_batches'  # CHANGE this to your folder path
output_file = 'combined_output.csv'

# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

# Thread-safe file writing lock
lock = threading.Lock()

# Function to append one CSV file's content to the output file
def append_csv_file(file_path, progress_bar, is_first_file):
    with open(file_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Read the header
        rows = list(reader)    # Read all rows
        
        with lock:
            mode = 'w' if is_first_file else 'a'
            with open(output_file, mode, newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                if is_first_file:
                    writer.writerow(header)  # Write header only for the first file
                for row in rows:
                    writer.writerow(row)
        
        progress_bar.update(1)

# Main processing with multi-threading and progress bar
def combine_csvs():
    # Clear output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
    
    if not csv_files:
        print("⚠️ No CSV files found to combine.")
        return
    
    # Initialize progress bar
    with tqdm(total=len(csv_files), desc="Combining CSVs", unit="file") as pbar:
        # Use ThreadPoolExecutor for multi-threading
        with ThreadPoolExecutor() as executor:
            # Submit tasks, marking the first file
            futures = [
                executor.submit(append_csv_file, f, pbar, i == 0)
                for i, f in enumerate(csv_files)
            ]
            # Wait for all tasks to complete
            for future in futures:
                future.result()
    
    print(f"✅ Combined {len(csv_files)} CSV files into '{output_file}'")

if __name__ == "__main__":
    combine_csvs()