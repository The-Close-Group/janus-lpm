import requests
import pandas as pd
import json
import time
import math
from typing import List
import os
import logging
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('census_scraper.log'),
        logging.StreamHandler()
    ]
)

# Configuration
BASE_URL = "https://api.census.gov/data/2021/acs/acs5"
GEOGRAPHY = "zip code tabulation area:*"
BATCH_SIZE = 49  # Max 50 including NAME, no API key needed
MAX_REQUESTS = 500  # Census API limit
REQUEST_DELAY = 0  # Delay to avoid rate limiting
MAX_RETRIES = 3
OUTPUT_FILE = "census_data_zcta.csv"
TEMP_DIR = "temp_batches"
MAX_VARIABLES = 24500  # Limit to 500 requests * 49 variables
MAX_WORKERS = 10  # Number of concurrent requests

def setup_session():
    """Create a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    return session

def load_variables(json_file: str, max_variables: int = None) -> List[str]:
    """
    Load variable names from the provided JSON file.
    
    Args:
        json_file (str): Path to the JSON file.
        max_variables (int): Maximum number of variables to load.
        
    Returns:
        List[str]: List of variable names.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'variables' not in data:
            logging.error("Error: 'variables' key not found in JSON.")
            return []
        variables = [
            key for key in data['variables'].keys()
            if key not in ['for', 'in', 'ucgid'] and not key.startswith('GEO')
        ]
        if max_variables:
            variables = variables[:max_variables]
        logging.info(f"Loaded {len(variables)} variables: {variables[:5]}...")
        return variables
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        return []
    except FileNotFoundError:
        logging.error(f"Error: File {json_file} not found.")
        return []
    except Exception as e:
        logging.error(f"Unexpected error loading JSON: {e}")
        return []

def split_into_batches(items: List[str], batch_size: int) -> List[List[str]]:
    """Split a list into batches of specified size."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def fetch_census_data(variables: List[str], geography: str, session: requests.Session) -> List:
    """
    Fetch data from the Census API without an API key.
    
    Args:
        variables (List[str]): List of variable names.
        geography (str): Geography specification.
        session (requests.Session): Requests session.
        
    Returns:
        List: API response data.
    """
    get_params = ",".join(["NAME"] + variables)
    encoded_geography = quote(geography)
    url = f"{BASE_URL}?get={get_params}&for={encoded_geography}"
    logging.info(f"Requesting: {url[:100]}...")
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        try:
            data = response.json()
            logging.info(f"Successfully fetched data for {len(variables)} variables")
            return data
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}. Response: {response.text[:1000]}")
            return []
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error for variables {variables[:5]}...: {e}. Response: {response.text[:1000]}")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error for variables {variables[:5]}...: {e}")
        return []

def process_batch_to_dataframe(batch_data: List, variables: List[str]) -> pd.DataFrame:
    """Convert API response to a pandas DataFrame."""
    if not batch_data or len(batch_data) < 2:
        logging.warning("No data in batch response")
        return pd.DataFrame()
    headers = batch_data[0]
    data_rows = batch_data[1:]
    try:
        df = pd.DataFrame(data_rows, columns=headers)
        expected_columns = ["NAME"] + variables + [col for col in df.columns if col not in ["NAME"] + variables]
        df = df[expected_columns]
        logging.info(f"Processed batch into DataFrame with {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error processing batch to DataFrame: {e}")
        return pd.DataFrame()

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by converting numeric columns and handling missing values."""
    try:
        for col in df.columns:
            if col not in ['NAME', 'zip code tabulation area']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        df = df.where(pd.notnull(df), None)
        logging.info("DataFrame cleaned")
        return df
    except Exception as e:
        logging.error(f"Error cleaning DataFrame: {e}")
        return df

def save_batch(df: pd.DataFrame, batch_idx: int, temp_dir: str):
    """Save a batch DataFrame to a temporary CSV."""
    try:
        os.makedirs(temp_dir, exist_ok=True)
        batch_file = os.path.join(temp_dir, f"batch_{batch_idx}.csv")
        df.to_csv(batch_file, index=False, encoding='utf-8')
        logging.info(f"Saved batch {batch_idx} to {batch_file}")
        return batch_file
    except Exception as e:
        logging.error(f"Error saving batch {batch_idx}: {e}")
        return None

def process_batch(batch_idx, batch, geography, session, temp_dir):
    """Fetch, process, and save a batch."""
    logging.info(f"Fetching batch {batch_idx} with {len(batch)} variables...")
    batch_data = fetch_census_data(batch, geography, session)
    if batch_data:
        df = process_batch_to_dataframe(batch_data, batch)
        if not df.empty:
            batch_file = save_batch(df, batch_idx, temp_dir)
            return batch_file
    return None

def merge_batches(temp_dir: str, output_file: str):
    """Merge all batch CSVs into a final CSV."""
    try:
        batch_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.csv')]
        if not batch_files:
            logging.error("No batch files found to merge")
            return

        # Read first batch
        merged_df = pd.read_csv(batch_files[0])
        geography_cols = ['NAME', 'zip code tabulation area']
        merge_keys = geography_cols

        # Merge with remaining batches
        for batch_file in batch_files[1:]:
            df = pd.read_csv(batch_file)
            merged_df = merged_df.merge(df, on=merge_keys, how='outer', suffixes=('', '_dup'))
            for col in merged_df.columns:
                if col.endswith('_dup'):
                    merged_df.drop(col, axis=1, inplace=True)

        # Clean and save
        merged_df = clean_dataframe(merged_df)
        merged_df.to_csv(output_file, index=False, encoding='utf-8')
        logging.info(f"Merged data saved to {output_file} with {len(merged_df)} rows and {len(merged_df.columns)} columns")

        # Clean up temporary files
        for batch_file in batch_files:
            os.remove(batch_file)
        os.rmdir(temp_dir)
        logging.info("Cleaned up temporary files")
    except Exception as e:
        logging.error(f"Error merging batches: {e}")

def main(json_file: str):
    """Main function to fetch all variables and save to CSV."""
    logging.info("Starting Census API scraper for ZCTA")

    if not os.path.exists(json_file):
        logging.error(f"Error: {json_file} does not exist.")
        return

    variables = load_variables(json_file, max_variables=MAX_VARIABLES)
    if not variables:
        logging.error("No variables loaded. Exiting.")
        return

    total_requests = math.ceil(len(variables) / BATCH_SIZE)
    if total_requests > MAX_REQUESTS:
        logging.warning(f"Required {total_requests} requests exceed limit of {MAX_REQUESTS}. Using {MAX_VARIABLES} variables.")
        variables = variables[:MAX_VARIABLES]
        total_requests = MAX_REQUESTS

    logging.info(f"Processing {len(variables)} variables in {total_requests} requests.")

    variable_batches = split_into_batches(variables, BATCH_SIZE)
    session = setup_session()
    batch_files = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, batch in enumerate(variable_batches, 1):
            futures.append(executor.submit(process_batch, i, batch, GEOGRAPHY, session, TEMP_DIR))
        
        for future in concurrent.futures.as_completed(futures):
            batch_file = future.result()
            if batch_file:
                batch_files.append(batch_file)

    if not batch_files:
        logging.error("No data retrieved. Exiting.")
        return

    logging.info("Merging batch CSVs...")
    merge_batches(TEMP_DIR, OUTPUT_FILE)

if __name__ == "__main__":
    main("data.json")