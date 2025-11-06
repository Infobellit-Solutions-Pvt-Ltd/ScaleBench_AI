import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_config(config_path: str) -> dict:
    """Read configuration from JSON file."""
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing config file {config_path}: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading config file {config_path}: {str(e)}")
        sys.exit(1)

def read_csv(filename: str) -> List[List[str]]:
    try:
        with open(filename, 'r', newline='') as file:
            return list(csv.reader(file))
    except FileNotFoundError:
        logging.error(f"Input file not found: {filename}")
        sys.exit(1)
    except PermissionError:
        logging.error(f"Permission denied when trying to read: {filename}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading input file {filename}: {str(e)}")
        sys.exit(1)

def calculate_average(rows: List[List[str]], column_indices: List[int]) -> List[Optional[float]]:
    try:
        values = [[float(row[i]) for i in column_indices] for row in rows if row and all(row)]
        return [sum(col) / len(col) if col else None for col in zip(*values)]
    except ValueError as e:
        logging.error(f"Error calculating average: {str(e)}. Check if all values are numeric.")
        return [None] * len(column_indices)


def calculate_averages(input_csv_filename: str, output_csv_filename: str,
                       output_tokens: List[int], max_requests: int, user_count: int,
                       input_tokens: int = None, random_prompt: bool = False):
    
    if random_prompt==True:
        
        rows = read_csv(input_csv_filename)

        if not rows:
            logging.error(f"Input file is empty: {input_csv_filename}")
            sys.exit(1)

        header = rows[0]
        data_rows = rows[1:]
        
        # Calculate how many rows to average (max_requests * user_count)
        group_size = max_requests * user_count
        
        # Check if we have enough rows
        total_rows = len(data_rows)
        
        logging.info(f"Processing random_prompt mode: total_rows={total_rows}, group_size={group_size}")
        
        if total_rows < group_size:
            logging.warning(f"Only {total_rows} rows available, expected {group_size} rows")
            rows_to_average = data_rows
        else:
            # Take the last group_size rows for averaging
            rows_to_average = data_rows[-group_size:]
        
        logging.info(f"Averaging {len(rows_to_average)} rows")

        # columns to average (including all metrics columns)
        column_names = ["input_tokens", "output_tokens", "throughput(tokens/second)", "latency(ms)", "TTFT(ms)", "latency_per_token(ms/token)"]

        try:
            column_indices = [header.index(column) for column in column_names]
        except ValueError as e:
            logging.error(f"Error finding column indices: {str(e)}. Check if all required columns are present.")
            sys.exit(1)

        file_exists = os.path.exists(output_csv_filename)
        
        try:
            with open(output_csv_filename, mode='a', newline="") as file:
                writer = csv.writer(file)
                
                # Write header only if file is new
                if not file_exists:
                    writer.writerow(["user_counts"] + column_names + ["total_throughput(tokens/second)"])
                
                # Calculate average of the last group_size rows
                average = calculate_average(rows_to_average, column_indices)
                
                # Find throughput index in the average list (third index, 0-indexed = 2)
                throughput_idx = 2
                throughput_avg = average[throughput_idx]
                
                # Calculate total_throughput = throughput * user_count
                total_throughput = throughput_avg * user_count if throughput_avg is not None else None
                
                # Write the single averaged row with user_count and total_throughput
                writer.writerow([int(user_count)] + average + [total_throughput])
                
                logging.info(f"Average calculated and appended to {output_csv_filename}")

        except PermissionError:             
            logging.error(f"Permission denied when trying to write to: {output_csv_filename}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error writing to output file {output_csv_filename}: {str(e)}")
            sys.exit(1)

    else:
        rows = read_csv(input_csv_filename)
        if not rows:
            logging.error(f"Input file is empty: {input_csv_filename}")
            sys.exit(1)

        header = rows[0]
        data_rows = rows[1:]

        # columns to average
        column_names = ["throughput(tokens/second)", "latency(ms)", "TTFT(ms)", "latency_per_token(ms/token)"]

        try:
            column_indices = [header.index(c) for c in column_names]
            output_token_idx = header.index("output_tokens")
        except ValueError as e:
            logging.error(f"Error finding required columns: {str(e)}")
            sys.exit(1)

        group_size = max_requests * user_count
        total_groups = len(output_tokens)
        logging.info(f"Processing {total_groups} groups with {group_size} rows each")

        try:
            with open(output_csv_filename, mode='w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["user_counts", "input_tokens", "output_tokens"] + column_names + ["total_throughput(tokens/second)"])

                for i, token in enumerate(output_tokens):
                    start = i * group_size
                    end = start + group_size
                    group_rows = data_rows[start:end]

                    if not group_rows:
                        continue

                    avg_values = calculate_average(group_rows, column_indices)
                    
                    # Calculate total_throughput = throughput * user_count
                    # Throughput is the first column in column_names (index 0)
                    throughput_avg = avg_values[0]
                    total_throughput = throughput_avg * user_count if throughput_avg is not None else None
                    
                    writer.writerow([int(user_count), input_tokens, token] + avg_values + [total_throughput])

            logging.info(f"Averages successfully written to {output_csv_filename}")

        except Exception as e:
            logging.error(f"Error writing to {output_csv_filename}: {str(e)}")
            sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate averages from Locust results")
    parser.add_argument('--input_csv_filename', required=True, help='Input CSV file path')
    parser.add_argument('--output_csv_filename', required=True, help='Output CSV file path')
    parser.add_argument('--user_count', type=int, required=True, help='Number of users')
    parser.add_argument('--input_tokens', type=int, help='Number of input tokens (extracted from filename if not provided)')
    parser.add_argument('--random_prompt', action='store_true', help='Use random prompts (default: False)')
    parser.add_argument('--config_path', help='Path to config.json file')
    args = parser.parse_args()

    # Read config.json - use provided path or fallback to project root
    if args.config_path:
        config_path = Path(args.config_path)
    else:
        config_path = Path(__file__).parent.parent.parent / "config.json"
    
    if config_path.exists():
        config = read_config(str(config_path))
        output_tokens = config.get('output_tokens', [])
        max_requests = config.get('max_requests', 5)
        user_count = args.user_count  
        random_prompt = args.random_prompt  
        
        input_tokens = args.input_tokens
        if not random_prompt and input_tokens is None:
            # Extract from filename like "32_input_tokens.csv"
            filename = Path(args.input_csv_filename).name
            try:
                input_tokens = int(filename.split('_')[0])
            except (ValueError, IndexError):
                logging.error(f"Could not extract input_tokens from filename: {filename}")
                logging.error("Please provide --input_tokens parameter")
                sys.exit(1)
        
        logging.info(f"Using config from {config_path}: output_tokens={output_tokens}, max_requests={max_requests}, user_count={user_count}, input_tokens={input_tokens}")
        logging.info(f"Using command line arguments: random_prompt={random_prompt}")
    else:
        logging.error(f"Config file not found at {config_path}")
        logging.error("Please ensure config.json exists in the project root directory")
        sys.exit(1)

    calculate_averages(args.input_csv_filename, args.output_csv_filename, output_tokens, max_requests, user_count, input_tokens, random_prompt)

# Example command to run this file:
# python3 metrics_processor.py --input_csv_filename "test_results/test3/1_User/32_input_tokens.csv" --output_csv_filename "test_results/test3/1_User/avg_32_input_tokens.csv" --user_count 1
