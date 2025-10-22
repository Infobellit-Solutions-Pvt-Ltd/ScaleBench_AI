import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

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
        column_names = ["input_tokens", "output_tokens", "throughput(tokens/second)", "latency(ms)", "TTFT(ms)", "latency_per_token(ms/token)"]
        rows = read_csv(input_csv_filename)

        if not rows:
            logging.error(f"Input file is empty: {input_csv_filename}")
            sys.exit(1)

        header = rows[0]

        try:
            column_indices = [header.index(column) for column in column_names]
        except ValueError as e:
            logging.error(f"Error finding column indices: {str(e)}. Check if all required columns are present.")
            sys.exit(1)

        empty_line_indices = [i for i, row in enumerate(rows) if not any(row)]
        if not empty_line_indices or empty_line_indices[-1] != len(rows) - 1:
            rows.append([''] * len(rows[0]))
        empty_line_indices = empty_line_indices + [len(rows)]

        try:
            with open(output_csv_filename, mode='w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(column_names)

                for i in range(len(empty_line_indices)):
                    start_index = 1 if i == 0 else empty_line_indices[i - 1] + 1
                    end_index = empty_line_indices[i]
                    group_rows = rows[start_index:end_index]
                    average = calculate_average(group_rows, column_indices)

                    if len(average) > 1:
                        writer.writerow(average) 

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
                writer.writerow(["user_counts", "input_tokens", "output_tokens"] + column_names)

                for i, token in enumerate(output_tokens):
                    start = i * group_size
                    end = start + group_size
                    group_rows = data_rows[start:end]

                    if not group_rows:
                        continue

                    avg_values = calculate_average(group_rows, column_indices)
                    writer.writerow([user_count, input_tokens, token] + avg_values)

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
    args = parser.parse_args()

    # Read config.json from the project root
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
