import subprocess
import json
import os
import shutil
import pandas as pd
from pathlib import Path
import keyboard  
import argparse
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the threshold for TTFT and latency
VALIDATION_CRITERION = {"TTFT": 2000, "latency_per_token": 200}

def validate_criterion(ttft, latency_per_token):
    """
    Validate the results against the defined threshold.
    """
    return (
        ttft <= VALIDATION_CRITERION["TTFT"]
        and latency_per_token <= VALIDATION_CRITERION["latency_per_token"]
        #and latency <= VALIDATION_CRITERION["latency"]
    )

def run_benchmark(config_file):
    """
    Run the benchmark using subprocess and the provided configuration file.
    """
    try:
        start_time = time.time()
        result = subprocess.run(
            ['scalebench', 'start', '--config', config_file],
            capture_output=True,
            text=True
        )
        run_time = time.time() - start_time
        return result, run_time
    except Exception as e:
        logging.error(f"Error running benchmark: {e}")
        return None, None

def copy_avg_response(config_file, result_dir, user_count):
    """
    Copy and rename avg_32_input_tokens.csv to the results folder based on the user count.
    """
    try:
        with open(config_file, "r") as file:
            config = json.load(file)

        out_dir = config.get("out_dir")
        if not out_dir:
            print("Error: 'out_dir' not specified in the config file.")
            return

        out_dir_path = Path(out_dir)

        if config.get("random_prompt"):
            user_folder = out_dir_path / f"{user_count}_User"
            avg_response_path = user_folder / "avg_Response.csv"
            if avg_response_path.exists():
                new_name = f"avg_Response_User{user_count}.csv"
                shutil.copy(avg_response_path, result_dir / new_name)
                print(f"Copied and renamed avg_Response.csv to {result_dir / new_name}")
            else:
                print(f"avg_Response.csv not found in {user_folder}")

        else:
            user_folder = out_dir_path / f"{user_count}_User"
            input_token = config.get("input_tokens")[0]
            avg_response_path = user_folder / f"avg_{input_token}_input_tokens.csv"
            if avg_response_path.exists():
                new_name = f"avg_{input_token}_input_token_User{user_count}.csv"
                shutil.copy(avg_response_path, result_dir / new_name)
                print(f"Copied and renamed avg_32_input_tokens.csv to {result_dir / new_name}")
            else:
                print(f"avg_32_input_tokens.csv not found in {user_folder}")

    except Exception as e:
        print(f"Unexpected error: {e}")

def extract_metrics_from_avg_response(config_file, result_dir, user_count):
    """
    Extract TTFT, latency, and latency_per_token values from the user's avg_Response.csv file.
    Reads the last row (most recent average) from the file.
    """
    try:
        with open(config_file, "r") as file:
            config = json.load(file)

        out_dir = config.get("out_dir")
        if not out_dir:
            print("Error: 'out_dir' not specified in the config file.")
            return None, None, None, None, None

        out_dir_path = Path(out_dir)
        
        if config.get("random_prompt"):
            # Read directly from the user's directory
            user_folder = out_dir_path / f"{user_count}_User"
            avg_response_path = user_folder / "avg_Response.csv"
        else:
            user_folder = out_dir_path / f"{user_count}_User"
            input_token = config.get("input_tokens", [32])[0]
            avg_response_path = user_folder / f"avg_{input_token}_input_tokens.csv"
        
        if avg_response_path.exists():
            df = pd.read_csv(avg_response_path)
            # Get the last row (most recent average)
            last_row_idx = len(df) - 1
            ttft = df["TTFT(ms)"].iloc[last_row_idx]
            latency_per_token = df["latency_per_token(ms/token)"].iloc[last_row_idx]
            latency = df["latency(ms)"].iloc[last_row_idx]
            throughput = df["throughput(tokens/second)"].iloc[last_row_idx]
            total_throughput = df["total_throughput(tokens/second)"].iloc[last_row_idx]
            return ttft, latency_per_token, latency, throughput, total_throughput
        else:
            print(f"{avg_response_path} not found")
            return None, None, None, None, None

    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return None, None, None, None, None

def binary_search_user_count(config_file, low, high, result_dir):
    """
    Perform binary search to refine the optimal user count after a failed validation.
    """
    while low < high:
        mid = (low + high) // 2

        # Update config with the mid user count
        with open(config_file, 'r') as file:
            config = json.load(file)

        config["user_counts"] = [mid]
        with open(config_file, 'w') as file:
            json.dump(config, file, indent=4)

        # Run the benchmark
        result, runtime = run_benchmark(config_file)
        if result is None:
            print("Benchmark run failed during binary search. Exiting.")
            break

        # Copy and extract metrics
        copy_avg_response(config_file, result_dir, mid)
        metrics = extract_metrics_from_avg_response(config_file, result_dir, mid)
        if any(metric is None for metric in metrics):
            print("Error extracting metrics during binary search. Exiting.")
            break

        ttft, latency_per_token, latency, throughput, total_throughput = metrics
        print(f"Binary Search - User Count: {mid}, TTFT: {ttft} ms, Latency: {latency} ms, "
              f"Latency per Token: {latency_per_token} ms/token, Throughput: {throughput} tokens/second, Total Throughput: {total_throughput} tokens/second")

        if validate_criterion(ttft, latency_per_token):
            # Threshold met; continue searching upward
            low = mid + 1
        else:
            # Threshold not met; search downward
            high = mid

    # Return the highest user count that met the criteria
    return low - 1
  
def run_benchmark_with_incremental_requests(config_file, optimal_user_count, result_dir):
    """
    Run the benchmark for 10 iterations at the optimal user count.
    If validation fails during the iterations, reduce the user count and retry.
    """
    print(f"Starting continuous benchmark with {optimal_user_count} users...")
    max_iterations = 10
    
    try:
        current_user_count = optimal_user_count
        iteration = 0
        failed_validations = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration}/{max_iterations} with {current_user_count} users ---")
            
            # Update config with current user count
            update_config(config_file, current_user_count)

            # Run the benchmark
            result, benchmark_time = run_benchmark(config_file)
            if result is None:
                print("Benchmark run failed. Stopping continuous benchmark.")
                break

            # Copy and extract metrics
            copy_avg_response(config_file, result_dir, current_user_count)

            metrics = extract_metrics_from_avg_response(config_file, result_dir, current_user_count)
            if not all(metric is not None for metric in metrics):
                print("Error extracting metrics. Exiting.")
                break
            
            ttft, latency_per_token, latency, throughput, total_throughput = metrics
            print(f"Continuous Run - User Count: {current_user_count}, TTFT: {ttft} ms, "
                  f"Latency: {latency} ms, Latency per Token: {latency_per_token} ms/token, "
                  f"Throughput: {throughput} tokens/second, Total Throughput: {total_throughput} tokens/second")

            # Check if performance is still acceptable
            if not validate_criterion(ttft, latency_per_token):
                failed_validations += 1
                print(f"Validation failed at iteration {iteration} for user count {current_user_count}. "
                      f"Total failures: {failed_validations}/10")
                
                # If 2 or more failures, reduce user count and reset
                if failed_validations >= 2:
                    print(f"\nValidation failed {failed_validations} times. Reducing user count and retrying...")
                    current_user_count -= 1
                    if current_user_count <= 0:
                        print("User count reduced to 0. Stopping benchmark.")
                        break
                    print(f"Reducing user count to {current_user_count} and resetting iterations...")
                    iteration = 0  # Reset iteration counter to start from 1 again
                    failed_validations = 0  # Reset failure counter
                    continue
                
                # If only 1 failure, continue to next iteration
                continue

            # If validation passed, save summary report and send data
            summary_report_path = generate_summary_report(
                config_file, current_user_count,
                [ttft, latency_per_token, latency, throughput, total_throughput],
                benchmark_time
            )
        
        # Check if we completed all 10 iterations successfully
        if iteration == max_iterations:
            print(f"\nAll {max_iterations} iterations completed successfully with {current_user_count} users.")
            print(f"So the optimal user count is {current_user_count}.")

    except KeyboardInterrupt:
        print("\nContinuous benchmarking stopped by user.")

def generate_summary_report(config_file, optimal_user_count, metrics, benchmark_time):
    """
    Generate a summary report with the benchmark metrics.
    """
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
        out_dir = config["out_dir"]
        summary_report_path1 = Path(out_dir) / "Results" / "summary_report.csv"
        summary_report_path2 = Path(out_dir) / "Results" / "summary_report.txt"
        summary_report_path1.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "User Counts": [int(optimal_user_count)],
            "Total Throughput(tokens/second)": [round(metrics[4], 3)],
            "TTFT (ms)": [int(metrics[0])],
            "Latency per Token(ms/token)": [round(metrics[1], 3)],
            "Throughput(tokens/second)": [round(metrics[3], 3)],
            "Latency(ms)": [round(metrics[2], 3)]
        }

        df = pd.DataFrame(report_data)
        df.to_csv(summary_report_path1, index=False)
        
        # Write text file manually since pandas doesn't have to_txt()
        with open(summary_report_path2, 'w') as txt_file:
            # Header
            txt_file.write("=" * 150 + "\n")
            txt_file.write(f"BENCHMARK RESULTS - RANDOM PROMPT MODE\n")
            txt_file.write("=" * 150 + "\n\n")
            
            # Column headers
            txt_file.write(f"{'User Counts':<12} {'Throughput (tokens/second)':<25} {'Latency (ms)':<15} {'TTFT (ms)':<15} {'Latency/Token (ms/token)':<25} {'Total Throughput (tokens/second)':<30}\n")
            txt_file.write("-" * 150 + "\n")
            
            # Data row - format matches write_to_txt function
            txt_file.write(f"{int(optimal_user_count):<12} {round(metrics[3], 3):<25} {round(metrics[2], 3):<15} {round(metrics[0], 2):<15} {round(metrics[1], 2):<25} {round(metrics[4], 2):<30}\n")
            
            # Footer
            txt_file.write("\n" + "=" * 150 + "\n")
            txt_file.write(f"Total Records: 1\n")
            txt_file.write("=" * 150 + "\n")
        
        print(f"Summary report generated at {summary_report_path1} and {summary_report_path2}")
        return summary_report_path1, summary_report_path2
    
    except Exception as e:
        print(f"Error generating summary report: {e}")
        return None, None

def adjust_user_count(config_file, result_dir):
    """
    Adjust the user count and run the benchmark until the threshold is met or the optimal user count is found.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)

    user_count = config.get("user_counts")[0]  
    previous_user_count = 0
    increment = config.get("increment_user")[0]
    out_dir = config.get("out_dir")
    if not out_dir:
        logging.error("Error: 'out_dir' not specified in the config file.")
        return
    
    optimal_user_count = 0
    final_metrics = None
    benchmark_time = 0

    while True:
        # Update config with the current user count
        config["user_counts"] = [user_count]
        with open(config_file, 'w') as file:
            json.dump(config, file, indent=4)

        # Run the benchmark
        result, benchmark_time = run_benchmark(config_file)
        if result is None:
            logging.error("Benchmark run failed. Exiting.")
            break

        # Copy and extract metrics
        copy_avg_response(config_file, result_dir, user_count)
        metrics = extract_metrics_from_avg_response(config_file, result_dir, user_count)
        if any(metric is None for metric in metrics):
            print("Error extracting metrics. Exiting.")
            break

        ttft, latency_per_token, latency, throughput, total_throughput = metrics
        print(f"User Count: {user_count}, TTFT: {ttft} ms, Latency: {latency} ms, "
              f"Latency per Token: {latency_per_token} ms/token, Throughput: {throughput} tokens/second, "
              f"Total Throughput: {total_throughput} tokens/second")

        if validate_criterion(ttft, latency_per_token):
            print(f"Threshold met for {user_count} users.")
            previous_user_count = user_count
            optimal_user_count = user_count
            final_metrics = metrics  # Store the metrics for the last valid run
            user_count += increment
        else:
            print(f"Threshold not met for {user_count} users.")
            optimal_user_count = binary_search_user_count(config_file, previous_user_count, user_count, result_dir)
            final_metrics = extract_metrics_from_avg_response(config_file, result_dir, optimal_user_count)
            break
    
    # Generate the summary report with the final metrics
    if final_metrics and optimal_user_count > 0:
        generate_summary_report(config_file, optimal_user_count, final_metrics, benchmark_time)
        return optimal_user_count  # Ensure a valid return
    else:
        print("No valid optimal user count found. Exiting.")
        return None  # Return None explicitly

def update_config(config_file, optimal_user_count):
    """
    Update the config file with the optimal user count.
    """
    with open(config_file, "r") as file:
        config = json.load(file)
    
    if optimal_user_count is not None and optimal_user_count > 0:
        config["optimal_user_count"] = optimal_user_count
        config["user_counts"] = [optimal_user_count]  # Ensure this is a valid list
        config["max_requests"] = 1
    else:
        print("Warning: No valid optimal user count found. Keeping existing configuration.")
    
    with open(config_file, "w") as file:
        json.dump(config, file, indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark automation script")
    parser.add_argument("config_file", type=str, help="Path to the benchmark configuration file")
    args = parser.parse_args()

    # Read the config file to get the output directory
    # with open(args.config_file, "r") as file:
    #     config = json.load(file)
    
    # out_dir = config.get("out_dir")
    # if not out_dir:
    #     print("Error: 'out_dir' not specified in the config file.")
        
    
    # result_dir = Path(out_dir) / "Results"
    # result_dir.mkdir(parents=True, exist_ok=True)
    
    # optimal_user_count = adjust_user_count(args.config_file, result_dir)    
    
    # if optimal_user_count is not None:
    #     run_benchmark_with_incremental_requests(args.config_file, optimal_user_count, result_dir)

