import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def process_csv_files(directory_path, random_prompt):
    data = []

    if random_prompt==True:
        csv_files = sorted(directory_path.glob('avg_Response.csv'))

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                user_counts = row['user_counts']
                input_tokens = row['input_tokens']
                output_token = row['output_tokens']
                throughput = row['throughput(tokens/second)']
                latency = row['latency(ms)']
                ttft = row['TTFT(ms)']
                token_latency = row['latency_per_token(ms/token)']
                total_throughput = row['total_throughput(tokens/second)']
                data.append((user_counts, input_tokens, output_token, throughput, latency, ttft, token_latency, total_throughput))
                
    else:
        csv_files = sorted(directory_path.glob('avg_*_input_tokens.csv'), 
                            key=lambda x: int(''.join(filter(str.isdigit, x.stem))))

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                user_counts = row['user_counts']
                input_tokens = row['input_tokens']
                output_tokens = row['output_tokens']
                throughput = row['throughput(tokens/second)']
                latency = row['latency(ms)']
                TTFT = row['TTFT(ms)']
                latency_per_token = row['latency_per_token(ms/token)']
                total_throughput = row['total_throughput(tokens/second)']
                data.append((user_counts, input_tokens, output_tokens, throughput, latency, TTFT, latency_per_token, total_throughput))

    return data

def write_to_csv(data, output_file, random_prompt):
    with open(output_file, 'w') as f:   
        # Use same format for both random_prompt and fixed prompt
        f.write('user_counts,input_tokens,output_tokens,throughput(tokens/second),latency(ms),TTFT(ms),latency_per_token(ms/token),total_throughput(tokens/second)\n')
        for value in data:
            f.write(f'{value[0]},{value[1]},{value[2]},{value[3]},{value[4]},{value[5]},{value[6]},{value[7]}\n')

def write_to_txt(data, output_file, random_prompt):
    with open(output_file, 'w') as f:   
        mode_label = "RANDOM PROMPT MODE" if random_prompt else "FIXED PROMPT MODE"
        f.write("=" * 100 + "\n")
        f.write(f"BENCHMARK RESULTS - {mode_label}\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'User Counts':<12} {'Input Tokens':<15} {'Output Tokens':<15} {'Throughput (tokens/second)':<25} {'Latency (ms)':<15} {'TTFT (ms)':<15} {'Latency/Token (ms/token)':<25} {'Total Throughput (tokens/second)':<30}\n")
        f.write("-" * 180 + "\n")
        for value in data:
            f.write(f"{value[0]:<12} {round(float(value[1]), 2):<15} {round(float(value[2]), 2):<15} {round(float(value[3]), 2):<25} {round(float(value[4]), 2):<15} {round(float(value[5]), 2):<15} {round(float(value[6]), 2):<25} {round(float(value[7]), 2):<30}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Total Records: {len(data)}\n")
        f.write("=" * 100 + "\n")

def plot_line_chart(data, x_label, y_label, title, output_file):
    plt.figure(figsize=(12, 8))

    x_values = data[x_label]
    y_values = data[y_label]

    # Get unique x values and calculate averages for each x value
    unique_x_values = sorted(set(x_values))
    x_ticks_positions = np.arange(len(unique_x_values))
    
    # Calculate average y values for each unique x value
    avg_y_values = []
    for x_val in unique_x_values:
        mask = x_values == x_val
        avg_y = y_values[mask].mean()
        avg_y_values.append(avg_y)

    plt.plot(x_ticks_positions, avg_y_values, marker='o', linewidth=2, markersize=8, label=f'Average {y_label}')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Add value annotations
    for i, (x_pos, y_val) in enumerate(zip(x_ticks_positions, avg_y_values)):
        plt.annotate(f'{y_val:.2f}', xy=(x_pos, y_val), ha='center', va='bottom', fontsize=10)

    plt.xticks(x_ticks_positions, [str(num) for num in unique_x_values])
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_benchmark_results(base_directory, random_prompt=False):
    base_directory = Path(base_directory)
    csv_output_file = base_directory / 'aggregated_data.csv'
    txt_output_file = base_directory / 'aggregated_data.txt'
    
    data = []
    # Sort directories by user count to ensure proper sequence (1, 3, 10)
    directories = [d for d in base_directory.iterdir() if d.is_dir() and "_User" in d.name]
    directories.sort(key=lambda x: int(''.join(filter(str.isdigit, x.name))))
    
    for directory in directories:
        directory_data = process_csv_files(directory, random_prompt)
        data.extend(directory_data)   

    # Save data in both CSV and TXT formats
    write_to_csv(data, csv_output_file, random_prompt)
    write_to_txt(data, txt_output_file, random_prompt)
    
    print(f"Aggregated data has been written to:")
    print(f"  CSV format: {csv_output_file}")
    print(f"  TXT format: {txt_output_file}")

    df = pd.read_csv(csv_output_file)

    plot_line_chart(df[['user_counts', 'latency_per_token(ms/token)']], 
                    'user_counts', 'latency_per_token(ms/token)', 
                    'User Counts vs Token Latency', 
                    base_directory / 'token_latency_plot.png')

    plot_line_chart(df[['user_counts', 'throughput(tokens/second)']], 
                    'user_counts', 'throughput(tokens/second)', 
                    'User Counts vs Throughput', 
                    base_directory / 'throughput_plot.png')

    plot_line_chart(df[['user_counts', 'TTFT(ms)']], 
                    'user_counts', 'TTFT(ms)', 
                    'User Counts vs Time to First Token', 
                    base_directory / 'ttft_plot.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV files and generate plots.')
    parser.add_argument('base_directory', type=str, help='The base directory containing the result directories.')
    parser.add_argument('--random_prompt', type=bool, default=False, help='Use random prompts (default: False)')
    args = parser.parse_args()
    plot_benchmark_results(args.base_directory, args.random_prompt)