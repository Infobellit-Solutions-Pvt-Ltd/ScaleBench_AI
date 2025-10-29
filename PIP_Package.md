# ScaleBench_AI: LLM Inference Benchmarking Tool by Infobell IT

scalebench is a CLI-based tool designed to benchmark LLM (Large Language Model) inference endpoints. It helps evaluate performance using real-world prompts with configurable parameters and visualized results.

## Features
- Easy-to-use CLI interface
- Benchmark LLM inference across multiple Inference Servers
- Measures key performance metrics: latency, throughput, and TTFT (Time to First Token)
- Support for varying input and output token lengths
- Simulate concurrent users to test scalability
- Determine the optimal number of concurrent users the server can handle while maintaining: TTFT < 2000 ms and Token latency < 200 ms
- Detailed logging and progress tracking

## Supported Inference Servers
  - TGI
  - vLLM
  - Ollama
  - Llamacpp
  - NIMS
  - SGLang
  
## Performance metrics:

The performance metrics captured for varying input and output tokens and parallel users while running the benchmark includes 
- Latency (ms/token)
- TTFT(ms)
- Throughput(tokens/sec) 

## Installation

You can install scalebench using pip:

```bash
pip install scalebench
```

Alternatively, you can install from source:

```bash
git clone https://github.com/Infobellit-Solutions-Pvt-Ltd/ScaleBench_AI
cd scalebench
pip install -e .
```

## Usage

scalebench provides a simple CLI interface for running LLM Inference benchmarks.

Below are the steps to run a sample test, assuming the generation endpoint is active.

### 1. Download the Dataset and create a default `config.json`

Before running a benchmark, you need to download and filter the dataset:

```bash
scalebench dataprep
```
This command will:
- Download the filtered ShareGPT dataset from Huggingface
- Create a default `config.json` file in your working directory


### 2. Configure the Benchmark

Edit the generated `config.json` file to match your LLM server configuration. Below is a sample:

```json
{
    "_comment": "scalebench Configuration",
    "out_dir": "Results",
    "base_url": "http://localhost:8000/v1/completions",
    "tokenizer_path": "/path/to/tokenizer/",
    "inference_server": "vLLM",
    "model": "/model",
    "random_prompt": true,
    "max_requests": 1,
    "user_counts": [
        10
    ],
    "increment_user": [
        100
    ],
    "input_tokens": [
        32
    ],
    "output_tokens": [
        256
    ]
}

```
**Note:** Modify base_url, tokenizer_path, model, and other fields according to your LLM deployment.

#### Prompt Configuration Modes

scalebench supports two input modes depending on your test requirements:

##### 1. Fixed Input Tokens

If you want to run the benchmark with a **fixed number of input tokens**:

* Set `"random_prompt": false`
* Define both `input_tokens` and `output_tokens` explicitly

##### 2. Random Input Length

If you prefer using **randomized prompts** from the dataset:

* Set `"random_prompt": true`
* Provide only `output_tokens` — scalebench will choose random input lengths from the dataset

#### User Load Configuration (For `optimaluserrun`)

To perform optimal user benchmarking:

* Use `user_counts` to set the **initial number of concurrent users**
* Use `increment_user` to define how many users to add per step

Example:

```json
"user_counts": [10],
"increment_user": [100]
```

In this case, the benchmark will start with 10 users and increase by 100 in each iteration until performance thresholds are hit.

#### Tokenizer Configuration

scalebench allows two ways to configure the tokenizer used for benchmarking:

##### Option 1: Use a Custom Tokenizer

Set the `TOKENIZER` environment variable to the path of your desired tokenizer.

##### Option 2: Use Default Fallback

If `TOKENIZER` is not set or is empty, scalebench falls back to a built-in default tokenizer:

This ensures the tool remains functional, but the fallback tokenizer may not align with your model's behavior. Use it only for testing or when no tokenizer is specified.

---

> 💡 **Best Practice:** Always specify the correct tokenizer that matches your LLM model for accurate benchmarking results.

---

Use these combinations as per your requirement to effectively benchmark your LLM endpoint.


### 3. Run the Benchmark

**Option A: Standard Benchmarking**

Use the start command to run a basic benchmark:

```bash
scalebench start --config path/to/config.json
```

**Option B: Optimal User Load Benchmarking**

To find the optimal number of concurrent users for your LLM endpoint:

```bash
scalebench optimaluserrun --config path/to/config.json
```

### 4. Plot the Results

Visualize the benchmark results using the built-in plotting tool:

```bash
scalebench plot --results-dir path/to/your/results_dir --config path/to/config.json
```

**Note:** The `--config` parameter is optional and defaults to `config.json`. It is used to read the `random_prompt` setting from the configuration file.

## Output

scalebench will create a `results` directory (or the directory specified in `out_dir`) containing:

- CSV files with raw benchmark data
- Averaged results for each combination of users, input tokens, and output tokens
- Log files for each Locust run

## Analyzing Results

After the benchmark completes, you can find CSV files in the output directory. These files contain information about latency, throughput, and TTFT for each test configuration.
