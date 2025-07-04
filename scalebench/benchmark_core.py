import os
import subprocess
import logging
from pathlib import Path
from typing import List
from tqdm import tqdm
import signal
import pkg_resources
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScaleBench:
    """Core class for running LLM inference benchmarks."""

    def __init__(
        self,
        output_dir: str,
        api_url: str,
        inference_server: str,
        model_name: str = None,
        max_requests: int = 5,
        user_counts: List[int] = [1],
        input_tokens: List[int] = None,
        output_tokens: List[int] = None,
        dataset_dir: str = "Input_Dataset",
        random_prompt: bool = False,
        tokenizer_path: str = None
    ):
        """Initialize ScaleBench.
        
        Args:
            output_dir: Directory to store benchmark results
            api_url: URL of the inference API
            inference_server: Type of inference server
            model_name: Name of the model to benchmark
            max_requests: Maximum number of requests per user
            user_counts: List of concurrent user counts
            input_tokens: List of input token counts
            output_tokens: List of output token counts
            dataset_dir: Directory containing input datasets
            random_prompt: Whether to use random prompts
            tokenizer_path: Path to tokenizer
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if not api_url:
            raise ValueError("API URL is required")
        if not inference_server:
            raise ValueError("Inference server type is required")
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        if not user_counts:
            raise ValueError("user_counts cannot be empty")
        if any(u < 1 for u in user_counts):
            raise ValueError("All user counts must be positive")
        if not random_prompt:
            if not input_tokens:
                raise ValueError("input_tokens required when not using random prompts")
            if not output_tokens:
                raise ValueError("output_tokens required when not using random prompts")
            if any(t < 1 for t in input_tokens + output_tokens):
                raise ValueError("All token counts must be positive")
        
        self.output_dir = Path(output_dir)
        self.api_url = api_url
        self.inference_server = inference_server
        self.model_name = model_name
        self.max_requests = max_requests
        self.user_counts = user_counts
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.dataset_dir = Path(dataset_dir)
        self.random_prompt = random_prompt
        self.tokenizer_path = tokenizer_path
        
    def run_benchmark(self) -> None:
        """Run the benchmark with configured parameters.
        
        Raises:
            RuntimeError: If benchmark execution fails
            OSError: If directory creation fails
        """
        # Create output directories
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            locust_logs_dir = self.output_dir / "locust_logs"
            locust_logs_dir.mkdir(exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create output directories: {e}")
            
        # Validate dataset directory
        if not self.dataset_dir.exists():
            raise RuntimeError(
                f"Dataset directory {self.dataset_dir} not found. "
                "Run 'scalebench dataprep' first."
            )
        if not any(self.dataset_dir.iterdir()):
            raise RuntimeError(
                f"Dataset directory {self.dataset_dir} is empty. "
                "Run 'scalebench dataprep' to download datasets."
            )
        
        if self.random_prompt:
            logging.info("Using random queries from Dataset.csv")
            total_requests = sum(self.user_counts) * self.max_requests
            logging.info(f"Total requests to be sent: {total_requests}")

            for u in self.user_counts:
                user_dir = self.output_dir / f"{u}_User"
                user_dir.mkdir(exist_ok=True)

                user_file = user_dir / "Response.csv"
                user_file.touch()

                logging.info(f"Running Locust with users={u}")
                self._run_locust(u, output_tokens=self.output_tokens, output_file=user_file, logs_dir=locust_logs_dir)

                self._calculate_average(user_dir=user_dir, random_prompt=self.random_prompt)
               
        else:
            logging.info("Using custom queries from Input_Dataset")
            total_requests = sum(self.user_counts) * self.max_requests * len(self.input_tokens) * len(self.output_tokens)
            logging.info(f"Total requests to be sent: {total_requests}")

            for u in self.user_counts:
                user_dir = self.output_dir / f"{u}_User"
                user_dir.mkdir(exist_ok=True)

                for input_token in self.input_tokens:
                    user_file = user_dir / f"{input_token}_input_tokens.csv"
                    user_file.touch()

                    for output_token in self.output_tokens:
                        logging.info(f"Running Locust with users={u}, input_tokens={input_token}, and output_tokens={output_token}")
                        self._run_locust(u, user_file, locust_logs_dir, input_token, output_token)
                    
                    self._calculate_average(user_dir, input_token)
                    

    def _run_locust(self, users: int, output_file: Path, logs_dir: Path, input_tokens: int = None, output_tokens: int = None) -> None:
        """Run Locust load test with specified parameters.
        
        Args:
            users: Number of concurrent users
            output_file: Path to save test results
            logs_dir: Directory for log files
            input_tokens: Number of input tokens (optional)
            output_tokens: Number of output tokens (optional)
            
        Raises:
            RuntimeError: If Locust process fails or times out
            ValueError: If parameters are invalid
        """
        if users < 1:
            raise ValueError("Number of users must be positive")
        if input_tokens is not None and input_tokens < 1:
            raise ValueError("Input tokens must be positive")
        if output_tokens is not None and output_tokens < 1:
            raise ValueError("Output tokens must be positive")
        
        # Set up environment variables
        env = os.environ.copy()

        if self.inference_server in ["Ollama", "vLLM", "NIMS"]:
            env["MODEL_NAME"] = self.model_name

        if self.random_prompt:
            env.update({
                "MAX_REQUESTS": str(self.max_requests),
                "NUM_USERS": str(users),
                "MAX_NEW_TOKENS": str(output_tokens),
                "API_URL": self.api_url,
                "INFERENCE_SERVER": self.inference_server,
                "INPUT_DATASET": str(self.dataset_dir / "Dataset.csv"),
                "OUTPUT_FILE": str(output_file),
                "RANDOM_PROMPT": "True",
                "TOKENIZER": str(self.tokenizer_path)
            })

            locust_file = pkg_resources.resource_filename('scalebench', 'benchmark_controller.py')
            command = [
                "locust",
                "-f", locust_file,
                "--headless",
                "-H", self.api_url,
                "-u", str(users),
                "-r", str(users)
            ]

            log_file_path = logs_dir / f"locust_log_u{users}.log"
            
            total_requests = users * self.max_requests
            with tqdm(total=total_requests, desc=f"Requests (u={users})", leave=True) as pbar, \
                open(log_file_path, 'w') as log_file:
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
                
                generated_text_count = 0
                for line in iter(process.stdout.readline, ''):
                    log_file.write(line)
                    log_file.flush()
                    
                    if "Generated Text:" in line:
                        generated_text_count += 1
                        if generated_text_count % users == 0:
                            update_amount = users
                            pbar.update(update_amount)

                    if pbar.n >= total_requests:
                        process.terminate()
                        break

                remaining = generated_text_count - pbar.n
                if remaining > 0:
                    pbar.update(remaining)

                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logging.warning("Locust didn't terminate gracefully. Forcing termination.")
                    process.kill()

            if process.returncode != 0 and process.returncode != -signal.SIGTERM.value:
                logging.error(f"Locust command failed with return code {process.returncode}. Check the log file: {log_file_path}")

        else:
            env.update({
                "MAX_REQUESTS": str(self.max_requests),
                "NUM_USERS": str(users),
                "MAX_NEW_TOKENS": str(output_tokens),
                "API_URL": self.api_url,
                "INFERENCE_SERVER": self.inference_server,
                "INPUT_DATASET": str(self.dataset_dir / f"Dataset_{input_tokens}.csv"),
                "OUTPUT_FILE": str(output_file),
                "TOKENIZER": str(self.tokenizer_path)
            })

            locust_file = pkg_resources.resource_filename('scalebench', 'benchmark_controller.py')
            command = [
                "locust",
                "-f", locust_file,
                "--headless",
                "-H", self.api_url,
                "-u", str(users),
                "-r", str(users)
            ]

            log_file_path = logs_dir / f"locust_log_u{users}_in{input_tokens}_out{output_tokens}.log"
            
            total_requests = users * self.max_requests
            with tqdm(total=total_requests, desc=f"Requests (u={users}, in={input_tokens}, out={output_tokens})", leave=True) as pbar, \
                open(log_file_path, 'w') as log_file:
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
                
                generated_text_count = 0
                for line in iter(process.stdout.readline, ''):
                    log_file.write(line)
                    log_file.flush()
                    
                    if "Generated Text:" in line:
                        generated_text_count += 1
                        if generated_text_count % users == 0:
                            update_amount = users
                            pbar.update(update_amount)

                    if pbar.n >= total_requests:
                        process.terminate()
                        break

                remaining = generated_text_count - pbar.n
                if remaining > 0:
                    pbar.update(remaining)

                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logging.warning("Locust didn't terminate gracefully. Forcing termination.")
                    process.kill()

            if process.returncode != 0 and process.returncode != -signal.SIGTERM.value:
                logging.error(f"Locust command failed with return code {process.returncode}. Check the log file: {log_file_path}")

    def _calculate_average(self, user_dir: Path, input_token: int = None, random_prompt: bool = False):

        if self.random_prompt:
            input_file=user_dir / "Response.csv"
            output_file = user_dir / f"avg_Response.csv"

            avg_script = pkg_resources.resource_filename('scalebench', 'utils/avg_locust_results.py')
            command = [
                "python3",
                avg_script,
                "--input_csv_filename", str(input_file),
                "--output_csv_filename", str(output_file),
                "--random_prompt"
                ]
            
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error calculating average: {e}")
                raise

        else:
            input_file = user_dir / f"{input_token}_input_tokens.csv"
            output_file = user_dir / f"avg_{input_token}_input_tokens.csv"
            
            avg_script = pkg_resources.resource_filename('scalebench', 'utils/avg_locust_results.py')
            command = [
                "python3",
                avg_script,
                "--input_csv_filename", str(input_file),
                "--output_csv_filename", str(output_file),
                "--tokens"
            ] + [str(t) for t in self.output_tokens]

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error calculating average: {e}")
                raise

def run_scalebench(
    output_dir: str,
    api_url: str,
    inference_server: str,
    model_name: str = None,
    max_requests: int = 5,
    user_counts: List[int] = [1],
    input_tokens: List[int] = None,
    output_tokens: List[int] = None
) -> None:
    """Run a ScaleBench benchmark with the given parameters.
    
    Args:
        output_dir: Directory to store benchmark results
        api_url: URL of the inference API
        inference_server: Type of inference server
        model_name: Name of the model to benchmark
        max_requests: Maximum number of requests per user
        user_counts: List of concurrent user counts
        input_tokens: List of input token counts
        output_tokens: List of output token counts
        
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If benchmark execution fails
        OSError: If directory operations fail
    """
    try:
        benchmark = ScaleBench(
            output_dir=output_dir,
            api_url=api_url,
            inference_server=inference_server,
            model_name=model_name,
            max_requests=max_requests,
            user_counts=user_counts,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        benchmark.run_benchmark()
    except Exception as e:
        logging.error(f"Benchmark failed: {str(e)}")
        raise
