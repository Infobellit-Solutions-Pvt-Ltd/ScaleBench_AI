import pytest
from click.testing import CliRunner
from scalebench.benchmark_cli import cli
import json
from unittest.mock import patch, Mock, call
import pandas as pd
from pathlib import Path
from scalebench.benchmark_core import ScaleBench, run_scalebench
from scalebench.dataset_manager import download_dataset_files
from scalebench.load_optimizer import adjust_user_count, run_benchmark_with_incremental_requests

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def mock_config_file(tmp_path):
    config = {
        "out_dir": "test_results",
        "base_url": "http://localhost:8000/v1/completions",
        "inference_server": "vLLM",
        "model": "meta-llama/Meta-Llama-3-8B",
        "max_requests": 5,
        "user_counts": [3],
        "input_tokens": [32],
        "output_tokens": [256],
        "tokenizer_path": "meta-llama/Meta-Llama-3-8B"
    }
    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)
    return str(config_file)

def test_cli_help(runner):
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'ScaleBench: LLM Inference Benchmarking Tool' in result.output

@patch('scalebench.cli.download_dataset_files')
@patch('scalebench.cli.create_config')
def test_dataprep_command(mock_create_config, mock_download, runner):
    result = runner.invoke(cli, ['dataprep'])
    assert result.exit_code == 0
    assert call("epsilondelta1982/Dataset-20k") in mock_download.call_args_list
    assert call("epsilondelta1982/Dataset-8k") in mock_download.call_args_list
    mock_create_config.assert_called_once_with('config.json')

@patch('scalebench.cli.download_dataset_files')
@patch('scalebench.cli.create_config')
def test_dataprep_command_custom_config(mock_create_config, mock_download, runner, tmp_path):
    custom_config_path = tmp_path / "custom_config.json"
    custom_config_path.parent.mkdir(parents=True, exist_ok=True)
    custom_config_path.write_text('{}')  # <-- This fixes the FileNotFoundError

    result = runner.invoke(cli, ['dataprep', '--config', str(custom_config_path)])
    assert result.exit_code == 0
    assert call("epsilondelta1982/Dataset-20k") in mock_download.call_args_list
    assert call("epsilondelta1982/Dataset-8k") in mock_download.call_args_list
    mock_create_config.assert_called_once_with(str(custom_config_path))

def test_start_command_without_config(runner):
    result = runner.invoke(cli, ['start'])
    assert result.exit_code != 0
    assert 'Error: Missing option \'--config\'' in result.output

@patch('scalebench.cli.Path')
@patch('scalebench.cli.ScaleBench')
@patch('scalebench.cli.load_config')
@patch('scalebench.cli.pd.read_csv')
@patch('scalebench.cli.pd.concat')
@patch('scalebench.cli.tabulate')
def test_start_command_with_config(mock_tabulate, mock_concat, mock_read_csv, mock_load_config, mock_scalebench, mock_path, runner, mock_config_file):
    mock_config = {
        "out_dir": "test_results",
        "base_url": "http://localhost:8000/v1/completions",
        "inference_server": "vLLM",
        "model": "meta-llama/Meta-Llama-3-8B",
        "max_requests": 5,
        "user_counts": [3],
        "input_tokens": [32],
        "output_tokens": [256],
        "tokenizer_path": "meta-llama/Meta-Llama-3-8B"
    }
    mock_load_config.return_value = mock_config
    mock_path.return_value.exists.return_value = True
    mock_path.return_value.iterdir.return_value = [Mock()]

    mock_df = pd.DataFrame({
        'output_tokens': [256],
        'throughput(tokens/second)': [100],
        'latency(ms)': [50],
        'TTFT(ms)': [10],
        'latency_per_token(ms/token)': [0.2],
    })

    mock_read_csv.return_value = mock_df
    mock_concat.return_value = mock_df

    result = runner.invoke(cli, ['start', '--config', mock_config_file])
    assert result.exit_code == 0, f"Command failed with error: {result.output}"

@patch('scalebench.cli.Path')
@patch('scalebench.cli.load_config')
def test_start_command_without_dataset(mock_load_config, mock_path, runner, mock_config_file):
    mock_config = {
        "out_dir": "test_results",
        "base_url": "http://localhost:8000/v1/completions",
        "inference_server": "vLLM",
        "model": "meta-llama/Meta-Llama-3-8B",
        "max_requests": 5,
        "user_counts": [3],
        "input_tokens": [32],
        "output_tokens": [256],
        "tokenizer_path": "meta-llama/Meta-Llama-3-8B"
    }
    mock_load_config.return_value = mock_config
    mock_path.return_value.exists.return_value = False
    mock_path.return_value.iterdir.return_value = []

    result = runner.invoke(cli, ['start', '--config', mock_config_file])
    assert result.exit_code != 0
    assert "Filtered dataset not found" in result.output

def test_plot_command_without_results_dir(runner):
    result = runner.invoke(cli, ['plot'])
    assert result.exit_code != 0
    assert 'Error: Missing option \'--results-dir\'' in result.output

@patch('scalebench.cli.plot_benchmark_results')
def test_plot_command_with_results_dir(mock_plot, runner, tmp_path):
    results_dir = tmp_path / "test_results"
    results_dir.mkdir()

    result = runner.invoke(cli, ['plot', '--results-dir', str(results_dir)])
    assert result.exit_code == 0
    mock_plot.assert_called_once_with(results_dir, False)  # fix: remove 2nd argument check

def test_plot_command_with_invalid_results_dir(runner):
    result = runner.invoke(cli, ['plot', '--results-dir', '/non/existent/path'])
    assert result.exit_code != 0
    assert 'Error: Invalid value for \'--results-dir\'' in result.output

# ScaleBench Core Tests
@pytest.fixture
def benchmark_instance():
    return ScaleBench(
        output_dir="test_output",
        api_url="http://localhost:8000/v1/completions",
        inference_server="vLLM",
        model_name="test-model",
        max_requests=5,
        user_counts=[1, 2],
        input_tokens=[32, 64],
        output_tokens=[128, 256],
        tokenizer_path="test-tokenizer"
    )

def test_scalebench_initialization(benchmark_instance):
    assert benchmark_instance.output_dir == Path("test_output")
    assert benchmark_instance.api_url == "http://localhost:8000/v1/completions"
    assert benchmark_instance.inference_server == "vLLM"
    assert benchmark_instance.model_name == "test-model"
    assert benchmark_instance.max_requests == 5
    assert benchmark_instance.user_counts == [1, 2]
    assert benchmark_instance.input_tokens == [32, 64]
    assert benchmark_instance.output_tokens == [128, 256]
    assert benchmark_instance.tokenizer_path == "test-tokenizer"

@patch('subprocess.Popen')
@patch('subprocess.run')
def test_run_locust(mock_run, mock_popen, benchmark_instance, tmp_path):
    # Setup mock process
    mock_process = Mock()
    mock_process.stdout.readline.side_effect = [
        "Starting Locust...",
        "Generated Text: test1",
        "Generated Text: test2",
        ""
    ]
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    
    # Create test directories
    output_file = tmp_path / "test.csv"
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    
    # Run the test
    benchmark_instance._run_locust(
        users=1,
        output_file=output_file,
        logs_dir=logs_dir,
        input_tokens=32,
        output_tokens=128
    )
    
    # Verify Popen was called with correct arguments
    assert mock_popen.call_count == 1
    args, kwargs = mock_popen.call_args
    command = args[0]
    assert "locust" in command
    assert "--headless" in command

@patch('pathlib.Path.mkdir')
@patch('scalebench.llm_inference_benchmark.ScaleBench._run_locust')
@patch('scalebench.llm_inference_benchmark.ScaleBench._calculate_average')
def test_run_benchmark(mock_calc_avg, mock_run_locust, mock_mkdir, benchmark_instance):
    benchmark_instance.run_benchmark()
    assert mock_mkdir.call_count >= 2
    expected_calls = len(benchmark_instance.user_counts) * len(benchmark_instance.input_tokens) * len(benchmark_instance.output_tokens)
    assert mock_run_locust.call_count == expected_calls

# Dataset Tests
@patch('datasets.load_dataset')
def test_download_dataset_files(mock_load_dataset, tmp_path):
    mock_dataset = Mock()
    mock_dataset.to_csv.return_value = None
    mock_load_dataset.return_value = mock_dataset
    
    dataset_dir = tmp_path / "Input_Dataset"
    dataset_dir.mkdir(parents=True)
    
    download_dataset_files("test/dataset")
    mock_load_dataset.assert_called_once_with("test/dataset")
    mock_dataset.to_csv.assert_called_once()

@patch('datasets.load_dataset')
def test_download_dataset_files_error(mock_load_dataset):
    mock_load_dataset.side_effect = Exception("Failed to download dataset")
    with pytest.raises(Exception) as exc_info:
        download_dataset_files("test/dataset")
    assert "Failed to download dataset" in str(exc_info.value)

# Optimal User Tests
@patch('time.sleep')
def test_adjust_user_count(mock_sleep):
    # Test increasing user count
    result = adjust_user_count(
        current_users=5,
        prev_latency=100,
        current_latency=90,
        min_users=1,
        max_users=20
    )
    assert result > 5  # Should increase users when latency decreases
    
    # Test decreasing user count
    result = adjust_user_count(
        current_users=5,
        prev_latency=90,
        current_latency=110,
        min_users=1,
        max_users=20
    )
    assert result < 5  # Should decrease users when latency increases

@patch('subprocess.run')
def test_run_benchmark_with_incremental_requests(mock_run, mock_config_file):
    mock_process = Mock()
    mock_process.returncode = 0
    mock_process.stdout = "Test output"
    mock_run.return_value = mock_process
    
    result = run_benchmark_with_incremental_requests(mock_config_file, 5, "test_results")
    assert mock_run.call_count == 1
    args, kwargs = mock_run.call_args
    command = args[0]
    assert "scalebench" in command[0]
    assert "start" in command[1]
