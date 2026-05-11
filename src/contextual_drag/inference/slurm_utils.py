"""
SLURM and job partitioning utilities for vLLM serving script.

This module handles SLURM environment detection, partition management,
and job information display for distributed batch processing.
"""
import os
import socket
import subprocess
from typing import Dict, Any


def get_partition_id(args) -> int:
    """Get the current partition ID from args or SLURM environment"""
    if args.partition_id is not None:
        return args.partition_id

    # Try to get from SLURM environment
    slurm_array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if slurm_array_task_id is not None:
        return int(slurm_array_task_id)

    # Default to 0 if no SLURM environment and no explicit partition_id
    return 0


def get_slurm_info() -> Dict[str, str]:
    """Get comprehensive SLURM job information from environment variables"""
    slurm_vars = [
        'SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID',
        'SLURM_CLUSTER_NAME', 'SLURM_PARTITION', 'SLURM_QOS', 'SLURM_ACCOUNT',
        'SLURM_JOB_USER', 'SLURM_SUBMIT_DIR', 'SLURM_SUBMIT_HOST',
        'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS',
        'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE', 'SLURM_GRES',
        'SLURM_JOB_START_TIME', 'SLURM_JOB_END_TIME', 'SLURM_TIMELIMIT'
    ]

    slurm_info = {}
    for var in slurm_vars:
        value = os.environ.get(var)
        if value is not None:
            slurm_info[var] = value

    return slurm_info


def get_system_info() -> Dict[str, str]:
    """Get system information including hostname, node details, and GPU info"""
    system_info = {}

    # Basic system info
    system_info['hostname'] = socket.gethostname()
    system_info['fqdn'] = socket.getfqdn()

    # Try to get additional system details
    try:
        # Get username
        system_info['user'] = os.environ.get('USER', 'unknown')

        # Get current working directory
        system_info['cwd'] = os.getcwd()

        # Get load average
        if os.path.exists('/proc/loadavg'):
            with open('/proc/loadavg', 'r') as f:
                system_info['load_avg'] = f.read().strip()

        # Get memory info
        if os.path.exists('/proc/meminfo'):
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if line.startswith('MemTotal:'):
                        system_info['total_memory'] = line.split()[1] + ' kB'
                        break

        # Try to get GPU info using nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_lines = result.stdout.strip().split('\n')
                system_info['gpus'] = [line.strip() for line in gpu_lines if line.strip()]
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            system_info['gpus'] = ['GPU info unavailable']

    except Exception as e:
        system_info['error'] = f"Error gathering system info: {e}"

    return system_info


def print_job_info(args, partition_id: int, config: Dict[str, Any]) -> None:
    """Print comprehensive job information at startup"""
    print("=" * 80)
    print("VLLM BATCH INFERENCE JOB STARTING")
    print("=" * 80)
    print(f"Timestamp: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script: {__file__}")
    print()

    # System information
    system_info = get_system_info()
    print("=== SYSTEM INFO ===")
    print(f"Hostname: {system_info.get('hostname', 'unknown')}")
    print(f"FQDN: {system_info.get('fqdn', 'unknown')}")
    print(f"User: {system_info.get('user', 'unknown')}")
    print(f"Working directory: {system_info.get('cwd', 'unknown')}")
    if 'load_avg' in system_info:
        print(f"Load average: {system_info['load_avg']}")
    if 'total_memory' in system_info:
        print(f"Total memory: {system_info['total_memory']}")

    # GPU information
    if 'gpus' in system_info:
        print(f"GPU(s):")
        for i, gpu in enumerate(system_info['gpus']):
            print(f"  GPU {i}: {gpu}")

    if 'error' in system_info:
        print(f"System info error: {system_info['error']}")
    print()

    # SLURM information
    slurm_info = get_slurm_info()
    print("=== SLURM JOB INFO ===")
    if slurm_info:
        # Core job info
        for key in ['SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID']:
            if key in slurm_info:
                print(f"{key}: {slurm_info[key]}")

        # Cluster and partition info
        print()
        for key in ['SLURM_CLUSTER_NAME', 'SLURM_PARTITION', 'SLURM_QOS', 'SLURM_ACCOUNT']:
            if key in slurm_info:
                print(f"{key}: {slurm_info[key]}")

        # Node and resource info
        print()
        for key in ['SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_CPUS_PER_TASK']:
            if key in slurm_info:
                print(f"{key}: {slurm_info[key]}")

        # Memory and GPU resources
        for key in ['SLURM_MEM_PER_NODE', 'SLURM_GRES']:
            if key in slurm_info:
                print(f"{key}: {slurm_info[key]}")

        # Timing info
        print()
        for key in ['SLURM_JOB_START_TIME', 'SLURM_JOB_END_TIME', 'SLURM_TIMELIMIT']:
            if key in slurm_info:
                print(f"{key}: {slurm_info[key]}")

        # Submit info
        print()
        for key in ['SLURM_JOB_USER', 'SLURM_SUBMIT_DIR', 'SLURM_SUBMIT_HOST']:
            if key in slurm_info:
                print(f"{key}: {slurm_info[key]}")

    else:
        print("No SLURM environment detected (running locally)")

    print()
    print("=== PARTITION INFO ===")
    print(f"Partition ID: {partition_id}")
    print(f"Total partitions: {args.num_partitions}")
    print()

    print("=== MODEL CONFIG ===")
    print(f"Model config alias: {config['model_config_alias']}")
    print(f"Model path: {config['model_name']}")
    print(f"Context length: {config['context_length']}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print()

    print("=== SAMPLING PARAMETERS ===")
    sampling_config = config['sampling_params']
    print(f"Temperature: {sampling_config.get('temperature', 'Not set')}")
    print(f"Top-p: {sampling_config.get('top_p', 'Not set')}")
    print(f"Top-k: {sampling_config.get('top_k', 'Not set')}")
    print(f"Max tokens: {sampling_config.get('max_tokens', 'Not set')}")
    print(f"Responses per question (n): {sampling_config.get('n', 'Not set')}")
    print(f"Seed: {sampling_config.get('seed', 'Not set')}")
    if 'repetition_penalty' in sampling_config:
        print(f"Repetition penalty: {sampling_config['repetition_penalty']}")
    if 'presence_penalty' in sampling_config:
        print(f"Presence penalty: {sampling_config['presence_penalty']}")
    if 'frequency_penalty' in sampling_config:
        print(f"Frequency penalty: {sampling_config['frequency_penalty']}")

    # Show overrides if any
    overrides = []
    if args.temperature is not None:
        overrides.append(f"temperature={args.temperature}")
    if args.top_p is not None:
        overrides.append(f"top_p={args.top_p}")
    if args.top_k is not None:
        overrides.append(f"top_k={args.top_k}")
    if args.max_tokens is not None:
        overrides.append(f"max_tokens={args.max_tokens}")

    if overrides:
        print(f"Parameter overrides: {', '.join(overrides)}")
    print()

    print("=== DATA CONFIG ===")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max questions: {args.max_questions if args.max_questions else 'No limit'}")
    print(f"Batch size: {args.batch_size}")
    print()

    print("=== TEMPLATE CONFIG ===")
    print(f"Template path: {getattr(args, 'prompt_template_path', 'Not specified')}")
    print(f"Template key: {getattr(args, 'prompt_template_key', 'Not specified')}")
    print(f"Task name: {getattr(args, 'task_name', 'inference')}")
    print()

    print("=== GENERATION CONFIG ===")
    print(f"Thinking mode: {'enabled' if config['enable_thinking'] else 'disabled'}")
    print(f"Resume mode: {'enabled' if args.resume else 'disabled'}")
    print()
    print("=" * 80)
