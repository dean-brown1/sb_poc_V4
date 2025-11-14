"""
Utility Functions

Includes:
- Config loading (YAML)
- Telemetry logging (JSONL)
- Random seed setting
- Result saving
"""

import os
import json
import yaml
import random
import torch
import numpy as np
from datetime import datetime
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dict with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_info():
    """
    Get current git commit hash for reproducibility
    
    Returns:
        Dict with git info, or None if not in git repo
    """
    try:
        import subprocess
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        
        # Check if there are uncommitted changes
        try:
            subprocess.check_output(
                ['git', 'diff-index', '--quiet', 'HEAD', '--'],
                stderr=subprocess.DEVNULL
            )
            dirty = False
        except subprocess.CalledProcessError:
            dirty = True
        
        return {
            'commit': commit,
            'branch': branch,
            'dirty': dirty
        }
    except:
        return None


class TelemetryLogger:
    """
    Logger for training telemetry
    
    Writes per-step metrics to JSONL file for downstream analysis
    """
    
    def __init__(self, log_path):
        """
        Args:
            log_path: Path to JSONL log file
        """
        self.log_path = log_path
        self.log_file = open(log_path, 'w')
        
    def log_step(self, step, metrics):
        """
        Log metrics for a single training step
        
        Args:
            step: Training step number
            metrics: Dict of metric name -> value
        """
        entry = {'step': step, **metrics}
        self.log_file.write(json.dumps(entry) + '\n')
        
        # Flush every 10 steps to ensure data is written
        if step % 10 == 0:
            self.log_file.flush()
        
    def __del__(self):
        """Ensure file is closed"""
        self.close()

    def log_routing(self, data):
        """
        Log routing alignment data to separate file
        
        Args:
            data: Dict with routing alignment info
        """
        # Create routing log file if it doesn't exist
        if not hasattr(self, 'routing_log_file'):
            routing_path = str(self.log_path).replace('training_log.jsonl', 'routing_alignment.jsonl')
            self.routing_log_file = open(routing_path, 'w')
        
        self.routing_log_file.write(json.dumps(data) + '\n')
        self.routing_log_file.flush()
    
    def close(self):
        """Close log files"""
        if self.log_file:
            self.log_file.close()
        if hasattr(self, 'routing_log_file') and self.routing_log_file:
            self.routing_log_file.close()


def save_experiment_results(output_dir, config, training_summary, evaluation_results):
    """
    Save complete experiment results
    
    Creates:
    - config.yaml: Configuration used
    - results.json: Comprehensive results
    - training_log.jsonl: Already created by TelemetryLogger
    
    Args:
        output_dir: Directory to save results
        config: Configuration dict
        training_summary: Training metadata
        evaluation_results: Evaluation metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Compile full results
    results = {
        'experiment': config['experiment'],
        'timestamp': datetime.now().isoformat(),
        'git_info': get_git_info(),
        'config': config,
        'training': training_summary,
        'evaluation': evaluation_results
    }
    
    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}")
    print(f"   - config.yaml")
    print(f"   - results.json")
    print(f"   - training_log.jsonl")


def load_training_log(log_path):
    """
    Load training log from JSONL file
    
    Args:
        log_path: Path to training_log.jsonl
        
    Returns:
        List of step metrics
    """
    logs = []
    with open(log_path, 'r') as f:
        for line in f:
            logs.append(json.loads(line))
    return logs


def print_config_summary(config):
    """
    Print readable summary of configuration
    
    Args:
        config: Configuration dict
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT: {config['experiment']['name']}")
    print("="*70)
    print(f"Mode: {config['experiment']['mode']}")
    print(f"Seed: {config['experiment']['seed']}")
    print(f"Model: {config['model']['base_model']}")
    
    if config['experiment']['mode'] == 'schemabank':
        sb = config['schemabank']
        print(f"\nSchemaBank Configuration:")
        print(f"  Schemas: {sb['num_schemas']}")
        print(f"  Rank: {sb['rank']}")
        print(f"  Top-k: {sb['topk']}")
        print(f"  Layers: {sb['layers']}")
        
        print(f"\nThree-Stage Training:")
        for stage_name, stage_config in config['training']['stages'].items():
            print(f"  {stage_name}: {stage_config['steps']} steps")
    else:
        print(f"\nBaseline: LoRA adapters only")
        print(f"  Training steps: {config['training']['total_steps']}")
    
    print("="*70 + "\n")


def compute_training_summary(telemetry_log):
    """
    Compute summary statistics from training log
    
    Args:
        telemetry_log: List of step metrics from training_log.jsonl
        
    Returns:
        Dict with training summary
    """
    if not telemetry_log:
        return {}
    
    # Extract losses
    losses = [entry.get('loss', 0) for entry in telemetry_log]
    
    # Find stage boundaries (if stages exist)
    stages = {}
    current_stage = None
    stage_start = 0
    
    for entry in telemetry_log:
        stage = entry.get('stage')
        if stage and stage != current_stage:
            if current_stage:
                # Save previous stage
                stage_losses = losses[stage_start:entry['step']]
                stages[current_stage] = {
                    'steps': len(stage_losses),
                    'initial_loss': stage_losses[0] if stage_losses else 0,
                    'final_loss': stage_losses[-1] if stage_losses else 0,
                    'mean_loss': np.mean(stage_losses) if stage_losses else 0
                }
            current_stage = stage
            stage_start = entry['step']
    
    # Handle last stage
    if current_stage:
        stage_losses = losses[stage_start:]
        stages[current_stage] = {
            'steps': len(stage_losses),
            'initial_loss': stage_losses[0] if stage_losses else 0,
            'final_loss': stage_losses[-1] if stage_losses else 0,
            'mean_loss': np.mean(stage_losses) if stage_losses else 0
        }
    
    return {
        'total_steps': len(telemetry_log),
        'initial_loss': losses[0] if losses else 0,
        'final_loss': losses[-1] if losses else 0,
        'mean_loss': np.mean(losses) if losses else 0,
        'stages': stages
    }
