# src/telemetry.py

import json
from pathlib import Path


class TelemetryLogger:
    """
    Logger for training telemetry and routing alignment
    
    Tracks:
    - Training step metrics (loss, lr, grad_norm)
    - Routing alignment (tags vs router decisions)
    - Evaluation results
    """
    
    def __init__(self, output_dir):
        """
        Initialize telemetry logger
        
        Args:
            output_dir: Directory to save logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_log = []
        self.routing_log = []
        self.evaluation_results = {}
    
    def log_step(self, step, metrics):
        """
        Log training step metrics
        
        Args:
            step: Training step number
            metrics: Dict with loss, lr, grad_norm, etc.
        """
        entry = {'step': step, **metrics}
        self.training_log.append(entry)
    
    def log_routing(self, data):
        """
        Log routing alignment data
        
        Tracks how well router decisions align with input tags.
        
        Args:
            data: Dict with step, stage, input_tags, router_decision, alignment, etc.
        """
        self.routing_log.append(data)
    
    def log_evaluation(self, eval_name, results):
        """
        Log evaluation results
        
        Args:
            eval_name: Name of evaluation (e.g., 'gsm8k', 'long_context')
            results: Dict with evaluation metrics
        """
        self.evaluation_results[eval_name] = results
    
    def save_training_log(self):
        """Save training log to JSONL file"""
        log_path = self.output_dir / 'training_log.jsonl'
        with open(log_path, 'w') as f:
            for entry in self.training_log:
                f.write(json.dumps(entry) + '\n')
        print(f"✓ Saved training log: {log_path}")
    
    def save_routing_log(self):
        """Save routing alignment log to JSONL file"""
        if not self.routing_log:
            return  # No routing data to save
        
        log_path = self.output_dir / 'routing_alignment.jsonl'
        with open(log_path, 'w') as f:
            for entry in self.routing_log:
                f.write(json.dumps(entry) + '\n')
        print(f"✓ Saved routing alignment log: {log_path}")
    
    def save_evaluation_results(self):
        """Save evaluation results to JSON file"""
        if not self.evaluation_results:
            return
        
        results_path = self.output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        print(f"✓ Saved evaluation results: {results_path}")
    
    def save_all(self):
        """Save all logs"""
        self.save_training_log()
        self.save_routing_log()
        self.save_evaluation_results()
    
    def get_training_summary(self):
        """
        Get summary statistics from training
        
        Returns:
            Dict with summary stats
        """
        if not self.training_log:
            return {}
        
        losses = [e['loss'] for e in self.training_log if 'loss' in e]
        
        summary = {
            'total_steps': len(self.training_log),
            'initial_loss': losses[0] if losses else None,
            'final_loss': losses[-1] if losses else None,
            'mean_loss': sum(losses) / len(losses) if losses else None
        }
        
        # Per-stage summaries
        stages = {}
        for entry in self.training_log:
            stage = entry.get('stage', 'unknown')
            if stage not in stages:
                stages[stage] = []
            if 'loss' in entry:
                stages[stage].append(entry['loss'])
        
        summary['stages'] = {}
        for stage, losses in stages.items():
            summary['stages'][stage] = {
                'steps': len(losses),
                'initial_loss': losses[0] if losses else 0,
                'final_loss': losses[-1] if losses else 0,
                'mean_loss': sum(losses) / len(losses) if losses else 0
            }
        
        return summary
    
    def get_routing_summary(self):
        """
        Get summary statistics from routing alignment
        
        Returns:
            Dict with routing alignment stats
        """
        if not self.routing_log:
            return {}
        
        alignments = [e['alignment'] for e in self.routing_log]
        
        # Per-stage alignment
        stage_alignments = {}
        for entry in self.routing_log:
            stage = entry.get('stage', 'unknown')
            if stage not in stage_alignments:
                stage_alignments[stage] = []
            stage_alignments[stage].append(entry['alignment'])
        
        summary = {
            'total_logged': len(self.routing_log),
            'mean_alignment': sum(alignments) / len(alignments),
            'perfect_alignment_rate': sum(1 for a in alignments if a == 1.0) / len(alignments),
            'zero_alignment_rate': sum(1 for a in alignments if a == 0.0) / len(alignments),
            'stages': {}
        }
        
        for stage, aligns in stage_alignments.items():
            summary['stages'][stage] = {
                'mean_alignment': sum(aligns) / len(aligns),
                'samples': len(aligns)
            }
        
        return summary