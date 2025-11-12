"""
Training Module

Includes:
- Three-stage SchemaBank training (router → schemas → joint)
- Baseline LoRA-only training
- Model loading and preparation
- Training utilities
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

from .data import (
    create_dataloader, 
    get_tag_dropout_rate,
    pack_gsm8k_with_tags
)
from .model import attach_schemabank_last2
from datasets import Dataset
from torch.utils.data import DataLoader


def load_base_model_and_tokenizer(model_name, torch_dtype=torch.bfloat16):
    """
    Load base model and tokenizer
    
    Args:
        model_name: Hugging Face model identifier
        torch_dtype: Data type for model
        
    Returns:
        (model, tokenizer, hidden_size)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    
    # Get hidden size
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not determine hidden size from model config")
    
    # Disable caching for training
    model.config.use_cache = False
    
    return model, tokenizer, hidden_size


def add_lora_adapters(model, lora_config):
    """
    Add LoRA adapters to model
    
    Args:
        model: Base model
        lora_config: Dict with LoRA configuration
        
    Returns:
        Model with LoRA adapters
    """
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias=lora_config['bias'],
        task_type=lora_config['task_type']
    )
    
    model = get_peft_model(model, peft_config)
    
    return model


def prepare_model(config, mode='baseline'):
    """
    Prepare model for training
    
    Args:
        config: Full configuration dict
        mode: 'baseline' or 'schemabank'
        
    Returns:
        (model, tokenizer, hidden_size)
    """
    # Load base model
    model, tokenizer, hidden_size = load_base_model_and_tokenizer(
        config['model']['base_model'],
        torch_dtype=getattr(torch, config['model']['torch_dtype'])
    )
    
    # Add LoRA adapters
    model = add_lora_adapters(model, config['lora'])
    
    # Add SchemaBank if in schemabank mode
    if mode == 'schemabank':
        sb_config = config['schemabank']
        model = attach_schemabank_last2(
            model,
            H=hidden_size,
            S=sb_config['num_schemas'],
            r=sb_config['rank'],
            topk=sb_config['topk'],
            ad=sb_config['attr_dim']
        )
        # adapters are now stored as model.schemabank_adapters by attach_schemabank_last2
        print(f"✓ SchemaBank attached: {sb_config['num_schemas']} schemas, rank {sb_config['rank']}")
    return model, tokenizer, hidden_size

# ========== Baseline Training ==========

def train_baseline(model, tagged_data, tokenizer, config, telemetry_logger):
    """
    Train baseline model (LoRA adapters only, no SchemaBank)
    
    Simple single-stage training for comparison.
    
    Args:
        model: Model with LoRA adapters
        tagged_data: Prepared dataset
        tokenizer: Tokenizer
        config: Configuration dict
        telemetry_logger: TelemetryLogger instance
        
    Returns:
        Trained model
    """
    device = next(model.parameters()).device
    training_config = config['training']
    
    print("\n" + "="*70)
    print("BASELINE TRAINING: LoRA Adapters Only")
    print("="*70)
    print(f"Steps: {training_config['total_steps']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print("="*70 + "\n")
    
    # Get trainable parameters (LoRA only)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(5, int(training_config['warmup_ratio'] * training_config['total_steps'])),
        num_training_steps=training_config['total_steps']
    )
    
    # Create dataloader (no tags for baseline)
    dataloader = create_dataloader(
        tagged_data,
        tokenizer,
        max_len=training_config['seq_len'],
        batch_size=training_config['batch_size'],
        tag_dropout_rate=1.0,  # Drop all tags
        shuffle=True
    )
    
    # Training loop
    model.train()
    step = 0
    pbar = tqdm(total=training_config['total_steps'], desc="Training Baseline")
    
    while step < training_config['total_steps']:
        for batch in dataloader:
            if step >= training_config['total_steps']:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, 
                training_config['max_grad_norm']
            ).item()
            optimizer.step()
            scheduler.step()
            
            # Log metrics
            telemetry_logger.log_step(step + 1, {
                'stage': 'baseline_training',
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'grad_norm': grad_norm
            })
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    print("\n✓ Baseline training complete\n")
    
    return model


# ========== SchemaBank Three-Stage Training ==========

def train_stage1_router(model, tagged_data, tokenizer, config, telemetry_logger):
    """
    Stage 1: Router Pre-training with Tag Curriculum
    
    Progressive tag dropout:
    - 0-25% steps:   100% tags (learn schema mapping)
    - 25-50% steps:   75% tags (start generalizing)
    - 50-75% steps:   50% tags (more independent)
    - 75-100% steps:  25% tags (mostly autonomous)
    
    Args:
        model: Model with SchemaBank
        tagged_data: Prepared dataset
        tokenizer: Tokenizer
        config: Configuration dict
        telemetry_logger: TelemetryLogger instance
        
    Returns:
        Model with trained router
    """
    device = next(model.parameters()).device
    stage_config = config['training']['stages']['stage1_router_pretrain']
    steps = stage_config['steps']
    
    print("\n" + "="*70)
    print("STAGE 1: Router Pre-training with Tag Curriculum")
    print("="*70)
    print("Training: Router weights only")
    print("Frozen: Base model + LoRA + Schema transforms")
    print(f"Steps: {steps}")
    print("\nTag Schedule:")
    print("  Quarter 1 (0-25%):   100% tags → Learn mapping")
    print("  Quarter 2 (25-50%):   75% tags → Start generalizing")
    print("  Quarter 3 (50-75%):   50% tags → More independent")
    print("  Quarter 4 (75-100%):  25% tags → Mostly autonomous")
    print("="*70 + "\n")
    
    # Freeze everything except router
    for param in model.parameters():
        param.requires_grad = False
    
    adapters = getattr(model, "schemabank_adapters", None)
    if not adapters:
        raise ValueError("No SchemaBank adapters found on model")
    
    # Unfreeze only router weights
    for adapter in adapters:
        adapter.router.weight.requires_grad = True
    
    # Higher learning rate for router-only training
    optimizer = torch.optim.AdamW(
        [adapter.router.weight for adapter in adapters],
        lr=1e-3,  # 10x higher than normal
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(5, int(0.05 * steps)),
        num_training_steps=steps
    )
    
    # Training loop with tag curriculum
    model.train()
    step = 0
    pbar = tqdm(total=steps, desc="Stage 1: Router + Tags")
    last_quarter = -1
    dataloader = None
    
    while step < steps:
        # Calculate current quarter and dropout rate
        progress = step / max(steps, 1)
        current_quarter = int(progress * 4)
        dropout_rate = get_tag_dropout_rate(step, steps)
        
        # Re-create dataloader when entering new quarter
        if current_quarter != last_quarter:
            print(f"\n→ Quarter {current_quarter+1}/4: dropout={dropout_rate:.2f} (keeping {(1-dropout_rate)*100:.0f}% of tags)")
            dataloader = create_dataloader(
                tagged_data,
                tokenizer,
                max_len=config['training']['seq_len'],
                batch_size=config['training']['batch_size'],
                tag_dropout_rate=dropout_rate,
                shuffle=True
            )
            dataloader_iter = iter(dataloader)
            last_quarter = current_quarter
        
        # Get next batch
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Restart dataloader
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [adapter.router.weight for adapter in adapters],
            config['training']['max_grad_norm']
        ).item()
        optimizer.step()
        scheduler.step()
        
        # Log metrics
        telemetry_logger.log_step(step + 1, {
            'stage': 'stage1_router_pretrain',
            'loss': loss.item(),
            'tag_dropout': dropout_rate,
            'lr': optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm
        })
        
        step += 1
        pbar.update(1)
    
    pbar.close()
    print("\n✓ Stage 1 complete: Router learned from outcomes + tags\n")
    
    return model


def train_stage2_schemas(model, tagged_data, tokenizer, config, telemetry_logger):
    """
    Stage 2: Schema Training (NO TAGS)
    
    Router is frozen, schemas learn useful transformations.
    No tags - pure outcome-based learning for schemas.
    
    Args:
        model: Model with trained router
        tagged_data: Prepared dataset
        tokenizer: Tokenizer
        config: Configuration dict
        telemetry_logger: TelemetryLogger instance
        
    Returns:
        Model with trained schemas
    """
    device = next(model.parameters()).device
    stage_config = config['training']['stages']['stage2_schema_train']
    steps = stage_config['steps']
    ortho_weight = config['schemabank']['regularization']['ortho_weight']
    
    print("\n" + "="*70)
    print("STAGE 2: Schema Training")
    print("="*70)
    print("Training: Schema U/V matrices + LoRA adapters")
    print("Frozen: Base model + Router")
    print(f"Steps: {steps}")
    print("No tags: Router frozen, schemas learn from outcomes")
    print("="*70 + "\n")
    
    # Freeze base model, unfreeze LoRA and schemas
    trainable_params = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    
    adapters = getattr(model, "schemabank_adapters", None)
    if not adapters:
        raise ValueError("No SchemaBank adapters found on model")
    
    # Unfreeze schema matrices (router stays frozen)
    for adapter in adapters:
        adapter.router.weight.requires_grad = False
        adapter.U.requires_grad = True
        adapter.V.requires_grad = True
        trainable_params.extend([adapter.U, adapter.V])
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(5, int(0.05 * steps)),
        num_training_steps=steps
    )
    
    # Create dataloader (no tags)
    dataloader = create_dataloader(
        tagged_data,
        tokenizer,
        max_len=config['training']['seq_len'],
        batch_size=config['training']['batch_size'],
        tag_dropout_rate=1.0,  # Drop all tags
        shuffle=True
    )
    
    # Training loop
    model.train()
    step = 0
    pbar = tqdm(total=steps, desc="Stage 2: Schemas")
    
    while step < steps:
        for batch in dataloader:
            if step >= steps:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            # Task loss + orthonormality regularization
            ce_loss = outputs.loss
            
            ortho_loss = sum(adapter.regs() for adapter in adapters)
            loss = ce_loss + ortho_weight * ortho_loss
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                config['training']['max_grad_norm']
            ).item()
            optimizer.step()
            scheduler.step()
            
            # Log metrics
            telemetry_logger.log_step(step + 1, {
                'stage': 'stage2_schema_train',
                'loss': loss.item(),
                'ce_loss': ce_loss.item(),
                'ortho_loss': ortho_loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'grad_norm': grad_norm
            })
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    print("\n✓ Stage 2 complete: Schemas learned transformations\n")
    
    return model


def train_stage3_joint(model, tagged_data, tokenizer, config, telemetry_logger):
    """
    Stage 3: Joint Fine-tuning (NO TAGS)
    
    Router and schemas train together to optimize task performance.
    
    Args:
        model: Model with trained router and schemas
        tagged_data: Prepared dataset
        tokenizer: Tokenizer
        config: Configuration dict
        telemetry_logger: TelemetryLogger instance
        
    Returns:
        Final trained model
    """
    device = next(model.parameters()).device
    stage_config = config['training']['stages']['stage3_joint_finetune']
    steps = stage_config['steps']
    ortho_weight = config['schemabank']['regularization']['ortho_weight']
    
    print("\n" + "="*70)
    print("STAGE 3: Joint Fine-tuning")
    print("="*70)
    print("Training: Router + Schema matrices + LoRA adapters")
    print("Frozen: Base model only")
    print(f"Steps: {steps}")
    print("No tags: Pure outcome-based optimization")
    print("="*70 + "\n")
    
    # Freeze base model only, unfreeze LoRA + SchemaBank
    trainable_params = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    
    adapters = getattr(model, "schemabank_adapters", None)
    if not adapters:
        raise ValueError("No SchemaBank adapters found on model")
    
    # Unfreeze router and schemas
    for adapter in adapters:
        adapter.router.weight.requires_grad = True
        adapter.U.requires_grad = True
        adapter.V.requires_grad = True
        trainable_params.extend([adapter.router.weight, adapter.U, adapter.V])
    
    # Lower learning rate for stability
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=5e-5,  # Half of normal
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(5, int(0.05 * steps)),
        num_training_steps=steps
    )
    
    # Create dataloader (no tags)
    dataloader = create_dataloader(
        tagged_data,
        tokenizer,
        max_len=config['training']['seq_len'],
        batch_size=config['training']['batch_size'],
        tag_dropout_rate=1.0,  # Drop all tags
        shuffle=True
    )
    
    # Training loop
    model.train()
    step = 0
    pbar = tqdm(total=steps, desc="Stage 3: Joint")
    
    while step < steps:
        for batch in dataloader:
            if step >= steps:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            # Task loss + orthonormality regularization
            ce_loss = outputs.loss
            
            ortho_loss = sum(adapter.regs() for adapter in adapters)
            loss = ce_loss + ortho_weight * ortho_loss
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                config['training']['max_grad_norm']
            ).item()
            optimizer.step()
            scheduler.step()
            
            # Log metrics
            telemetry_logger.log_step(step + 1, {
                'stage': 'stage3_joint_finetune',
                'loss': loss.item(),
                'ce_loss': ce_loss.item(),
                'ortho_loss': ortho_loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'grad_norm': grad_norm
            })
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    print("\n✓ Stage 3 complete: System optimized\n")
    
    return model


def train_schemabank(model, tagged_data, tokenizer, config, telemetry_logger):
    """
    Complete three-stage SchemaBank training
    
    Args:
        model: Model with SchemaBank adapters
        tagged_data: Prepared dataset
        tokenizer: Tokenizer
        config: Configuration dict
        telemetry_logger: TelemetryLogger instance
        
    Returns:
        Fully trained model
    """
    # Stage 1: Router pre-training
    model = train_stage1_router(model, tagged_data, tokenizer, config, telemetry_logger)
    
    # Stage 2: Schema training
    model = train_stage2_schemas(model, tagged_data, tokenizer, config, telemetry_logger)
    
    # Stage 3: Joint fine-tuning
    model = train_stage3_joint(model, tagged_data, tokenizer, config, telemetry_logger)
    
    return model
