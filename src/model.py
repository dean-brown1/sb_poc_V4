"""
SchemaBank Model Architecture

Implements sparse mixture-of-experts style routing with low-rank schema transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SchemaBank(nn.Module):
    """
    SchemaBank: Sparse routing with low-rank schema transformations
    
    Architecture:
    - Router: Maps hidden states to schema selection logits
    - U, V: Low-rank transformation matrices per schema (rank r)
    - Attributes: Learnable schema characteristics (dim ad)
    - Gate: Per-token gating based on schema attributes
    
    Args:
        H: Hidden dimension of base model
        S: Number of schemas (default: 32)
        r: Rank of low-rank transformations (default: 16)
        topk: Number of schemas to activate per token (default: 2)
        ad: Attribute dimension for schema gating (default: 32)
    """
    
    def __init__(self, H, S=32, r=16, topk=2, ad=32):
        super().__init__()
        self.S = S
        self.r = r
        self.topk = topk
        
        # Router: H -> S schema logits
        self.router = nn.Linear(H, S, bias=False)
        
        # Low-rank transformations: U: (S, H, r), V: (S, r, H)
        self.U = nn.Parameter(torch.randn(S, H, r) * 0.02)
        self.V = nn.Parameter(torch.randn(S, r, H) * 0.02)
        
        # Schema attributes for gating
        self.attr = nn.Parameter(torch.randn(S, ad) * 0.02)
        self.aproj = nn.Linear(ad, 1, bias=False)

    def forward(self, h, return_gate=False):
        """
        Forward pass through SchemaBank
        
        Args:
            h: Hidden states (B, T, H)
            return_gate: If True, also return routing gates (B, T, S)
            
        Returns:
            out: Transformed hidden states (B, T, H)
            g (optional): Routing gate values (B, T, S)
        """
        B, T, H = h.shape
        

        # Router selection: top-k sparse softmax
        sc = self.router(h)  # (B, T, S)

            # STORE ROUTER LOGITS FOR SUPERVISION
        self.last_router_logits = sc.mean(dim=1)  # Average over sequence -> (B, S)

        idx = sc.topk(self.topk, dim=-1).indices
        m = torch.zeros_like(sc).scatter_(-1, idx, 1.0)  # Sparse mask
        g = F.softmax(sc, dim=-1) * m  # Masked softmax
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)  # Renormalize
        
        # Apply low-rank transformations per schema
        hE = h.unsqueeze(2).expand(B, T, self.S, H)  # (B, T, S, H)
        hU = torch.einsum("btsh,shr->btsr", hE, self.U)  # Project down
        hUV = torch.einsum("btsr,srh->btsh", hU, self.V)  # Project up
        
        # Mix selected schemas
        out = torch.einsum("bts,btsh->bth", g, hUV)  # (B, T, H)
        
        # Token-scalar gating from schema attributes
        aw = torch.sigmoid(self.aproj(self.attr).squeeze(-1))  # (S,)
        gate = (g @ aw).unsqueeze(-1)  # (B, T, 1)
        out = out * (0.9 + 0.2 * gate)  # Scale output
        # out = h + out  # REMOVE for accuracy increase
        
        return (out, g) if return_gate else out

    def regs(self):
        """
        Regularization losses
        
        Returns orthonormality loss for V matrices.
        Note: Entropy regularization removed - router learns from task outcomes.
        """
        ortho = 0.0
        for s in range(self.S):
            V = self.V[s]  # (r, H)
            G = V @ V.t()  # Gram matrix (r, r)
            I = torch.eye(G.size(0), device=V.device, dtype=V.dtype)
            ortho += F.mse_loss(G, I)
        
        return ortho


def unwrap_to_base(model):
    """Extract base model from wrapped PEFT model"""
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "model"):
        return base.model
    return getattr(model, "model", model)


def get_block_list(model):
    """
    Get transformer block list from various architectures
    
    Supports: LLaMA, Qwen, GPT-NeoX, OPT, and similar architectures
    """
    base = unwrap_to_base(model)
    
    candidates = [
        getattr(base, "layers", None),  # LLaMA, Qwen
        getattr(getattr(base, "model", None), "layers", None),
        getattr(getattr(base, "transformer", None), "h", None),  # GPT-2
        getattr(getattr(base, "gpt_neox", None), "layers", None),
        getattr(getattr(base, "decoder", None), "layers", None),  # OPT
    ]
    
    for c in candidates:
        if c is not None and hasattr(c, "__len__") and len(c) >= 2:
            return c
    
    raise ValueError("Cannot locate transformer blocks. Unsupported model architecture.")


def attach_schemabank_last2(model, H, S=32, r=16, topk=2, ad=32):
    """
    Attach SchemaBank adapters as proper submodules (V3 method - FIXED)
    """
    blocks = get_block_list(model)
    idxs = [-2, -1]
    
    try:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
    
    adapters = []
    
    # Attach SchemaBank as a proper submodule to each block
    for i, block_idx in enumerate(idxs):
        block = blocks[block_idx]
        
        # Create adapter and register it as a submodule
        adapter = SchemaBank(H, S, r, topk, ad).to(device=device, dtype=dtype)
        block.add_module('schemabank_adapter', adapter)
        adapters.append(adapter)
        
        # Store original forward for this specific block
        original_forward = block.forward
        
        # Create wrapper that captures THIS specific adapter
        def make_forward_wrapper(orig_fwd, sb_adapter):
            def new_forward(hidden_states, *args, **kwargs):
                # Call original forward
                output = orig_fwd(hidden_states, *args, **kwargs)
                
                # Transform the output using THIS adapter
                if isinstance(output, tuple):
                    hidden_states_out = output[0]
                    extras = output[1:]
                    transformed = sb_adapter(hidden_states_out)
                    return (transformed,) + extras
                else:
                    return sb_adapter(output)
            
            return new_forward
        
        # Replace forward method with wrapped version
        block.forward = make_forward_wrapper(original_forward, adapter)
    
    # Store reference to adapters on the model
    model.schemabank_adapters = nn.ModuleList(adapters)
    
    return model


def get_schemabank_parameters(model):
    """
    Get SchemaBank parameters for selective freezing/training
    
    Returns dict with:
        - router_params: Router weights
        - schema_params: U, V transformation matrices
        - gate_params: Attribute and projection parameters
    """
    if not hasattr(model, 'schemabank_adapters'):
        return None
    
    router_params = []
    schema_params = []
    gate_params = []
    
    for adapter in model.schemabank_adapters:
        router_params.append(adapter.router.weight)
        schema_params.extend([adapter.U, adapter.V])
        gate_params.extend([adapter.attr, adapter.aproj.weight])
    
    return {
        'router': router_params,
        'schemas': schema_params,
        'gates': gate_params
    }
