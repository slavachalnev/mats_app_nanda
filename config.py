from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SAEConfig:
    
    # Data Generating Function (Model + Training Distibuion)
    model_name: str = "gpt2-small"
    hook_point: str = "blocks.{layer}.hook_resid_pre"
    hook_point_layer: int = 2
    dataset_path = "Skylion007/openwebtext",
    is_dataset_tokenized=False,
    context_size: int = 128

    # SAE Parameters
    d_in: int = 768,
    expansion_factor: int = 4
    from_pretrained_path: Optional[str] = None
    d_sae: Optional[int] = None

    # Training Parameters
    l1_coefficient: float = 1e-3
    lp_norm: float = 1
    lr: float = 3e-4
    lr_scheduler_name: str = (
        "constantwithwarmup"  # constant, constantwithwarmup, linearwarmupdecay, cosineannealing, cosineannealingwarmup
    )
    lr_warm_up_steps: int = 500
    train_batch_size: int = 4096

    dead_feature_threshold: float = 1e-8

    # WANDB
    log_to_wandb: bool = False
    wandb_project: Optional[str] = None
    run_name: Optional[str] = None
    wandb_log_frequency: int = 100

    # Misc
    device: str = "cpu"
    # dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.log_to_wandb and self.wandb_project is None:
            raise ValueError("If log_to_wandb is True, wandb_project must be set.")
        if self.d_sae is None:
            self.d_sae = self.d_in * self.expansion_factor
        if self.run_name is None:
            self.run_name = f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}"

    def to_dict(self):
        return asdict(self)

