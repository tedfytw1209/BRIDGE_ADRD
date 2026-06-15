import wandb
import config


def init_wandb(job_type, run_name=None, group=None, extra_config=None):
    """Initialize a wandb run. Call once per script execution.

    Args:
        job_type: One of 'training', 'imputation', 'classification'.
        run_name: Optional display name for the run.
        group: Optional group name (e.g. derived from output_dir) to cluster
               all pipeline steps for one experiment together in the W&B UI.
        extra_config: Optional dict merged into logged config.
    """
    cfg = dict(config.config)
    if extra_config:
        cfg.update(extra_config)
    wandb.init(
        project="PSN-GNN-Research",
        job_type=job_type,
        name=run_name,
        group=group,
        config=cfg,
    )


def finish_wandb():
    """Finish the current wandb run."""
    wandb.finish()
