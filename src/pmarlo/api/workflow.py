import logging
from typing import Any, Optional
from pathlib import Path

from pmarlo.config import JOINT_USE_REWEIGHT
from ..workflow.joint import WorkflowConfig as JointWorkflowConfig
from ..workflow.joint import JointWorkflow

logger = logging.getLogger("pmarlo")


def build_joint_workflow(
    shards_root: Path,
    temperature_ref_K: float,
    tau_steps: int,
    n_clusters: int,
    *,
    use_reweight: Optional[bool] = None,
) -> Any:
    """Construct a :class:`JointWorkflow` using library defaults only."""

    if use_reweight is None:
        use_reweight = JOINT_USE_REWEIGHT.get()

    logger.info(
        "[workflow] Building joint workflow: tau_steps=%d, n_clusters=%d, temperature=%.1fK, use_reweight=%s",
        tau_steps,
        n_clusters,
        temperature_ref_K,
        use_reweight,
    )
    logger.debug("[workflow] Shards root directory: %s", shards_root)

    cfg = JointWorkflowConfig(
        shards_root=Path(shards_root),
        temperature_ref_K=temperature_ref_K,
        tau_steps=int(tau_steps),
        n_clusters=int(n_clusters),
        use_reweight=bool(use_reweight),
    )

    workflow = JointWorkflow(cfg)
    logger.debug("[workflow] JointWorkflow instance created successfully")

    return workflow  # type: ignore[no-any-return]
