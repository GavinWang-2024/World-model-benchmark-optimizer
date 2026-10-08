"""Plug-in optimization modules (WorldCache, AdaCache, quantization, drift correction). Concrete modules are added in Phases 5-7 (see build_plan.md).

Importing this package registers the built-in modules so they can be named in
`OptimizationStack(modules=[...])`. Registration is cheap: torch/torchao are
only imported when a module is actually applied.
"""

from worldoptbench.optimizations.adacache import AdaCacheModule
from worldoptbench.optimizations.base import (
    ModuleReport,
    OptimizationModule,
    available_modules,
    get_module_class,
    library_report,
    register_module,
)
from worldoptbench.optimizations.channels_last import ChannelsLastModule
from worldoptbench.optimizations.chunked import ChunkedRolloutModule
from worldoptbench.optimizations.cuda_graphs import CudaGraphsModule
from worldoptbench.optimizations.cudnn_benchmark import CudnnBenchmarkModule
from worldoptbench.optimizations.diffusion import (
    AttentionBackendModule,
    CfgTruncationModule,
    FasterCacheModule,
    FewerStepsModule,
    FirstBlockCacheModule,
    LayerSkipModule,
    LayerwiseCastingModule,
    PyramidAttentionBroadcastModule,
    TaylorSeerCacheModule,
)
from worldoptbench.optimizations.diffusion_more import (
    CrossAttentionKVCacheModule,
    MagCacheModule,
    SchedulerModule,
    UncondReuseModule,
    VaeModule,
)
from worldoptbench.optimizations.gumbel_sampling import GumbelSamplingModule
from worldoptbench.optimizations.latent_noise import LatentNoiseModule
from worldoptbench.optimizations.lean_scan import LeanScanModule
from worldoptbench.optimizations.low_rank import LowRankModule
from worldoptbench.optimizations.no_dist_validation import NoDistValidationModule
from worldoptbench.optimizations.precision import PrecisionModule
from worldoptbench.optimizations.quantization import QuantizationModule
from worldoptbench.optimizations.sparse_decode import SparseDecodeModule
from worldoptbench.optimizations.tensorrt_backend import TensorRTModule
from worldoptbench.optimizations.tensorrt_decoder import TensorRTDecoderModule
from worldoptbench.optimizations.tf32 import Tf32Module
from worldoptbench.optimizations.worldcache import WorldCacheModule

__all__ = [
    "AdaCacheModule",
    "AttentionBackendModule",
    "CfgTruncationModule",
    "ChannelsLastModule",
    "ChunkedRolloutModule",
    "CrossAttentionKVCacheModule",
    "CudaGraphsModule",
    "CudnnBenchmarkModule",
    "FasterCacheModule",
    "FewerStepsModule",
    "FirstBlockCacheModule",
    "GumbelSamplingModule",
    "LatentNoiseModule",
    "LayerSkipModule",
    "LayerwiseCastingModule",
    "LeanScanModule",
    "LowRankModule",
    "MagCacheModule",
    "ModuleReport",
    "NoDistValidationModule",
    "OptimizationModule",
    "PrecisionModule",
    "PyramidAttentionBroadcastModule",
    "QuantizationModule",
    "SchedulerModule",
    "SparseDecodeModule",
    "TaylorSeerCacheModule",
    "TensorRTDecoderModule",
    "TensorRTModule",
    "Tf32Module",
    "UncondReuseModule",
    "VaeModule",
    "WorldCacheModule",
    "available_modules",
    "get_module_class",
    "library_report",
    "register_module",
]
