"""Accelerator models for Optical Neural Networks."""

from .architecture import ONNTile, MZITile
from .arithmetic import PartialProduct, bit_serial_multiply, build_partial_products, shift_accumulate
from .core import dot_product_bit_serial, mvm_bit_serial, mvm_reference
from .inference import InferenceEngine, InferenceResult
from .metrics import MetricsEngine, MZIMetricsEngine, ElectronicBaselineEngine, PPAMetrics
from .omac import OMAC, MZI_OMAC
