"""Generate Triton config.pbtxt files for PatchCore ensemble deployment.

Produces three config types:
  - Extractor (ONNX Runtime backend)
  - Scorer (Python backend, per-category)
  - Ensemble (routes IMAGE → extractor → scorer)
"""


def generate_extractor_config(extractor_name: str) -> str:
    """Config for the shared DINOv3 ONNX feature extractor."""
    return f"""name: "{extractor_name}"
platform: "onnxruntime_onnx"
max_batch_size: 0
input [
  {{
    name: "IMAGE"
    data_type: TYPE_FP32
    dims: [ -1, 3, -1, -1 ]
  }}
]
output [
  {{
    name: "FEATURES"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }},
  {{
    name: "PATCH_SHAPES"
    data_type: TYPE_INT64
    dims: [ -1, 2 ]
  }}
]
"""


def generate_scorer_config(scorer_name: str) -> str:
    """Config for a per-category Python backend scorer."""
    return f"""name: "{scorer_name}"
backend: "python"
max_batch_size: 0
input [
  {{
    name: "FEATURES"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }},
  {{
    name: "PATCH_SHAPES"
    data_type: TYPE_INT64
    dims: [ -1, 2 ]
  }}
]
output [
  {{
    name: "SCORE"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }},
  {{
    name: "MASK"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
  }}
]
"""


def generate_ensemble_config(
    ensemble_name: str, extractor_name: str, scorer_name: str
) -> str:
    """Config for the ensemble that chains extractor → scorer."""
    return f"""name: "{ensemble_name}"
platform: "ensemble"
max_batch_size: 0
input [
  {{
    name: "IMAGE"
    data_type: TYPE_FP32
    dims: [ -1, 3, -1, -1 ]
  }}
]
output [
  {{
    name: "SCORE"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }},
  {{
    name: "MASK"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
  }}
]
ensemble_scheduling {{
  step [
    {{
      model_name: "{extractor_name}"
      model_version: -1
      input_map {{
        key: "IMAGE"
        value: "IMAGE"
      }}
      output_map {{
        key: "FEATURES"
        value: "features_to_scorer"
      }}
      output_map {{
        key: "PATCH_SHAPES"
        value: "shapes_to_scorer"
      }}
    }},
    {{
      model_name: "{scorer_name}"
      model_version: -1
      input_map {{
        key: "FEATURES"
        value: "features_to_scorer"
      }}
      input_map {{
        key: "PATCH_SHAPES"
        value: "shapes_to_scorer"
      }}
      output_map {{
        key: "SCORE"
        value: "SCORE"
      }}
      output_map {{
        key: "MASK"
        value: "MASK"
      }}
    }}
  ]
}}
"""
