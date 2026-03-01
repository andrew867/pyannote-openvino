import warnings

# Apply at import time so it is active during test collection imports.
warnings.filterwarnings(
    "ignore",
    message=r"(?s)\s*torchcodec is not installed correctly.*",
    module=r"pyannote\.audio\.core\.io",
)
