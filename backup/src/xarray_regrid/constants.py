from enum import Enum


class GridType(Enum):
    """Enumeration of grid types supported by the regridder."""

    RECTILINEAR = "rectilinear"
    """Rectilinear grid where coordinates are 1D arrays."""

    CURVILINEAR = "curvilinear"
    """Curvilinear grid where coordinates are 2D arrays."""


# Grid validation constants
GRID_TYPE_VALIDATION_ERROR = (
    "Unsupported grid type detected. "
    "Only rectilinear and curvilinear grids are supported."
)
