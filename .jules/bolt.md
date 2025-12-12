## 2024-10-26 - Robust Caching for Xarray Objects

**Learning:** Caching based on `id()` is fragile. Two `xarray` objects can be identical in value but have different memory addresses, leading to cache misses. A robust caching strategy must be based on the object's content, not its identity.

**Action:** When implementing caching for `xarray` objects, create a key based on the object's metadata, such as the coordinates and dimensions. This can be achieved by creating a hashable representation of the relevant attributes.
