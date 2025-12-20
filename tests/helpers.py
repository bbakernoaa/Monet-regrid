
import numpy as np
import xarray as xr

class Grid:
    """A helper class to create rectilinear grids for testing."""
    def __init__(self, north, east, south, west, resolution_lat, resolution_lon):
        self.north = north
        self.east = east
        self.south = south
        self.west = west
        self.resolution_lat = resolution_lat
        self.resolution_lon = resolution_lon

    def create_regridding_dataset(self):
        """Creates an xarray Dataset representing the grid."""
        if np.remainder((self.north - self.south), self.resolution_lat) > 0:
            lat = np.arange(self.south, self.north, self.resolution_lat)
        else:
            lat = np.arange(self.south, self.north + self.resolution_lat, self.resolution_lat)

        if np.remainder((self.east - self.west), self.resolution_lon) > 0:
            lon = np.arange(self.west, self.east, self.resolution_lon)
        else:
            lon = np.arange(self.west, self.east + self.resolution_lon, self.resolution_lon)
        return xr.Dataset(coords={'latitude': lat, 'longitude': lon})
