"""
WavelengthMultiplexer - WDM channel assignment for photonic data paths.

Manages wavelength-division multiplexing (WDM) channel allocation for
parallel data paths in silicon photonic AI accelerators.
"""

from typing import Dict, List, Optional
import numpy as np


class WavelengthMultiplexer:
    """
    Wavelength-Division Multiplexing (WDM) channel manager.

    Assigns optical wavelengths to parallel data paths in a photonic
    accelerator. Supports standard C-band WDM grid allocation.
    """

    def __init__(
        self,
        num_wavelengths: int,
        base_wavelength_nm: float = 1550.0,
        spacing_nm: float = 0.8,
    ):
        """
        Initialize the WDM multiplexer.

        Args:
            num_wavelengths: Total number of WDM channels available
            base_wavelength_nm: Shortest wavelength in nm (default: 1550.0 nm, C-band center)
            spacing_nm: Channel spacing in nm (default: 0.8 nm ≈ 100 GHz ITU grid)
        """
        self.num_wavelengths = num_wavelengths
        self.base_wavelength_nm = base_wavelength_nm
        self.spacing_nm = spacing_nm

        # Compute wavelength grid
        self._wavelengths = np.array(
            [base_wavelength_nm + i * spacing_nm for i in range(num_wavelengths)]
        )

        # Assignment state: path_id -> channel_index
        self._assignments: Dict[int, int] = {}

    @property
    def wavelengths(self) -> np.ndarray:
        """Return the WDM wavelength grid in nm."""
        return self._wavelengths.copy()

    def assign_wavelengths(self, data_paths: List[int]) -> Dict[int, float]:
        """
        Assign wavelengths to a list of parallel data path IDs.

        Wavelengths are assigned round-robin from the WDM grid.
        If more paths than wavelengths, paths wrap around (wavelength reuse
        on separate spatial modes).

        Args:
            data_paths: List of data path IDs to assign

        Returns:
            Dict mapping path_id -> assigned wavelength in nm
        """
        result: Dict[int, float] = {}
        for i, path_id in enumerate(data_paths):
            channel_idx = i % self.num_wavelengths
            self._assignments[path_id] = channel_idx
            result[path_id] = float(self._wavelengths[channel_idx])
        return result

    def get_channel(self, path_id: int) -> float:
        """
        Get the wavelength assigned to a specific data path.

        Args:
            path_id: Data path identifier

        Returns:
            Assigned wavelength in nm

        Raises:
            KeyError: If path_id has not been assigned a wavelength
        """
        if path_id not in self._assignments:
            raise KeyError(
                f"Path {path_id} has not been assigned a wavelength. "
                f"Call assign_wavelengths() first."
            )
        channel_idx = self._assignments[path_id]
        return float(self._wavelengths[channel_idx])

    def demultiplex(self, assignments: Dict[int, float]) -> Dict[float, List[int]]:
        """
        Reverse a wavelength assignment: group path IDs by their wavelength.

        Args:
            assignments: Dict of path_id -> wavelength_nm (from assign_wavelengths)

        Returns:
            Dict mapping wavelength_nm -> list of path_ids sharing that wavelength
        """
        demux: Dict[float, List[int]] = {}
        for path_id, wavelength in assignments.items():
            if wavelength not in demux:
                demux[wavelength] = []
            demux[wavelength].append(path_id)
        return demux

    def channel_count(self) -> int:
        """
        Return the total number of WDM channels available.

        Returns:
            Number of wavelength channels
        """
        return self.num_wavelengths

    def get_channel_index(self, path_id: int) -> int:
        """
        Get the channel index (0-based) for a path.

        Args:
            path_id: Data path identifier

        Returns:
            Channel index

        Raises:
            KeyError: If path_id has not been assigned
        """
        if path_id not in self._assignments:
            raise KeyError(f"Path {path_id} not assigned.")
        return self._assignments[path_id]

    def frequency_thz(self, wavelength_nm: float) -> float:
        """
        Convert wavelength in nm to frequency in THz.

        Args:
            wavelength_nm: Wavelength in nanometers

        Returns:
            Frequency in THz
        """
        c_nm_per_s = 2.998e17  # speed of light in nm/s
        return c_nm_per_s / wavelength_nm / 1e12

    def __repr__(self) -> str:
        return (
            f"WavelengthMultiplexer("
            f"channels={self.num_wavelengths}, "
            f"base={self.base_wavelength_nm}nm, "
            f"spacing={self.spacing_nm}nm)"
        )
