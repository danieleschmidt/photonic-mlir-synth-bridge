"""
MeshScheduler - Pipeline scheduling for photonic MZI mesh circuits.

Implements Clements decomposition to generate efficient pipeline stages
for NxN unitary MZI meshes used in photonic neural networks.
"""

from typing import List, Tuple
import numpy as np


class MeshScheduler:
    """
    Pipeline scheduler for photonic MZI mesh circuits.

    Computes pipeline stages for a Clements decomposition of an N-port
    unitary mesh. Each stage is a set of MZIs that can operate in parallel
    (non-overlapping waveguide pairs).
    """

    def __init__(self, mesh_size: int):
        """
        Initialize the mesh scheduler.

        Args:
            mesh_size: Dimension of the mesh (number of waveguides)
        """
        self.mesh_size = mesh_size
        self._stages: List[List[Tuple[int, int]]] = []
        self._computed = False

    def schedule_clements(self, size: int) -> "MeshScheduler":
        """
        Generate Clements decomposition pipeline stages for an NxN mesh.

        The Clements decomposition arranges N*(N-1)/2 MZIs in N-1 columns
        (pipeline stages). Odd columns operate on pairs (0,1),(2,3),...
        and even columns on pairs (1,2),(3,4),...

        Args:
            size: Mesh size (N). Updates internal state.

        Returns:
            self (for chaining)
        """
        self.mesh_size = size
        self._stages = []

        # Clements layout: alternating even/odd column pairings
        for col in range(size - 1):
            stage = []
            start = col % 2  # 0 for even columns, 1 for odd
            for row in range(start, size - 1, 2):
                # MZI connecting waveguides (row, row+1)
                # MZI index: encode as (row, col) pair
                stage.append((row, col))
            if stage:
                self._stages.append(stage)

        self._computed = True
        return self

    def _ensure_scheduled(self):
        """Lazily compute schedule if not already done."""
        if not self._computed:
            self.schedule_clements(self.mesh_size)

    def get_stages(self) -> List[List[Tuple[int, int]]]:
        """
        Return the list of pipeline stages.

        Each stage is a list of (row, col) MZI indices representing
        MZIs that operate in parallel within that pipeline stage.

        Returns:
            List of stages, each stage a list of (row, col) tuples
        """
        self._ensure_scheduled()
        return list(self._stages)

    def get_depth(self) -> int:
        """
        Return the total pipeline depth (number of stages).

        Returns:
            Number of pipeline stages
        """
        self._ensure_scheduled()
        return len(self._stages)

    def get_critical_path(self) -> List[Tuple[int, int]]:
        """
        Return the critical path through the mesh.

        The critical path is the sequence of MZIs along the longest
        path from any input to any output. For a Clements mesh, this
        traverses one MZI per pipeline stage.

        Returns:
            List of (row, col) MZI indices forming the critical path
        """
        self._ensure_scheduled()
        if not self._stages:
            return []

        # For Clements, critical path = one MZI from each stage
        # Trace the path that follows waveguide 0 through the mesh
        path = []
        current_row = 0
        for col, stage in enumerate(self._stages):
            # Find MZI in this stage involving current_row
            for (row, c) in stage:
                if row == current_row or row + 1 == current_row:
                    path.append((row, c))
                    # After this MZI, track the lower output
                    current_row = row + 1
                    break
            else:
                # No MZI in this stage touches current_row; row passes through
                # but still counts for depth
                if stage:
                    path.append(stage[0])

        return path

    def get_mzi_count(self) -> int:
        """Return total number of MZIs in the mesh."""
        self._ensure_scheduled()
        return sum(len(s) for s in self._stages)

    def get_parallelism(self) -> List[int]:
        """Return the number of parallel MZIs per stage."""
        self._ensure_scheduled()
        return [len(s) for s in self._stages]
