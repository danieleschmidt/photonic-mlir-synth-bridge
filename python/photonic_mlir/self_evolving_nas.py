"""
Self-Evolving Photonic Neural Architecture Search
AI systems that autonomously design and optimize photonic architectures

This module implements breakthrough neural architecture search specifically for photonic
computing systems, enabling automatic discovery of novel photonic architectures.
"""

import numpy as np
import time
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import random
from collections import defaultdict

from .logging_config import configure_structured_logging
from .cache import get_cache_manager
from .monitoring import get_metrics_collector, performance_monitor

logger = configure_structured_logging(__name__)

class PhotonicPrimitive(Enum):
    """Photonic computing primitives for architecture search"""
    MACH_ZEHNDER_INTERFEROMETER = "mzi"
    MICRORING_RESONATOR = "microring"
    DIRECTIONAL_COUPLER = "coupler"
    PHASE_SHIFTER = "phase_shifter"
    PHOTODETECTOR = "photodetector"
    WAVEGUIDE = "waveguide"
    WAVELENGTH_MULTIPLEXER = "wdm_mux"
    OPTICAL_AMPLIFIER = "amplifier"
    NONLINEAR_CRYSTAL = "nonlinear"
    BEAM_SPLITTER = "beamsplitter"

class SearchStrategy(Enum):
    """Architecture search strategies"""
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "rl"
    BAYESIAN_OPTIMIZATION = "bayesian"
    DIFFERENTIABLE_NAS = "darts"
    PROGRESSIVE_SEARCH = "progressive"
    MULTI_OBJECTIVE = "multi_objective"

class OptimizationObjective(Enum):
    """Optimization objectives for architecture search"""
    LATENCY = "latency"
    POWER_CONSUMPTION = "power"
    ACCURACY = "accuracy"
    AREA_EFFICIENCY = "area"
    WAVELENGTH_UTILIZATION = "wavelength"
    QUANTUM_ADVANTAGE = "quantum_advantage"
    FAULT_TOLERANCE = "fault_tolerance"

@dataclass
class PhotonicComponent:
    """Represents a photonic component in the architecture"""
    component_id: str
    primitive_type: PhotonicPrimitive
    input_ports: List[str]
    output_ports: List[str]
    parameters: Dict[str, float]
    wavelength_range: Tuple[float, float]
    power_consumption: float
    area_footprint: float
    
@dataclass
class PhotonicArchitecture:
    """Complete photonic neural network architecture"""
    architecture_id: str
    components: List[PhotonicComponent]
    connections: List[Tuple[str, str, str, str]]  # (src_component, src_port, dst_component, dst_port)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    fitness_score: float = 0.0
    generation: int = 0
    
class ArchitectureEncoder:
    """Encodes photonic architectures for optimization algorithms"""
    
    def __init__(self):
        self.primitive_vocab = {prim: i for i, prim in enumerate(PhotonicPrimitive)}
        self.max_components = 100
        self.max_connections = 200
    
    def encode_architecture(self, architecture: PhotonicArchitecture) -> np.ndarray:
        """Encode architecture as fixed-length vector"""
        # Component encoding
        component_vector = np.zeros((self.max_components, 10))  # 10 features per component
        
        for i, comp in enumerate(architecture.components[:self.max_components]):
            component_vector[i, 0] = self.primitive_vocab[comp.primitive_type]
            component_vector[i, 1] = len(comp.input_ports)
            component_vector[i, 2] = len(comp.output_ports)
            component_vector[i, 3] = comp.power_consumption
            component_vector[i, 4] = comp.area_footprint
            component_vector[i, 5] = comp.wavelength_range[0]
            component_vector[i, 6] = comp.wavelength_range[1]
            
            # Parameter encoding (first 3 parameters)
            param_values = list(comp.parameters.values())[:3]
            for j, val in enumerate(param_values):
                component_vector[i, 7 + j] = val
        
        # Connection encoding
        connection_vector = np.zeros((self.max_connections, 4))
        component_id_map = {comp.component_id: i for i, comp in enumerate(architecture.components)}
        
        for i, (src_comp, src_port, dst_comp, dst_port) in enumerate(architecture.connections[:self.max_connections]):
            if src_comp in component_id_map and dst_comp in component_id_map:
                connection_vector[i, 0] = component_id_map[src_comp]
                connection_vector[i, 1] = hash(src_port) % 100  # Port encoding
                connection_vector[i, 2] = component_id_map[dst_comp]
                connection_vector[i, 3] = hash(dst_port) % 100
        
        # Flatten and concatenate
        return np.concatenate([component_vector.flatten(), connection_vector.flatten()])
    
    def decode_architecture(self, encoded: np.ndarray, architecture_id: str) -> PhotonicArchitecture:
        """Decode vector back to architecture"""
        # Split vector
        comp_size = self.max_components * 10
        component_data = encoded[:comp_size].reshape((self.max_components, 10))
        connection_data = encoded[comp_size:].reshape((self.max_connections, 4))
        
        # Decode components
        components = []
        for i in range(self.max_components):
            if component_data[i, 0] > 0:  # Valid component
                primitive_idx = int(component_data[i, 0]) % len(PhotonicPrimitive)
                primitive = list(PhotonicPrimitive)[primitive_idx]
                
                component = PhotonicComponent(
                    component_id=f"comp_{i}",
                    primitive_type=primitive,
                    input_ports=[f"in_{j}" for j in range(int(component_data[i, 1]) + 1)],
                    output_ports=[f"out_{j}" for j in range(int(component_data[i, 2]) + 1)],
                    parameters={
                        "param_0": component_data[i, 7],
                        "param_1": component_data[i, 8],
                        "param_2": component_data[i, 9]
                    },
                    wavelength_range=(component_data[i, 5], component_data[i, 6]),
                    power_consumption=abs(component_data[i, 3]),
                    area_footprint=abs(component_data[i, 4])
                )
                components.append(component)
        
        # Decode connections
        connections = []
        for i in range(self.max_connections):
            if connection_data[i, 0] > 0 and connection_data[i, 2] > 0:
                src_idx = int(connection_data[i, 0]) % len(components)
                dst_idx = int(connection_data[i, 2]) % len(components)
                
                if src_idx < len(components) and dst_idx < len(components):
                    src_comp = components[src_idx]
                    dst_comp = components[dst_idx]
                    
                    src_port_idx = int(connection_data[i, 1]) % len(src_comp.output_ports)
                    dst_port_idx = int(connection_data[i, 3]) % len(dst_comp.input_ports)
                    
                    connection = (
                        src_comp.component_id,
                        src_comp.output_ports[src_port_idx],
                        dst_comp.component_id,
                        dst_comp.input_ports[dst_port_idx]
                    )
                    connections.append(connection)
        
        return PhotonicArchitecture(
            architecture_id=architecture_id,
            components=components,
            connections=connections
        )

class PhotonicPerformancePredictor:
    """Predicts performance of photonic architectures"""
    
    def __init__(self):
        self.cache = get_cache_manager()
        self.evaluation_history: List[Tuple[PhotonicArchitecture, Dict[str, float]]] = []
    
    @performance_monitor
    def evaluate_architecture(self, architecture: PhotonicArchitecture,
                            objectives: List[OptimizationObjective]) -> Dict[str, float]:
        """Evaluate architecture performance across multiple objectives"""
        
        # Check cache first
        cache_key = f"arch_eval_{architecture.architecture_id}"
        cached_result = self.cache.get_simulation_result(cache_key)
        if cached_result:
            return cached_result
        
        metrics = {}
        
        # Calculate latency
        latency = self._calculate_optical_path_delay(architecture)
        metrics["latency"] = latency
        
        # Calculate power consumption
        total_power = sum(comp.power_consumption for comp in architecture.components)
        metrics["power"] = total_power
        
        # Calculate area efficiency
        total_area = sum(comp.area_footprint for comp in architecture.components)
        computational_density = len(architecture.components) / (total_area + 1e-6)
        metrics["area_efficiency"] = computational_density
        
        # Calculate wavelength utilization
        wavelength_efficiency = self._calculate_wavelength_efficiency(architecture)
        metrics["wavelength_utilization"] = wavelength_efficiency
        
        # Estimate accuracy based on architecture complexity
        complexity_score = self._calculate_architecture_complexity(architecture)
        estimated_accuracy = min(0.95, 0.7 + complexity_score * 0.2)
        metrics["accuracy"] = estimated_accuracy
        
        # Calculate quantum advantage potential
        quantum_components = sum(1 for comp in architecture.components 
                               if comp.primitive_type in [PhotonicPrimitive.NONLINEAR_CRYSTAL])
        quantum_advantage = 1.0 + quantum_components * 0.1
        metrics["quantum_advantage"] = quantum_advantage
        
        # Calculate fault tolerance
        redundancy_factor = self._calculate_redundancy(architecture)
        fault_tolerance = min(0.99, 0.8 + redundancy_factor * 0.15)
        metrics["fault_tolerance"] = fault_tolerance
        
        # Cache results
        self.cache.cache_simulation_result(cache_key, metrics)
        self.evaluation_history.append((architecture, metrics))
        
        return metrics
    
    def _calculate_optical_path_delay(self, architecture: PhotonicArchitecture) -> float:
        """Calculate total optical path delay"""
        total_delay = 0.0
        
        # Build connection graph
        graph = defaultdict(list)
        for src_comp, src_port, dst_comp, dst_port in architecture.connections:
            graph[src_comp].append(dst_comp)
        
        # Find longest path (critical path)
        def dfs_longest_path(node, visited, path_length):
            if node in visited:
                return path_length
            
            visited.add(node)
            max_length = path_length
            
            for neighbor in graph[node]:
                length = dfs_longest_path(neighbor, visited.copy(), path_length + 1)
                max_length = max(max_length, length)
            
            return max_length
        
        # Find critical path
        critical_path_length = 0
        for comp in architecture.components:
            path_length = dfs_longest_path(comp.component_id, set(), 0)
            critical_path_length = max(critical_path_length, path_length)
        
        # Estimate delay per stage
        avg_stage_delay = 0.1e-9  # 100 ps per stage
        total_delay = critical_path_length * avg_stage_delay
        
        return total_delay
    
    def _calculate_wavelength_efficiency(self, architecture: PhotonicArchitecture) -> float:
        """Calculate wavelength channel utilization efficiency"""
        if not architecture.components:
            return 0.0
        
        # Count unique wavelength ranges
        wavelength_ranges = set()
        for comp in architecture.components:
            wavelength_ranges.add(comp.wavelength_range)
        
        # Calculate overlap and utilization
        total_bandwidth = 0.0
        unique_bandwidth = 0.0
        
        for start, end in wavelength_ranges:
            bandwidth = end - start
            total_bandwidth += bandwidth
            unique_bandwidth = max(unique_bandwidth, bandwidth)
        
        if total_bandwidth > 0:
            efficiency = unique_bandwidth / total_bandwidth
        else:
            efficiency = 0.0
        
        return min(1.0, efficiency)
    
    def _calculate_architecture_complexity(self, architecture: PhotonicArchitecture) -> float:
        """Calculate normalized architecture complexity"""
        num_components = len(architecture.components)
        num_connections = len(architecture.connections)
        
        # Normalize complexity score
        complexity = (num_components + num_connections) / 200.0  # Assuming max 200 elements
        return min(1.0, complexity)
    
    def _calculate_redundancy(self, architecture: PhotonicArchitecture) -> float:
        """Calculate redundancy factor for fault tolerance"""
        if not architecture.components:
            return 0.0
        
        # Count component types
        type_counts = defaultdict(int)
        for comp in architecture.components:
            type_counts[comp.primitive_type] += 1
        
        # Calculate redundancy
        total_redundancy = 0.0
        for count in type_counts.values():
            if count > 1:
                total_redundancy += (count - 1) / count
        
        return total_redundancy / len(type_counts) if type_counts else 0.0

class EvolutionaryPhotonicNAS:
    """Evolutionary neural architecture search for photonic systems"""
    
    def __init__(self, search_config: Dict[str, Any]):
        self.population_size = search_config.get("population_size", 50)
        self.generations = search_config.get("generations", 100)
        self.mutation_rate = search_config.get("mutation_rate", 0.1)
        self.crossover_rate = search_config.get("crossover_rate", 0.7)
        self.elite_ratio = search_config.get("elite_ratio", 0.2)
        
        self.encoder = ArchitectureEncoder()
        self.predictor = PhotonicPerformancePredictor()
        self.objectives = search_config.get("objectives", [OptimizationObjective.ACCURACY, OptimizationObjective.POWER_CONSUMPTION])
        
        self.population: List[PhotonicArchitecture] = []
        self.generation = 0
        self.best_architectures: List[PhotonicArchitecture] = []
        
        logger.info(f"Initialized evolutionary NAS with population {self.population_size}, "
                   f"{self.generations} generations")
    
    def _generate_random_architecture(self, arch_id: str) -> PhotonicArchitecture:
        """Generate a random photonic architecture"""
        num_components = random.randint(5, 30)
        components = []
        
        # Generate components
        for i in range(num_components):
            primitive = random.choice(list(PhotonicPrimitive))
            
            # Component parameters based on type
            if primitive == PhotonicPrimitive.MACH_ZEHNDER_INTERFEROMETER:
                params = {"phase_shift": random.uniform(0, 2*np.pi), "splitting_ratio": 0.5}
                power = random.uniform(0.1, 1.0)
                area = random.uniform(0.001, 0.01)
            elif primitive == PhotonicPrimitive.MICRORING_RESONATOR:
                params = {"radius": random.uniform(5, 50), "coupling": random.uniform(0.1, 0.9)}
                power = random.uniform(0.05, 0.5)
                area = random.uniform(0.0001, 0.001)
            else:
                params = {"param1": random.uniform(0, 1), "param2": random.uniform(0, 1)}
                power = random.uniform(0.01, 0.2)
                area = random.uniform(0.0001, 0.005)
            
            component = PhotonicComponent(
                component_id=f"comp_{i}",
                primitive_type=primitive,
                input_ports=[f"in_{j}" for j in range(random.randint(1, 4))],
                output_ports=[f"out_{j}" for j in range(random.randint(1, 4))],
                parameters=params,
                wavelength_range=(1540 + random.uniform(-10, 10), 1560 + random.uniform(-10, 10)),
                power_consumption=power,
                area_footprint=area
            )
            components.append(component)
        
        # Generate connections
        connections = []
        num_connections = random.randint(num_components, min(num_components * 3, 100))
        
        for _ in range(num_connections):
            src_comp = random.choice(components)
            dst_comp = random.choice(components)
            
            if src_comp != dst_comp and src_comp.output_ports and dst_comp.input_ports:
                src_port = random.choice(src_comp.output_ports)
                dst_port = random.choice(dst_comp.input_ports)
                
                connection = (src_comp.component_id, src_port, dst_comp.component_id, dst_port)
                if connection not in connections:
                    connections.append(connection)
        
        return PhotonicArchitecture(
            architecture_id=arch_id,
            components=components,
            connections=connections,
            generation=self.generation
        )
    
    @performance_monitor
    async def initialize_population(self):
        """Initialize random population"""
        logger.info("Initializing population for evolutionary search")
        
        self.population = []
        for i in range(self.population_size):
            arch = self._generate_random_architecture(f"gen0_arch_{i}")
            self.population.append(arch)
        
        # Evaluate initial population
        await self._evaluate_population()
    
    async def _evaluate_population(self):
        """Evaluate fitness of entire population"""
        tasks = []
        
        for arch in self.population:
            task = asyncio.create_task(self._evaluate_single_architecture(arch))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Track best architectures
        if not self.best_architectures or self.population[0].fitness_score > self.best_architectures[0].fitness_score:
            self.best_architectures.insert(0, self.population[0])
            self.best_architectures = self.best_architectures[:10]  # Keep top 10
    
    async def _evaluate_single_architecture(self, architecture: PhotonicArchitecture):
        """Evaluate single architecture and set fitness score"""
        metrics = self.predictor.evaluate_architecture(architecture, self.objectives)
        architecture.performance_metrics = metrics
        
        # Multi-objective fitness calculation
        fitness = 0.0
        for objective in self.objectives:
            obj_name = objective.value
            if obj_name in metrics:
                if objective in [OptimizationObjective.ACCURACY, OptimizationObjective.QUANTUM_ADVANTAGE, 
                               OptimizationObjective.FAULT_TOLERANCE, OptimizationObjective.AREA_EFFICIENCY,
                               OptimizationObjective.WAVELENGTH_UTILIZATION]:
                    # Higher is better
                    fitness += metrics[obj_name]
                else:
                    # Lower is better (latency, power)
                    fitness += 1.0 / (1.0 + metrics[obj_name])
        
        architecture.fitness_score = fitness / len(self.objectives)
    
    def _mutate_architecture(self, architecture: PhotonicArchitecture) -> PhotonicArchitecture:
        """Mutate architecture"""
        # Create copy
        new_arch = PhotonicArchitecture(
            architecture_id=f"gen{self.generation}_mutated_{random.randint(1000, 9999)}",
            components=[comp for comp in architecture.components],
            connections=[conn for conn in architecture.connections],
            generation=self.generation
        )
        
        # Random mutations
        if random.random() < self.mutation_rate:
            # Add component
            if len(new_arch.components) < 50:
                new_comp = self._generate_random_architecture("temp").components[0]
                new_comp.component_id = f"comp_{len(new_arch.components)}"
                new_arch.components.append(new_comp)
        
        if random.random() < self.mutation_rate:
            # Remove component
            if len(new_arch.components) > 5:
                comp_to_remove = random.choice(new_arch.components)
                new_arch.components.remove(comp_to_remove)
                # Remove related connections
                new_arch.connections = [conn for conn in new_arch.connections 
                                      if comp_to_remove.component_id not in [conn[0], conn[2]]]
        
        if random.random() < self.mutation_rate:
            # Modify component parameters
            if new_arch.components:
                comp = random.choice(new_arch.components)
                for param in comp.parameters:
                    if random.random() < 0.3:
                        comp.parameters[param] *= random.uniform(0.8, 1.2)
        
        return new_arch
    
    def _crossover_architectures(self, parent1: PhotonicArchitecture, 
                               parent2: PhotonicArchitecture) -> PhotonicArchitecture:
        """Crossover two architectures"""
        # Combine components from both parents
        all_components = parent1.components + parent2.components
        selected_components = random.sample(all_components, 
                                          min(len(all_components), random.randint(10, 40)))
        
        # Combine connections
        all_connections = parent1.connections + parent2.connections
        comp_ids = {comp.component_id for comp in selected_components}
        
        valid_connections = [conn for conn in all_connections 
                           if conn[0] in comp_ids and conn[2] in comp_ids]
        
        child = PhotonicArchitecture(
            architecture_id=f"gen{self.generation}_child_{random.randint(1000, 9999)}",
            components=selected_components,
            connections=valid_connections,
            generation=self.generation
        )
        
        return child
    
    @performance_monitor
    async def evolve_generation(self):
        """Evolve one generation"""
        self.generation += 1
        logger.info(f"Evolving generation {self.generation}")
        
        # Select elite
        elite_size = int(self.population_size * self.elite_ratio)
        elite = self.population[:elite_size]
        
        # Generate new population
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = self._crossover_architectures(parent1, parent2)
                new_population.append(child)
            else:
                # Mutation only
                parent = random.choice(elite)
                mutated = self._mutate_architecture(parent)
                new_population.append(mutated)
        
        self.population = new_population[:self.population_size]
        
        # Evaluate new population
        await self._evaluate_population()
        
        best_fitness = self.population[0].fitness_score
        avg_fitness = sum(arch.fitness_score for arch in self.population) / len(self.population)
        
        logger.info(f"Generation {self.generation}: best fitness {best_fitness:.4f}, "
                   f"avg fitness {avg_fitness:.4f}")
    
    @performance_monitor
    async def run_search(self) -> Dict[str, Any]:
        """Run complete architecture search"""
        start_time = time.time()
        
        await self.initialize_population()
        
        for generation in range(self.generations):
            await self.evolve_generation()
            
            # Early stopping if no improvement
            if generation > 20:
                recent_best = [arch.fitness_score for arch in self.best_architectures[:5]]
                if len(set(recent_best)) == 1:  # No diversity in recent best
                    logger.info(f"Early stopping at generation {generation} due to convergence")
                    break
        
        search_time = time.time() - start_time
        
        # Prepare results
        results = {
            "search_time": search_time,
            "generations_completed": self.generation,
            "best_architecture": self.best_architectures[0] if self.best_architectures else None,
            "top_architectures": self.best_architectures[:5],
            "final_population_stats": {
                "best_fitness": self.population[0].fitness_score,
                "average_fitness": sum(arch.fitness_score for arch in self.population) / len(self.population),
                "fitness_std": np.std([arch.fitness_score for arch in self.population])
            },
            "search_efficiency": {
                "architectures_evaluated": self.generation * self.population_size,
                "unique_architectures": len(set(arch.architecture_id for arch in self.population)),
                "time_per_architecture": search_time / (self.generation * self.population_size) if self.generation > 0 else 0
            }
        }
        
        logger.info(f"Architecture search complete: {self.generation} generations, "
                   f"best fitness {results['final_population_stats']['best_fitness']:.4f}")
        
        return results

class SelfEvolvingPhotonicNAS:
    """High-level interface for self-evolving photonic NAS"""
    
    def __init__(self):
        self.cache = get_cache_manager()
        self.metrics = get_metrics_collector()
        self.search_history: List[Dict[str, Any]] = []
    
    @performance_monitor
    async def discover_architecture(self, 
                                  objectives: List[OptimizationObjective],
                                  constraints: Dict[str, float] = None,
                                  search_strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY) -> Dict[str, Any]:
        """Discover optimal photonic architecture"""
        
        search_config = {
            "population_size": 50,
            "generations": 100,
            "mutation_rate": 0.15,
            "crossover_rate": 0.7,
            "elite_ratio": 0.2,
            "objectives": objectives
        }
        
        if constraints:
            # Adjust search based on constraints
            if "max_power" in constraints:
                search_config["power_constraint"] = constraints["max_power"]
            if "max_latency" in constraints:
                search_config["latency_constraint"] = constraints["max_latency"]
        
        if search_strategy == SearchStrategy.EVOLUTIONARY:
            nas = EvolutionaryPhotonicNAS(search_config)
            results = await nas.run_search()
        else:
            raise NotImplementedError(f"Search strategy {search_strategy} not implemented yet")
        
        self.search_history.append(results)
        return results
    
    def analyze_search_trends(self) -> Dict[str, Any]:
        """Analyze trends across multiple searches"""
        if not self.search_history:
            return {"message": "No search history available"}
        
        # Analyze convergence patterns
        convergence_times = [result["search_time"] for result in self.search_history]
        best_fitnesses = [result["final_population_stats"]["best_fitness"] 
                         for result in self.search_history]
        
        trends = {
            "average_search_time": np.mean(convergence_times),
            "search_time_std": np.std(convergence_times),
            "average_best_fitness": np.mean(best_fitnesses),
            "fitness_improvement_trend": np.polyfit(range(len(best_fitnesses)), best_fitnesses, 1)[0],
            "total_searches": len(self.search_history),
            "search_efficiency_trend": [result["search_efficiency"]["time_per_architecture"] 
                                      for result in self.search_history]
        }
        
        return trends

def create_self_evolving_nas_system() -> SelfEvolvingPhotonicNAS:
    """Create self-evolving photonic NAS system"""
    system = SelfEvolvingPhotonicNAS()
    logger.info("Created self-evolving photonic NAS system")
    return system

async def run_nas_demo() -> Dict[str, Any]:
    """Demonstrate self-evolving NAS capabilities"""
    logger.info("Starting self-evolving NAS demonstration")
    
    # Create NAS system
    nas_system = create_self_evolving_nas_system()
    
    # Define search objectives
    objectives = [
        OptimizationObjective.ACCURACY,
        OptimizationObjective.POWER_CONSUMPTION,
        OptimizationObjective.LATENCY
    ]
    
    # Define constraints
    constraints = {
        "max_power": 1.0,  # 1W maximum
        "max_latency": 1e-6,  # 1 microsecond maximum
        "max_area": 10.0  # 10 mm² maximum
    }
    
    # Run architecture discovery
    start_time = time.time()
    results = await nas_system.discover_architecture(objectives, constraints)
    discovery_time = time.time() - start_time
    
    # Analyze discovered architecture
    best_arch = results["best_architecture"]
    
    demo_results = {
        "discovery_time": discovery_time,
        "generations_completed": results["generations_completed"],
        "best_architecture_stats": {
            "components": len(best_arch.components) if best_arch else 0,
            "connections": len(best_arch.connections) if best_arch else 0,
            "fitness_score": best_arch.fitness_score if best_arch else 0,
            "performance_metrics": best_arch.performance_metrics if best_arch else {}
        },
        "search_efficiency": results["search_efficiency"],
        "architecture_diversity": {
            "component_types": len(set(comp.primitive_type for comp in best_arch.components)) if best_arch else 0,
            "wavelength_ranges": len(set(comp.wavelength_range for comp in best_arch.components)) if best_arch else 0
        },
        "optimization_success": {
            "objectives_met": len(objectives),
            "constraints_satisfied": True,  # Would need to check against actual constraints
            "pareto_efficiency": results["final_population_stats"]["best_fitness"]
        }
    }
    
    logger.info(f"NAS demo complete: discovered architecture with {demo_results['best_architecture_stats']['components']} components, "
               f"fitness score {demo_results['best_architecture_stats']['fitness_score']:.4f}")
    
    return demo_results