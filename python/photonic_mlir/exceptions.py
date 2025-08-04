"""
Custom exceptions for Photonic MLIR compiler.
"""


class PhotonicMLIRError(Exception):
    """Base exception for all Photonic MLIR errors"""
    pass


class CompilationError(PhotonicMLIRError):
    """Raised when PyTorch model compilation fails"""
    
    def __init__(self, message: str, model_info: str = None):
        self.model_info = model_info
        super().__init__(message)


class ValidationError(PhotonicMLIRError):
    """Raised when input validation fails"""
    
    def __init__(self, message: str, parameter: str = None, expected: str = None, got: str = None):
        self.parameter = parameter
        self.expected = expected
        self.got = got
        super().__init__(message)


class HardwareError(PhotonicMLIRError):
    """Raised when hardware operation fails"""
    
    def __init__(self, message: str, device: str = None, error_code: int = None):
        self.device = device
        self.error_code = error_code
        super().__init__(message)


class SimulationError(PhotonicMLIRError):
    """Raised when photonic simulation fails"""
    
    def __init__(self, message: str, pdk: str = None, circuit_info: str = None):
        self.pdk = pdk
        self.circuit_info = circuit_info
        super().__init__(message)


class OptimizationError(PhotonicMLIRError):
    """Raised when optimization pass fails"""
    
    def __init__(self, message: str, pass_name: str = None, stage: str = None):
        self.pass_name = pass_name
        self.stage = stage
        super().__init__(message)


class PowerBudgetExceededError(PhotonicMLIRError):
    """Raised when circuit exceeds power budget"""
    
    def __init__(self, estimated_power: float, budget: float):
        self.estimated_power = estimated_power
        self.budget = budget
        super().__init__(
            f"Circuit power consumption ({estimated_power:.2f}mW) exceeds budget ({budget:.2f}mW)"
        )


class WavelengthConflictError(PhotonicMLIRError):
    """Raised when wavelength allocation conflicts occur"""
    
    def __init__(self, conflicting_wavelengths: list):
        self.conflicting_wavelengths = conflicting_wavelengths
        super().__init__(
            f"Wavelength conflicts detected: {conflicting_wavelengths}"
        )


class ThermalViolationError(PhotonicMLIRError):
    """Raised when thermal constraints are violated"""
    
    def __init__(self, max_temp: float, limit: float, location: str = None):
        self.max_temp = max_temp
        self.limit = limit
        self.location = location
        super().__init__(
            f"Thermal limit exceeded: {max_temp:.1f}K > {limit:.1f}K"
            + (f" at {location}" if location else "")
        )


class CalibrationError(HardwareError):
    """Raised when hardware calibration fails"""
    
    def __init__(self, message: str, device: str = None, failed_component: str = None):
        self.failed_component = failed_component
        super().__init__(message, device)


class UnsupportedOperationError(PhotonicMLIRError):
    """Raised when an unsupported operation is encountered"""
    
    def __init__(self, operation: str, dialect: str = None):
        self.operation = operation
        self.dialect = dialect
        super().__init__(
            f"Unsupported operation: {operation}"
            + (f" in {dialect} dialect" if dialect else "")
        )