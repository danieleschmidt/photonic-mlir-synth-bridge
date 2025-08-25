"""
Enhanced import utilities with comprehensive fallback support
"""

import sys
import warnings
import logging
from typing import Optional, Any, Dict, Union

logger = logging.getLogger(__name__)

class GracefulImporter:
    """Handles imports with comprehensive fallback support"""
    
    def __init__(self):
        self.failed_imports = set()
        self.fallback_cache = {}
        
    def safe_import(self, module_name: str, fallback_value: Any = None, 
                   warn: bool = True) -> Any:
        """Safely import module with fallback"""
        
        if module_name in self.failed_imports:
            return self.fallback_cache.get(module_name, fallback_value)
            
        try:
            module = __import__(module_name)
            return module
        except ImportError as e:
            self.failed_imports.add(module_name)
            self.fallback_cache[module_name] = fallback_value
            
            if warn:
                logger.warning(f"Failed to import {module_name}, using fallback: {e}")
                
            return fallback_value
    
    def import_with_alternatives(self, primary: str, 
                               alternatives: List[str]) -> Optional[Any]:
        """Try importing from list of alternatives"""
        
        for module_name in [primary] + alternatives:
            try:
                return __import__(module_name)
            except ImportError:
                continue
                
        logger.error(f"Failed to import any of: {[primary] + alternatives}")
        return None

# Global importer instance
_importer = GracefulImporter()

def safe_import(module_name: str, fallback_value: Any = None) -> Any:
    """Global safe import function"""
    return _importer.safe_import(module_name, fallback_value)

def require_module(module_name: str, install_hint: str = None) -> Any:
    """Import module or raise informative error"""
    try:
        return __import__(module_name)
    except ImportError as e:
        hint = f"\nInstall with: {install_hint}" if install_hint else ""
        raise ImportError(f"Required module '{module_name}' not found.{hint}") from e
