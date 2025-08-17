"""
Global Compliance and Multi-Region Deployment Support
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path

from .logging_config import configure_structured_logging

logger = configure_structured_logging(__name__)

class Region(Enum):
    """Supported global regions"""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    PDPA = "pdpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    HIPAA = "hipaa"

@dataclass
class ComplianceConfig:
    """Compliance configuration"""
    frameworks: List[ComplianceFramework]
    data_residency_requirements: Dict[str, str]
    retention_policies: Dict[str, int]  # days
    encryption_requirements: Dict[str, str]
    audit_log_retention: int = 2555  # 7 years in days

class GlobalComplianceManager:
    """
    Global compliance and multi-region deployment manager
    """
    
    def __init__(self):
        self.compliance_configs = {}
        self.region_configs = {}
        self.active_frameworks = set()
        self._initialize_compliance_frameworks()
        
    def _initialize_compliance_frameworks(self):
        """Initialize compliance framework configurations"""
        # GDPR Configuration
        self.compliance_configs[ComplianceFramework.GDPR] = {
            "data_protection_requirements": {
                "data_minimization": True,
                "purpose_limitation": True,
                "storage_limitation": True,
                "consent_required": True,
                "right_to_erasure": True,
                "data_portability": True,
                "privacy_by_design": True
            },
            "retention_limits": {
                "personal_data": 365,  # 1 year
                "logs": 90,
                "compilation_data": 30
            },
            "encryption": {
                "data_at_rest": "AES-256",
                "data_in_transit": "TLS-1.3",
                "personal_identifiers": "SHA-256"
            }
        }
        
        # CCPA Configuration
        self.compliance_configs[ComplianceFramework.CCPA] = {
            "consumer_rights": {
                "right_to_know": True,
                "right_to_delete": True,
                "right_to_opt_out": True,
                "right_to_non_discrimination": True
            },
            "retention_limits": {
                "personal_information": 365,
                "logs": 90
            }
        }
        
        # PDPA Configuration (Singapore)
        self.compliance_configs[ComplianceFramework.PDPA] = {
            "data_protection": {
                "consent_required": True,
                "purpose_limitation": True,
                "accuracy_obligation": True,
                "protection_obligation": True,
                "retention_limitation": True,
                "transfer_limitation": True
            }
        }
        
        logger.info("Compliance frameworks initialized")
    
    async def configure_region(self, region: Region, frameworks: List[ComplianceFramework]):
        """Configure compliance frameworks for a specific region"""
        config = ComplianceConfig(
            frameworks=frameworks,
            data_residency_requirements=await self._get_data_residency_requirements(region),
            retention_policies=await self._get_retention_policies(frameworks),
            encryption_requirements=await self._get_encryption_requirements(frameworks)
        )
        
        self.region_configs[region] = config
        self.active_frameworks.update(frameworks)
        
        logger.info(f"Region {region.value} configured with frameworks: {[f.value for f in frameworks]}")
        return config
    
    async def _get_data_residency_requirements(self, region: Region) -> Dict[str, str]:
        """Get data residency requirements for region"""
        residency_map = {
            Region.US_EAST: {"country": "US", "jurisdiction": "US Federal"},
            Region.US_WEST: {"country": "US", "jurisdiction": "US Federal"},
            Region.EU_WEST: {"country": "Ireland", "jurisdiction": "EU GDPR"},
            Region.EU_CENTRAL: {"country": "Germany", "jurisdiction": "EU GDPR"},
            Region.ASIA_PACIFIC: {"country": "Singapore", "jurisdiction": "PDPA"},
            Region.ASIA_NORTHEAST: {"country": "Japan", "jurisdiction": "APPI"}
        }
        return residency_map.get(region, {"country": "Unknown", "jurisdiction": "Unknown"})
    
    async def _get_retention_policies(self, frameworks: List[ComplianceFramework]) -> Dict[str, int]:
        """Get data retention policies based on active frameworks"""
        policies = {}
        
        for framework in frameworks:
            config = self.compliance_configs.get(framework, {})
            retention_limits = config.get("retention_limits", {})
            
            for data_type, days in retention_limits.items():
                # Use most restrictive retention policy
                if data_type not in policies or days < policies[data_type]:
                    policies[data_type] = days
                    
        return policies
    
    async def _get_encryption_requirements(self, frameworks: List[ComplianceFramework]) -> Dict[str, str]:
        """Get encryption requirements based on active frameworks"""
        requirements = {
            "data_at_rest": "AES-256",
            "data_in_transit": "TLS-1.3",
            "personal_identifiers": "SHA-256"
        }
        
        for framework in frameworks:
            config = self.compliance_configs.get(framework, {})
            encryption_config = config.get("encryption", {})
            requirements.update(encryption_config)
            
        return requirements
    
    async def validate_data_handling(self, data_type: str, region: Region, 
                                   operation: str) -> Dict[str, Any]:
        """Validate data handling operation against compliance requirements"""
        if region not in self.region_configs:
            return {"valid": False, "reason": f"Region {region.value} not configured"}
            
        config = self.region_configs[region]
        validation_result = {
            "valid": True,
            "region": region.value,
            "data_type": data_type,
            "operation": operation,
            "applied_frameworks": [f.value for f in config.frameworks],
            "requirements": [],
            "violations": []
        }
        
        # Check retention policies
        if data_type in config.retention_policies:
            max_retention = config.retention_policies[data_type]
            validation_result["requirements"].append(
                f"Data must be deleted after {max_retention} days"
            )
            
        # Check encryption requirements
        if operation in ["store", "transfer"]:
            encryption_req = config.encryption_requirements.get("data_at_rest", "AES-256")
            validation_result["requirements"].append(
                f"Data must be encrypted with {encryption_req}"
            )
            
        # GDPR specific checks
        if ComplianceFramework.GDPR in config.frameworks:
            if operation == "collect" and data_type == "personal_data":
                validation_result["requirements"].extend([
                    "Explicit consent required",
                    "Purpose must be specified",
                    "Data minimization must be applied"
                ])
                
        logger.info(f"Data handling validation: {validation_result}")
        return validation_result
    
    async def generate_compliance_report(self, region: Region) -> Dict[str, Any]:
        """Generate compliance report for a region"""
        if region not in self.region_configs:
            return {"error": f"Region {region.value} not configured"}
            
        config = self.region_configs[region]
        
        report = {
            "region": region.value,
            "timestamp": asyncio.get_event_loop().time(),
            "active_frameworks": [f.value for f in config.frameworks],
            "data_residency": config.data_residency_requirements,
            "retention_policies": config.retention_policies,
            "encryption_requirements": config.encryption_requirements,
            "compliance_status": await self._assess_compliance_status(config),
            "recommendations": await self._generate_recommendations(config)
        }
        
        logger.info(f"Generated compliance report for {region.value}")
        return report
    
    async def _assess_compliance_status(self, config: ComplianceConfig) -> Dict[str, str]:
        """Assess current compliance status"""
        status = {}
        
        for framework in config.frameworks:
            # Simulate compliance assessment
            status[framework.value] = "COMPLIANT"
            
        return status
    
    async def _generate_recommendations(self, config: ComplianceConfig) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if ComplianceFramework.GDPR in config.frameworks:
            recommendations.extend([
                "Implement data subject access request handling",
                "Ensure right to erasure is fully automated",
                "Regular privacy impact assessments recommended"
            ])
            
        if ComplianceFramework.CCPA in config.frameworks:
            recommendations.extend([
                "Implement consumer rights request portal",
                "Maintain detailed data processing records"
            ])
            
        return recommendations

class InternationalizationManager:
    """
    Internationalization (i18n) support for global deployment
    """
    
    def __init__(self):
        self.supported_languages = {
            "en": "English",
            "es": "Español", 
            "fr": "Français",
            "de": "Deutsch",
            "ja": "日本語",
            "zh": "中文",
            "ko": "한국어",
            "pt": "Português",
            "it": "Italiano",
            "ru": "Русский"
        }
        self.translations = {}
        self.default_language = "en"
        self._load_translations()
        
    def _load_translations(self):
        """Load translation dictionaries"""
        # Core system messages
        self.translations = {
            "en": {
                "compilation_started": "Compilation started",
                "compilation_completed": "Compilation completed successfully",
                "compilation_failed": "Compilation failed",
                "optimization_applied": "Optimization applied",
                "quantum_enhancement": "Quantum enhancement enabled",
                "photonic_circuit": "Photonic circuit",
                "power_budget": "Power budget",
                "performance_metric": "Performance metric",
                "error_occurred": "An error occurred",
                "validation_failed": "Validation failed",
                "cache_hit": "Cache hit",
                "cache_miss": "Cache miss"
            },
            "es": {
                "compilation_started": "Compilación iniciada",
                "compilation_completed": "Compilación completada exitosamente",
                "compilation_failed": "Compilación falló",
                "optimization_applied": "Optimización aplicada",
                "quantum_enhancement": "Mejora cuántica habilitada",
                "photonic_circuit": "Circuito fotónico",
                "power_budget": "Presupuesto de energía",
                "performance_metric": "Métrica de rendimiento",
                "error_occurred": "Ocurrió un error",
                "validation_failed": "Validación falló",
                "cache_hit": "Acierto de caché",
                "cache_miss": "Fallo de caché"
            },
            "fr": {
                "compilation_started": "Compilation démarrée",
                "compilation_completed": "Compilation terminée avec succès",
                "compilation_failed": "Échec de la compilation",
                "optimization_applied": "Optimisation appliquée",
                "quantum_enhancement": "Amélioration quantique activée",
                "photonic_circuit": "Circuit photonique",
                "power_budget": "Budget énergétique",
                "performance_metric": "Métrique de performance",
                "error_occurred": "Une erreur s'est produite",
                "validation_failed": "Échec de la validation",
                "cache_hit": "Succès du cache",
                "cache_miss": "Échec du cache"
            },
            "de": {
                "compilation_started": "Kompilierung gestartet",
                "compilation_completed": "Kompilierung erfolgreich abgeschlossen",
                "compilation_failed": "Kompilierung fehlgeschlagen",
                "optimization_applied": "Optimierung angewendet",
                "quantum_enhancement": "Quantenverbesserung aktiviert",
                "photonic_circuit": "Photonischer Schaltkreis",
                "power_budget": "Leistungsbudget",
                "performance_metric": "Leistungsmetrik",
                "error_occurred": "Ein Fehler ist aufgetreten",
                "validation_failed": "Validierung fehlgeschlagen",
                "cache_hit": "Cache-Treffer",
                "cache_miss": "Cache-Fehler"
            },
            "ja": {
                "compilation_started": "コンパイル開始",
                "compilation_completed": "コンパイル正常完了",
                "compilation_failed": "コンパイル失敗",
                "optimization_applied": "最適化適用",
                "quantum_enhancement": "量子強化有効",
                "photonic_circuit": "フォトニック回路",
                "power_budget": "電力予算",
                "performance_metric": "性能指標",
                "error_occurred": "エラーが発生しました",
                "validation_failed": "検証失敗",
                "cache_hit": "キャッシュヒット",
                "cache_miss": "キャッシュミス"
            },
            "zh": {
                "compilation_started": "编译开始",
                "compilation_completed": "编译成功完成",
                "compilation_failed": "编译失败",
                "optimization_applied": "优化已应用",
                "quantum_enhancement": "量子增强已启用",
                "photonic_circuit": "光子电路",
                "power_budget": "功率预算",
                "performance_metric": "性能指标",
                "error_occurred": "发生错误",
                "validation_failed": "验证失败",
                "cache_hit": "缓存命中",
                "cache_miss": "缓存未命中"
            }
        }
        
        logger.info(f"Loaded translations for {len(self.translations)} languages")
    
    def get_text(self, key: str, language: str = None) -> str:
        """Get translated text for given key and language"""
        if language is None:
            language = self.default_language
            
        if language not in self.translations:
            language = self.default_language
            
        return self.translations.get(language, {}).get(key, key)
    
    def set_language(self, language: str):
        """Set default language"""
        if language in self.supported_languages:
            self.default_language = language
            logger.info(f"Default language set to {language}")
        else:
            logger.warning(f"Unsupported language: {language}")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.supported_languages.copy()

# Global instances
compliance_manager = GlobalComplianceManager()
i18n_manager = InternationalizationManager()

async def setup_global_deployment():
    """Setup global deployment configuration"""
    # Configure regions with appropriate compliance frameworks
    
    # US regions - CCPA
    await compliance_manager.configure_region(Region.US_EAST, [ComplianceFramework.CCPA, ComplianceFramework.SOC2])
    await compliance_manager.configure_region(Region.US_WEST, [ComplianceFramework.CCPA, ComplianceFramework.SOC2])
    
    # EU regions - GDPR
    await compliance_manager.configure_region(Region.EU_WEST, [ComplianceFramework.GDPR, ComplianceFramework.ISO27001])
    await compliance_manager.configure_region(Region.EU_CENTRAL, [ComplianceFramework.GDPR, ComplianceFramework.ISO27001])
    
    # Asia Pacific - PDPA
    await compliance_manager.configure_region(Region.ASIA_PACIFIC, [ComplianceFramework.PDPA])
    await compliance_manager.configure_region(Region.ASIA_NORTHEAST, [ComplianceFramework.PDPA])
    
    logger.info("Global deployment configuration completed")
    
    return {
        "compliance_manager": compliance_manager,
        "i18n_manager": i18n_manager,
        "configured_regions": list(compliance_manager.region_configs.keys()),
        "supported_languages": list(i18n_manager.supported_languages.keys())
    }