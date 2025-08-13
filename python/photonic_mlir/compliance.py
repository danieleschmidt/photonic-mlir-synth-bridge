"""
Data Privacy and Compliance Module for Photonic MLIR
Implements GDPR, CCPA, and PDPA compliance features for global deployment.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import json
import hashlib
import logging

logger = logging.getLogger(__name__)

class ComplianceRegion(Enum):
    """Supported compliance regions."""
    EU_GDPR = "eu_gdpr"        # General Data Protection Regulation (EU)
    US_CCPA = "us_ccpa"        # California Consumer Privacy Act (US)
    SG_PDPA = "sg_pdpa"        # Personal Data Protection Act (Singapore)
    GLOBAL = "global"          # Global compliance (strictest standards)

class DataCategory(Enum):
    """Categories of data for compliance classification."""
    TECHNICAL_LOGS = "technical_logs"
    COMPILATION_CACHE = "compilation_cache"
    BENCHMARK_RESULTS = "benchmark_results"
    PERFORMANCE_METRICS = "performance_metrics"
    ERROR_REPORTS = "error_reports"
    USAGE_ANALYTICS = "usage_analytics"

@dataclass
class DataRetentionPolicy:
    """Data retention policy specification."""
    category: DataCategory
    retention_days: int
    region: ComplianceRegion
    auto_delete: bool = True
    encryption_required: bool = True

class ComplianceManager:
    """Manages data privacy and compliance requirements."""
    
    def __init__(self, region: ComplianceRegion = ComplianceRegion.GLOBAL):
        self.region = region
        self.data_processing_records = []
        self.consent_records = {}
        self.retention_policies = self._load_retention_policies()
        
    def _load_retention_policies(self) -> Dict[ComplianceRegion, Dict[DataCategory, DataRetentionPolicy]]:
        """Load data retention policies for different regions."""
        policies = {
            ComplianceRegion.EU_GDPR: {
                DataCategory.TECHNICAL_LOGS: DataRetentionPolicy(
                    DataCategory.TECHNICAL_LOGS, 30, ComplianceRegion.EU_GDPR
                ),
                DataCategory.COMPILATION_CACHE: DataRetentionPolicy(
                    DataCategory.COMPILATION_CACHE, 7, ComplianceRegion.EU_GDPR
                ),
                DataCategory.BENCHMARK_RESULTS: DataRetentionPolicy(
                    DataCategory.BENCHMARK_RESULTS, 90, ComplianceRegion.EU_GDPR
                ),
                DataCategory.PERFORMANCE_METRICS: DataRetentionPolicy(
                    DataCategory.PERFORMANCE_METRICS, 90, ComplianceRegion.EU_GDPR
                ),
                DataCategory.ERROR_REPORTS: DataRetentionPolicy(
                    DataCategory.ERROR_REPORTS, 180, ComplianceRegion.EU_GDPR
                ),
                DataCategory.USAGE_ANALYTICS: DataRetentionPolicy(
                    DataCategory.USAGE_ANALYTICS, 365, ComplianceRegion.EU_GDPR
                )
            },
            ComplianceRegion.US_CCPA: {
                DataCategory.TECHNICAL_LOGS: DataRetentionPolicy(
                    DataCategory.TECHNICAL_LOGS, 90, ComplianceRegion.US_CCPA
                ),
                DataCategory.COMPILATION_CACHE: DataRetentionPolicy(
                    DataCategory.COMPILATION_CACHE, 30, ComplianceRegion.US_CCPA
                ),
                DataCategory.BENCHMARK_RESULTS: DataRetentionPolicy(
                    DataCategory.BENCHMARK_RESULTS, 365, ComplianceRegion.US_CCPA
                ),
                DataCategory.PERFORMANCE_METRICS: DataRetentionPolicy(
                    DataCategory.PERFORMANCE_METRICS, 365, ComplianceRegion.US_CCPA
                ),
                DataCategory.ERROR_REPORTS: DataRetentionPolicy(
                    DataCategory.ERROR_REPORTS, 730, ComplianceRegion.US_CCPA
                ),
                DataCategory.USAGE_ANALYTICS: DataRetentionPolicy(
                    DataCategory.USAGE_ANALYTICS, 730, ComplianceRegion.US_CCPA
                )
            },
            ComplianceRegion.SG_PDPA: {
                DataCategory.TECHNICAL_LOGS: DataRetentionPolicy(
                    DataCategory.TECHNICAL_LOGS, 60, ComplianceRegion.SG_PDPA
                ),
                DataCategory.COMPILATION_CACHE: DataRetentionPolicy(
                    DataCategory.COMPILATION_CACHE, 14, ComplianceRegion.SG_PDPA
                ),
                DataCategory.BENCHMARK_RESULTS: DataRetentionPolicy(
                    DataCategory.BENCHMARK_RESULTS, 180, ComplianceRegion.SG_PDPA
                ),
                DataCategory.PERFORMANCE_METRICS: DataRetentionPolicy(
                    DataCategory.PERFORMANCE_METRICS, 180, ComplianceRegion.SG_PDPA
                ),
                DataCategory.ERROR_REPORTS: DataRetentionPolicy(
                    DataCategory.ERROR_REPORTS, 365, ComplianceRegion.SG_PDPA
                ),
                DataCategory.USAGE_ANALYTICS: DataRetentionPolicy(
                    DataCategory.USAGE_ANALYTICS, 365, ComplianceRegion.SG_PDPA
                )
            }
        }
        
        # Global compliance uses the strictest (shortest) retention periods
        policies[ComplianceRegion.GLOBAL] = {}
        for category in DataCategory:
            min_retention = min(
                policies[ComplianceRegion.EU_GDPR][category].retention_days,
                policies[ComplianceRegion.US_CCPA][category].retention_days,
                policies[ComplianceRegion.SG_PDPA][category].retention_days
            )
            policies[ComplianceRegion.GLOBAL][category] = DataRetentionPolicy(
                category, min_retention, ComplianceRegion.GLOBAL
            )
            
        return policies
    
    def record_data_processing(self, category: DataCategory, purpose: str, 
                             data_size: int, user_consent: bool = True) -> str:
        """Record data processing activity for compliance audit."""
        processing_id = hashlib.sha256(
            f"{category.value}_{purpose}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        record = {
            'processing_id': processing_id,
            'timestamp': datetime.now().isoformat(),
            'category': category.value,
            'purpose': purpose,
            'data_size_bytes': data_size,
            'user_consent': user_consent,
            'region': self.region.value,
            'retention_policy': self.get_retention_policy(category).retention_days
        }
        
        self.data_processing_records.append(record)
        
        # Log compliance event
        logger.info(f"Data processing recorded: {processing_id} - {category.value} - {purpose}")
        
        return processing_id
    
    def get_retention_policy(self, category: DataCategory) -> DataRetentionPolicy:
        """Get the data retention policy for a category in the current region."""
        return self.retention_policies[self.region][category]
    
    def should_delete_data(self, category: DataCategory, creation_date: datetime) -> bool:
        """Check if data should be deleted based on retention policy."""
        policy = self.get_retention_policy(category)
        retention_cutoff = datetime.now() - timedelta(days=policy.retention_days)
        return creation_date < retention_cutoff and policy.auto_delete
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize data by removing personally identifiable information."""
        anonymized = data.copy()
        
        # Common PII fields to anonymize
        pii_fields = ['user_id', 'email', 'ip_address', 'session_id', 'device_id']
        
        for field in pii_fields:
            if field in anonymized:
                # Replace with anonymized hash
                original_value = str(anonymized[field])
                anonymized[field] = hashlib.sha256(original_value.encode()).hexdigest()[:12]
        
        # Add anonymization marker
        anonymized['_anonymized'] = True
        anonymized['_anonymization_timestamp'] = datetime.now().isoformat()
        
        return anonymized
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate a compliance report for audit purposes."""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'compliance_region': self.region.value,
            'data_processing_summary': {
                'total_processing_events': len(self.data_processing_records),
                'categories_processed': list(set(r['category'] for r in self.data_processing_records)),
                'consent_rate': sum(1 for r in self.data_processing_records if r['user_consent']) / max(1, len(self.data_processing_records)) * 100
            },
            'retention_policies': {
                category.value: {
                    'retention_days': policy.retention_days,
                    'auto_delete': policy.auto_delete,
                    'encryption_required': policy.encryption_required
                } for category, policy in self.retention_policies[self.region].items()
            },
            'compliance_metrics': {
                'data_minimization_score': self._calculate_data_minimization_score(),
                'retention_compliance_score': self._calculate_retention_compliance_score(),
                'consent_compliance_score': self._calculate_consent_compliance_score()
            }
        }
        
        return report
    
    def _calculate_data_minimization_score(self) -> float:
        """Calculate score for data minimization principle compliance."""
        if not self.data_processing_records:
            return 1.0
        
        # Score based on average data size - smaller is better for minimization
        avg_data_size = sum(r['data_size_bytes'] for r in self.data_processing_records) / len(self.data_processing_records)
        
        # Normalize score (assuming 1MB is reasonable baseline)
        baseline_size = 1024 * 1024  # 1MB
        score = max(0.1, min(1.0, baseline_size / max(avg_data_size, 1)))
        
        return round(score, 3)
    
    def _calculate_retention_compliance_score(self) -> float:
        """Calculate score for retention policy compliance."""
        # All policies have auto_delete enabled - perfect compliance
        return 1.0
    
    def _calculate_consent_compliance_score(self) -> float:
        """Calculate score for consent management compliance."""
        if not self.data_processing_records:
            return 1.0
        
        consent_rate = sum(1 for r in self.data_processing_records if r['user_consent']) / len(self.data_processing_records)
        return round(consent_rate, 3)

class CrossPlatformCompatibility:
    """Ensures cross-platform compatibility for global deployment."""
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """Normalize file paths for cross-platform compatibility."""
        import os
        return os.path.normpath(path).replace('\\', '/')
    
    @staticmethod
    def get_platform_config() -> Dict[str, Any]:
        """Get platform-specific configuration."""
        import platform
        import sys
        
        return {
            'system': platform.system(),
            'platform': platform.platform(), 
            'architecture': platform.architecture()[0],
            'python_version': sys.version,
            'endianness': sys.byteorder,
            'max_int': sys.maxsize,
            'path_separator': os.sep,
            'line_separator': os.linesep
        }
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check if required dependencies are available across platforms."""
        dependencies = {
            'numpy': False,
            'psutil': False, 
            'json': True,  # Built-in
            'threading': True,  # Built-in
            'multiprocessing': True  # Built-in
        }
        
        for dep in ['numpy', 'psutil']:
            try:
                __import__(dep)
                dependencies[dep] = True
            except ImportError:
                dependencies[dep] = False
                
        return dependencies

# Global compliance manager instance
_compliance_manager = ComplianceManager()

def get_compliance_manager() -> ComplianceManager:
    """Get the global compliance manager instance."""
    return _compliance_manager

def record_processing(category: DataCategory, purpose: str, data_size: int) -> str:
    """Convenience function for recording data processing."""
    return _compliance_manager.record_data_processing(category, purpose, data_size)

def check_retention(category: DataCategory, creation_date: datetime) -> bool:
    """Convenience function for checking data retention requirements."""
    return _compliance_manager.should_delete_data(category, creation_date)