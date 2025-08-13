"""
International Localization (i18n) Support for Photonic MLIR
Implements multi-language support for global deployment.
"""

from typing import Dict, Any, Optional
from enum import Enum
import json
import os

class SupportedLanguage(Enum):
    """Supported languages for international deployment."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"

class I18nManager:
    """International localization manager for photonic MLIR."""
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        self.default_language = default_language
        self.current_language = default_language
        self.translations = self._load_translations()
        
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation dictionaries for all supported languages."""
        return {
            "en": {
                "compiler_error": "Compilation error occurred",
                "invalid_wavelength": "Invalid wavelength specification", 
                "power_budget_exceeded": "Power budget exceeded",
                "optimization_complete": "Optimization complete",
                "simulation_started": "Simulation started",
                "benchmark_running": "Benchmark in progress",
                "cache_miss": "Cache miss - computing result",
                "security_violation": "Security violation detected",
                "welcome": "Welcome to Photonic MLIR",
                "processing": "Processing photonic compilation..."
            },
            "es": {
                "compiler_error": "Error de compilación ocurrido",
                "invalid_wavelength": "Especificación de longitud de onda inválida",
                "power_budget_exceeded": "Presupuesto de energía excedido", 
                "optimization_complete": "Optimización completa",
                "simulation_started": "Simulación iniciada",
                "benchmark_running": "Benchmark en progreso",
                "cache_miss": "Fallo de caché - calculando resultado",
                "security_violation": "Violación de seguridad detectada",
                "welcome": "Bienvenido a Photonic MLIR",
                "processing": "Procesando compilación fotónica..."
            },
            "fr": {
                "compiler_error": "Erreur de compilation survenue",
                "invalid_wavelength": "Spécification de longueur d'onde invalide",
                "power_budget_exceeded": "Budget de puissance dépassé",
                "optimization_complete": "Optimisation terminée", 
                "simulation_started": "Simulation démarrée",
                "benchmark_running": "Benchmark en cours",
                "cache_miss": "Échec du cache - calcul du résultat",
                "security_violation": "Violation de sécurité détectée",
                "welcome": "Bienvenue dans Photonic MLIR",
                "processing": "Traitement de la compilation photonique..."
            },
            "de": {
                "compiler_error": "Kompilierungsfehler aufgetreten",
                "invalid_wavelength": "Ungültige Wellenlängenspezifikation",
                "power_budget_exceeded": "Leistungsbudget überschritten",
                "optimization_complete": "Optimierung abgeschlossen",
                "simulation_started": "Simulation gestartet", 
                "benchmark_running": "Benchmark läuft",
                "cache_miss": "Cache-Fehler - Ergebnis wird berechnet",
                "security_violation": "Sicherheitsverletzung erkannt",
                "welcome": "Willkommen bei Photonic MLIR",
                "processing": "Verarbeitung der photonischen Kompilierung..."
            },
            "ja": {
                "compiler_error": "コンパイルエラーが発生しました",
                "invalid_wavelength": "無効な波長の指定",
                "power_budget_exceeded": "消費電力予算を超過",
                "optimization_complete": "最適化完了",
                "simulation_started": "シミュレーション開始",
                "benchmark_running": "ベンチマーク実行中",
                "cache_miss": "キャッシュミス - 結果を計算中",
                "security_violation": "セキュリティ違反を検出",
                "welcome": "Photonic MLIRへようこそ",
                "processing": "フォトニックコンパイルを処理中..."
            },
            "zh": {
                "compiler_error": "发生编译错误",
                "invalid_wavelength": "无效的波长规格",
                "power_budget_exceeded": "功率预算超标",
                "optimization_complete": "优化完成",
                "simulation_started": "仿真开始", 
                "benchmark_running": "基准测试进行中",
                "cache_miss": "缓存未命中 - 计算结果中",
                "security_violation": "检测到安全违规",
                "welcome": "欢迎使用 Photonic MLIR",
                "processing": "正在处理光子编译..."
            }
        }
    
    def set_language(self, language: SupportedLanguage) -> None:
        """Set the current language for localization."""
        self.current_language = language
        
    def get_text(self, key: str, language: Optional[SupportedLanguage] = None) -> str:
        """Get localized text for the specified key."""
        lang = language or self.current_language
        lang_code = lang.value
        
        if lang_code in self.translations and key in self.translations[lang_code]:
            return self.translations[lang_code][key]
        
        # Fallback to English
        if key in self.translations[self.default_language.value]:
            return self.translations[self.default_language.value][key]
        
        # Ultimate fallback
        return key
    
    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return [lang.value for lang in SupportedLanguage]
    
    def format_message(self, key: str, **kwargs) -> str:
        """Get localized text with variable formatting."""
        template = self.get_text(key)
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError):
            return template

# Global i18n manager instance
_i18n_manager = I18nManager()

def get_i18n_manager() -> I18nManager:
    """Get the global i18n manager instance."""
    return _i18n_manager

def localize(key: str, language: Optional[SupportedLanguage] = None) -> str:
    """Convenience function for getting localized text."""
    return _i18n_manager.get_text(key, language)

def set_global_language(language: SupportedLanguage) -> None:
    """Set the global language for all localization."""
    _i18n_manager.set_language(language)