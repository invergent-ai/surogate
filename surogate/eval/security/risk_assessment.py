# surogate/eval/security/risk_assessment.py
"""Risk assessment for red-teaming results."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from .base import SeverityLevel
from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class VulnerabilityResult:
    """Result for a single vulnerability type."""
    vulnerability_type: str
    total_attacks: int
    successful_attacks: int
    failed_attacks: int
    success_rate: float
    severity: SeverityLevel
    attack_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """Risk assessment from red-teaming."""
    target_name: str
    vulnerabilities: List[VulnerabilityResult]
    overview: Optional[str] = None
    test_cases: Optional[List[Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'target_name': self.target_name,
            'timestamp': self.timestamp,
            'vulnerabilities': [
                {
                    'vulnerability_type': v.vulnerability_type,
                    'total_attacks': v.total_attacks,
                    'successful_attacks': v.successful_attacks,
                    'failed_attacks': v.failed_attacks,
                    'success_rate': v.success_rate,
                    'severity': v.severity.value,
                    'attack_breakdown': v.attack_breakdown
                }
                for v in self.vulnerabilities
            ],
            'overview': self.overview
        }

    def save(self, path: str):
        """Save risk assessment to file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Risk assessment saved to: {output_path}")

    def get_critical_vulnerabilities(self) -> List[VulnerabilityResult]:
        """Get critical severity vulnerabilities."""
        return [
            v for v in self.vulnerabilities
            if v.severity == SeverityLevel.CRITICAL
        ]

    def get_high_risk_vulnerabilities(self) -> List[VulnerabilityResult]:
        """Get high and critical severity vulnerabilities."""
        return [
            v for v in self.vulnerabilities
            if v.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        ]