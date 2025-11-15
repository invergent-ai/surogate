# surogate/eval/security/base.py
"""Base classes for red-teaming."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class AttackType(Enum):
    """Types of adversarial attacks."""
    # Single-turn attacks
    PROMPT_INJECTION = "prompt_injection"
    PROMPT_PROBING = "prompt_probing"
    ROLEPLAY = "roleplay"
    GRAY_BOX = "gray_box"
    MATH_PROBLEM = "math_problem"
    MULTILINGUAL = "multilingual"

    # Attack enhancements (encoding/obfuscation)
    BASE64 = "base64"
    LEETSPEAK = "leetspeak"
    ROT13 = "rot13"

    # Advanced single-turn attacks
    SYSTEM_OVERRIDE = "system_override"
    PERMISSION_ESCALATION = "permission_escalation"
    GOAL_REDIRECTION = "goal_redirection"
    LINGUISTIC_CONFUSION = "linguistic_confusion"
    INPUT_BYPASS = "input_bypass"
    CONTEXT_POISONING = "context_poisoning"

    # Multi-turn attacks
    LINEAR_JAILBREAKING = "linear_jailbreaking"
    TREE_JAILBREAKING = "tree_jailbreaking"
    CRESCENDO_JAILBREAKING = "crescendo_jailbreaking"
    SEQUENTIAL_JAILBREAK = "sequential_jailbreak"
    BAD_LIKERT_JUDGE = "bad_likert_judge"


class VulnerabilityType(Enum):
    """Types of vulnerabilities to scan for."""
    # Core Vulnerabilities
    BIAS = "bias"
    TOXICITY = "toxicity"
    PII_LEAKAGE = "pii_leakage"
    MISINFORMATION = "misinformation"
    ILLEGAL_ACTIVITY = "illegal_activity"
    PROMPT_LEAKAGE = "prompt_leakage"

    # Security Vulnerabilities
    BFLA = "bfla"  # Broken Function Level Authorization
    BOLA = "bola"  # Broken Object Level Authorization
    RBAC = "rbac"  # Role-Based Access Control
    DEBUG_ACCESS = "debug_access"
    SHELL_INJECTION = "shell_injection"
    SQL_INJECTION = "sql_injection"
    SSRF = "ssrf"  # Server-Side Request Forgery

    # Content Safety
    CHILD_PROTECTION = "child_protection"
    ETHICS = "ethics"
    FAIRNESS = "fairness"
    GRAPHIC_CONTENT = "graphic_content"
    PERSONAL_SAFETY = "personal_safety"

    # Business Vulnerabilities
    INTELLECTUAL_PROPERTY = "intellectual_property"
    COMPETITION = "competition"

    # Agentic Vulnerabilities
    GOAL_THEFT = "goal_theft"
    RECURSIVE_HIJACKING = "recursive_hijacking"
    ROBUSTNESS = "robustness"
    EXCESSIVE_AGENCY = "excessive_agency"

    # Custom
    CUSTOM = "custom"


class SeverityLevel(Enum):
    """Severity levels for vulnerabilities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AttackResult:
    """Result of a single attack."""
    attack_type: str
    vulnerability_type: str
    input: str
    output: str
    score: float
    passed: bool
    severity: Optional[SeverityLevel] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VulnerabilityScanResult:
    """Result of vulnerability scanning."""
    vulnerability_type: str
    total_attacks: int
    successful_attacks: int
    failed_attacks: int
    success_rate: float
    severity: SeverityLevel
    attack_results: List[AttackResult]
    metadata: Optional[Dict[str, Any]] = None