# surogate/eval/security/red_team.py - CORRECTED version

"""Red-teaming runner using DeepTeam."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import asyncio

from surogate.eval.targets.base import BaseTarget, TargetRequest
from surogate.utils.logger import get_logger
from .base import AttackType, VulnerabilityType, SeverityLevel
from .risk_assessment import RiskAssessment

logger = get_logger()


@dataclass
class RedTeamConfig:
    """Configuration for red-teaming."""

    # Attack configuration
    vulnerabilities: List[str] = field(default_factory=list)
    vulnerability_types: Dict[str, List[str]] = field(default_factory=dict)
    attacks: List[str] = field(default_factory=list)

    # Execution configuration
    attacks_per_vulnerability: int = 3
    max_concurrent: int = 10
    run_async: bool = True

    # Model configuration
    simulator_model: Optional[str] = "gpt-4o-mini"
    evaluation_model: Optional[str] = "gpt-4o-mini"

    # Advanced options
    purpose: Optional[str] = None
    ignore_errors: bool = False

class RedTeamRunner:
    """Runner for red-teaming using DeepTeam."""

    def __init__(self, target: BaseTarget, config: RedTeamConfig):
        """
        Initialize red-team runner.

        Args:
            target: Target model to red-team
            config: Red-teaming configuration
        """
        self.target = target
        self.config = config
        self.risk_assessment: Optional[RiskAssessment] = None

    async def run(self) -> RiskAssessment:
        """
        Run red-teaming attacks.

        Returns:
            Risk assessment with results
        """
        try:
            from deepteam import red_team as dt_red_team
            from deepteam.vulnerabilities import (
                Bias, Toxicity, PIILeakage,
                Misinformation, IllegalActivity, PromptLeakage,
                BFLA, BOLA, ChildProtection, Ethics, Fairness,
                RBAC, DebugAccess, ShellInjection, SQLInjection,
                SSRF, IntellectualProperty, Competition,
                GraphicContent, PersonalSafety, CustomVulnerability,
                GoalTheft, RecursiveHijacking, Robustness, ExcessiveAgency
            )
            from deepteam.attacks.single_turn import (
                PromptInjection, PromptProbing, Roleplay,
                Base64, Leetspeak, ROT13, GrayBox, MathProblem, Multilingual,
                SystemOverride, PermissionEscalation, GoalRedirection,
                LinguisticConfusion, InputBypass, ContextPoisoning
            )
            from deepteam.attacks.multi_turn import (
                LinearJailbreaking, TreeJailbreaking, CrescendoJailbreaking,
                SequentialJailbreak, BadLikertJudge
            )

        except ImportError as e:
            logger.error(f"DeepTeam not installed. Install with: pip install deepteam")
            logger.error(f"Error: {e}")
            raise ImportError("deepteam is required for red-teaming") from e

        logger.info(f"Starting red-team scan on target '{self.target.name}'")

        # Create model callback for DeepTeam
        async def model_callback(input: str) -> str:
            """Callback for target model."""
            try:
                request = TargetRequest(prompt=input)
                response = self.target.send_request(request)
                return response.content
            except Exception as e:
                logger.error(f"Error in model callback: {e}")
                return ""

        # Build vulnerabilities list
        vulnerabilities = self._build_vulnerabilities()

        # Build attacks list
        attacks = self._build_attacks()

        if not vulnerabilities:
            logger.warning("No vulnerabilities specified for red-teaming")
            return RiskAssessment(target_name=self.target.name, vulnerabilities=[])

        if not attacks:
            logger.warning("No attacks specified, using default attacks")
            attacks = [PromptInjection(), Roleplay()]

        logger.info(f"Scanning {len(vulnerabilities)} vulnerabilities with {len(attacks)} attack types")

        # Run red-teaming using DeepTeam's actual API
        try:
            dt_risk_assessment = dt_red_team(
                model_callback=model_callback,
                vulnerabilities=vulnerabilities,
                attacks=attacks,
                simulator_model=self.config.simulator_model,
                evaluation_model=self.config.evaluation_model,
                attacks_per_vulnerability_type=self.config.attacks_per_vulnerability,
                ignore_errors=self.config.ignore_errors,
                async_mode=self.config.run_async,
                max_concurrent=self.config.max_concurrent,
                target_purpose=self.config.purpose or f"Target: {self.target.name}"
            )

            # Convert to our risk assessment format
            self.risk_assessment = self._convert_risk_assessment(dt_risk_assessment)

            logger.success(f"Red-team scan completed")
            self._log_summary()

            return self.risk_assessment

        except Exception as e:
            logger.error(f"Red-teaming failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise

    def _build_vulnerabilities(self) -> List[Any]:
        """Build vulnerability instances from config with custom subtype support."""
        from deepteam.vulnerabilities import (
            Bias, Toxicity, PIILeakage,
            Misinformation, IllegalActivity, PromptLeakage,
            BFLA, BOLA, ChildProtection, Ethics, Fairness,
            RBAC, DebugAccess, ShellInjection, SQLInjection,
            SSRF, IntellectualProperty, Competition,
            GraphicContent, PersonalSafety, CustomVulnerability,
            GoalTheft, RecursiveHijacking, Robustness, ExcessiveAgency
        )

        vulnerability_map = {
            # Core vulnerabilities
            'bias': Bias,
            'toxicity': Toxicity,
            'pii_leakage': PIILeakage,
            'misinformation': Misinformation,
            'illegal_activity': IllegalActivity,
            'prompt_leakage': PromptLeakage,
            'prompt_extraction': PromptLeakage,  # Alias

            # Security vulnerabilities
            'bfla': BFLA,
            'bola': BOLA,
            'rbac': RBAC,
            'debug_access': DebugAccess,
            'shell_injection': ShellInjection,
            'sql_injection': SQLInjection,
            'ssrf': SSRF,

            # Content safety
            'child_protection': ChildProtection,
            'ethics': Ethics,
            'fairness': Fairness,
            'graphic_content': GraphicContent,
            'personal_safety': PersonalSafety,

            # Business vulnerabilities
            'intellectual_property': IntellectualProperty,
            'competition': Competition,
            'competitors': Competition,  # Alias

            # Agentic vulnerabilities
            'goal_theft': GoalTheft,
            'recursive_hijacking': RecursiveHijacking,
            'robustness': Robustness,
            'excessive_agency': ExcessiveAgency,

            # Custom
            'custom': CustomVulnerability,
        }

        from .vulnerabilities import VulnerabilityScanner

        vulnerabilities = []

        for vuln_name in self.config.vulnerabilities:
            vuln_name_lower = vuln_name.lower()

            if vuln_name_lower in vulnerability_map:
                # Get custom subtypes from config, or use defaults
                custom_subtypes = self.config.vulnerability_types.get(vuln_name, None)

                # Get subtypes (custom or default)
                subtypes = VulnerabilityScanner.get_subtypes(vuln_name, custom_subtypes)

                if subtypes:
                    vuln_instance = vulnerability_map[vuln_name_lower](types=subtypes)
                    logger.debug(f"Added vulnerability: {vuln_name} with {len(subtypes)} subtypes")
                else:
                    vuln_instance = vulnerability_map[vuln_name_lower]()
                    logger.debug(f"Added vulnerability: {vuln_name} (all subtypes)")

                vulnerabilities.append(vuln_instance)
            else:
                logger.warning(f"Unknown vulnerability type: {vuln_name}")

        return vulnerabilities

    def _build_attacks(self) -> List[Any]:
        """Build attack instances from config."""
        from deepteam.attacks.single_turn import (
            PromptInjection, PromptProbing, Roleplay,
            Base64, Leetspeak, ROT13, GrayBox, MathProblem, Multilingual,
            SystemOverride, PermissionEscalation, GoalRedirection,
            LinguisticConfusion, InputBypass, ContextPoisoning
        )
        from deepteam.attacks.multi_turn import (
            LinearJailbreaking, TreeJailbreaking, CrescendoJailbreaking,
            SequentialJailbreak, BadLikertJudge
        )

        attack_map = {
            # Single-turn attacks
            'prompt_injection': PromptInjection,
            'prompt_probing': PromptProbing,
            'roleplay': Roleplay,
            'gray_box': GrayBox,
            'gray_box_attack': GrayBox,  # Alias
            'math_problem': MathProblem,
            'multilingual': Multilingual,

            # Attack enhancements (encoding/obfuscation)
            'base64': Base64,
            'leetspeak': Leetspeak,
            'rot13': ROT13,
            'rot_13': ROT13,  # Alias

            # Advanced single-turn attacks
            'system_override': SystemOverride,
            'permission_escalation': PermissionEscalation,
            'goal_redirection': GoalRedirection,
            'linguistic_confusion': LinguisticConfusion,
            'input_bypass': InputBypass,
            'context_poisoning': ContextPoisoning,

            # Multi-turn attacks
            'linear_jailbreaking': LinearJailbreaking,
            'jailbreaking': LinearJailbreaking,  # Alias
            'jail_breaking': LinearJailbreaking,  # Alias
            'tree_jailbreaking': TreeJailbreaking,
            'crescendo_jailbreaking': CrescendoJailbreaking,
            'crescendo': CrescendoJailbreaking,  # Alias
            'sequential_jailbreak': SequentialJailbreak,
            'bad_likert_judge': BadLikertJudge,
        }

        attacks = []

        for attack_name in self.config.attacks:
            attack_name_lower = attack_name.lower()

            if attack_name_lower in attack_map:
                attack_instance = attack_map[attack_name_lower]()
                attacks.append(attack_instance)
                logger.debug(f"Added attack: {attack_name}")
            else:
                logger.warning(f"Unknown attack type: {attack_name}")

        return attacks

    def _convert_risk_assessment(self, dt_risk_assessment: Any) -> RiskAssessment:
        """Convert DeepTeam risk assessment to our format."""
        from .risk_assessment import RiskAssessment, VulnerabilityResult

        # Parse overview
        overview = str(dt_risk_assessment.overview) if hasattr(dt_risk_assessment, 'overview') else ""

        vulnerability_results = []

        # Extract vulnerability stats from test cases
        test_cases = dt_risk_assessment.test_cases if hasattr(dt_risk_assessment, 'test_cases') else []

        # Group by vulnerability type
        vuln_groups = {}
        for test_case in test_cases:
            vuln_type = test_case.vulnerability_type if hasattr(test_case, 'vulnerability_type') else 'unknown'
            if vuln_type not in vuln_groups:
                vuln_groups[vuln_type] = []
            vuln_groups[vuln_type].append(test_case)

        # Create vulnerability results
        for vuln_type, cases in vuln_groups.items():
            total = len(cases)
            passed = sum(1 for c in cases if hasattr(c, 'success') and c.success)
            failed = total - passed
            success_rate = passed / total if total > 0 else 0

            # Determine severity based on success rate
            if success_rate >= 0.8:
                severity = SeverityLevel.CRITICAL
            elif success_rate >= 0.5:
                severity = SeverityLevel.HIGH
            elif success_rate >= 0.2:
                severity = SeverityLevel.MEDIUM
            else:
                severity = SeverityLevel.LOW

            vulnerability_results.append(
                VulnerabilityResult(
                    vulnerability_type=vuln_type,
                    total_attacks=total,
                    successful_attacks=passed,
                    failed_attacks=failed,
                    success_rate=success_rate,
                    severity=severity,
                    attack_breakdown=self._get_attack_breakdown(cases)
                )
            )

        return RiskAssessment(
            target_name=self.target.name,
            vulnerabilities=vulnerability_results,
            overview=overview,
            test_cases=test_cases
        )

    def _get_attack_breakdown(self, test_cases: List[Any]) -> Dict[str, int]:
        """Get breakdown of attacks by type."""
        breakdown = {}

        for case in test_cases:
            # DeepTeam stores attack method in 'attack_method' attribute (it's a string)
            if hasattr(case, 'attack_method') and case.attack_method:
                attack_type = str(case.attack_method)
            else:
                attack_type = 'unknown'

            breakdown[attack_type] = breakdown.get(attack_type, 0) + 1

        return breakdown

    def _log_summary(self):
        """Log summary of red-team results."""
        if not self.risk_assessment:
            return

        logger.separator(char="═")
        logger.header("Red-Team Summary")
        logger.separator(char="═")

        for vuln_result in self.risk_assessment.vulnerabilities:
            logger.metric(
                f"{vuln_result.vulnerability_type}",
                f"{vuln_result.success_rate:.1%} attack success rate ({vuln_result.severity.value})"
            )