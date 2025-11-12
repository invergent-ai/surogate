from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class TestCaseType(Enum):
    """Test case types."""
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"


@dataclass
class TestCase:
    """Single-turn test case."""

    input: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate test case after initialization."""
        if not self.input:
            raise ValueError("Test case input cannot be empty")

    @property
    def type(self) -> TestCaseType:
        """Get test case type."""
        return TestCaseType.SINGLE_TURN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'input': self.input,
            'expected_output': self.expected_output,
            'metadata': self.metadata,
            'type': self.type.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        """Create from dictionary."""
        return cls(
            input=data['input'],
            expected_output=data.get('expected_output'),
            metadata=data.get('metadata', {})
        )


@dataclass
class Turn:
    """Single turn in a multi-turn conversation."""

    role: str  # 'user' or 'assistant'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate turn."""
        if self.role not in ['user', 'assistant']:
            raise ValueError(f"Invalid role: {self.role}. Must be 'user' or 'assistant'")
        if not self.content:
            raise ValueError("Turn content cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'role': self.role,
            'content': self.content,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Turn':
        """Create from dictionary."""
        return cls(
            role=data['role'],
            content=data['content'],
            metadata=data.get('metadata', {})
        )


@dataclass
class MultiTurnTestCase:
    """Multi-turn conversation test case."""

    turns: List[Turn]
    expected_final_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate multi-turn test case."""
        if not self.turns:
            raise ValueError("Multi-turn test case must have at least one turn")

        # Validate turn order (should start with user)
        if self.turns[0].role != 'user':
            raise ValueError("First turn must be from user")

    @property
    def type(self) -> TestCaseType:
        """Get test case type."""
        return TestCaseType.MULTI_TURN

    @property
    def num_turns(self) -> int:
        """Get number of turns."""
        return len(self.turns)

    def get_context(self) -> List[Dict[str, str]]:
        """Get conversation context in standard format."""
        return [
            {'role': turn.role, 'content': turn.content}
            for turn in self.turns
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'turns': [turn.to_dict() for turn in self.turns],
            'expected_final_output': self.expected_final_output,
            'metadata': self.metadata,
            'type': self.type.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiTurnTestCase':
        """Create from dictionary."""
        turns = [Turn.from_dict(turn_data) for turn_data in data['turns']]
        return cls(
            turns=turns,
            expected_final_output=data.get('expected_final_output'),
            metadata=data.get('metadata', {})
        )