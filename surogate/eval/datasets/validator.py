from typing import Dict, Any, List, Tuple, Optional
import polars as pl
from surogate.utils.logger import get_logger

logger = get_logger()


class DatasetValidator:
    """Validate dataset schemas and content."""

    # Standard schema for single-turn test cases
    SINGLE_TURN_SCHEMA = {
        'input': pl.Utf8,
        'expected_output': pl.Utf8,  # Optional
    }

    # Standard schema for multi-turn test cases
    MULTI_TURN_SCHEMA = {
        'turns': pl.List(pl.Struct),  # List of turn objects
        'expected_final_output': pl.Utf8,  # Optional
    }

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize dataset validator.

        Args:
            schema: Custom schema definition (optional)
        """
        self.schema = schema or self.SINGLE_TURN_SCHEMA

    def validate(self, df: pl.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate dataset DataFrame.

        Args:
            df: Polars DataFrame to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check if DataFrame is empty
        if df.is_empty():
            errors.append("Dataset is empty")
            return False, errors

        # Detect dataset type
        is_multi_turn = 'turns' in df.columns

        if is_multi_turn:
            return self._validate_multi_turn(df)
        else:
            return self._validate_single_turn(df)

    def _validate_single_turn(self, df: pl.DataFrame) -> Tuple[bool, List[str]]:
        """Validate single-turn dataset."""
        errors = []

        # Check required columns
        if 'input' not in df.columns:
            errors.append("Missing required column: 'input'")
            return False, errors

        # Check for empty inputs
        null_inputs = df.filter(pl.col('input').is_null()).height
        if null_inputs > 0:
            errors.append(f"Found {null_inputs} rows with null input")

        # FIX: Use str.len_chars() for Polars 0.19+
        try:
            empty_inputs = df.filter(pl.col('input').str.len_chars() == 0).height
        except AttributeError:
            # Fallback for older Polars versions
            empty_inputs = df.filter(pl.col('input').str.lengths() == 0).height

        if empty_inputs > 0:
            errors.append(f"Found {empty_inputs} rows with empty input")

        # Check data types
        if df['input'].dtype != pl.Utf8:
            errors.append(f"Column 'input' must be string type, got {df['input'].dtype}")

        if 'expected_output' in df.columns and df['expected_output'].dtype != pl.Utf8:
            errors.append(f"Column 'expected_output' must be string type, got {df['expected_output'].dtype}")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info(f"✓ Single-turn dataset validation passed ({len(df)} rows)")
        else:
            logger.warning(f"✗ Single-turn dataset validation failed with {len(errors)} errors")

        return is_valid, errors

    def _validate_multi_turn(self, df: pl.DataFrame) -> Tuple[bool, List[str]]:
        """Validate multi-turn dataset."""
        errors = []

        # Check required columns
        if 'turns' not in df.columns:
            errors.append("Missing required column: 'turns'")
            return False, errors

        # Check for empty turns
        null_turns = df.filter(pl.col('turns').is_null()).height
        if null_turns > 0:
            errors.append(f"Found {null_turns} rows with null turns")

        # Validate turn structure
        for idx, row in enumerate(df.iter_rows(named=True)):
            turns = row.get('turns')
            if not turns:
                errors.append(f"Row {idx}: Empty turns list")
                continue

            # Check first turn is from user
            if turns[0].get('role') != 'user':
                errors.append(f"Row {idx}: First turn must be from user")

            # Check all turns have required fields
            for turn_idx, turn in enumerate(turns):
                if 'role' not in turn or 'content' not in turn:
                    errors.append(f"Row {idx}, Turn {turn_idx}: Missing 'role' or 'content'")

                if turn.get('role') not in ['user', 'assistant']:
                    errors.append(f"Row {idx}, Turn {turn_idx}: Invalid role '{turn.get('role')}'")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info(f"✓ Multi-turn dataset validation passed ({len(df)} rows)")
        else:
            logger.warning(f"✗ Multi-turn dataset validation failed with {len(errors)} errors")

        return is_valid, errors

    def validate_file(self, filepath: str) -> Tuple[bool, List[str]]:
        """
        Validate a dataset file.

        Args:
            filepath: Path to dataset file

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        from .loader import DatasetLoader

        try:
            loader = DatasetLoader()
            df = loader.load(filepath)
            return self.validate(df)
        except Exception as e:
            return False, [f"Failed to load file: {str(e)}"]