from pathlib import Path
from typing import Optional, List, Union
import polars as pl
import json
from surogate.utils.logger import get_logger

from .test_case import TestCase, MultiTurnTestCase, Turn

logger = get_logger()


class DatasetLoader:
    """Load datasets from various formats using Polars."""

    SUPPORTED_FORMATS = ['.jsonl', '.csv']

    def __init__(self):
        """Initialize dataset loader."""
        pass

    def load(self, filepath: str, format: Optional[str] = None) -> pl.DataFrame:
        """
        Load dataset from file.

        Args:
            filepath: Path to dataset file
            format: File format (auto-detected if None)

        Returns:
            Polars DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format not supported
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        # Auto-detect format
        if format is None:
            format = path.suffix.lower()

        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.SUPPORTED_FORMATS}")

        logger.info(f"Loading dataset from {filepath} (format: {format})")

        if format == '.jsonl':
            df = self._load_jsonl(path)
        elif format == '.csv':
            df = self._load_csv(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"✓ Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        return df

    def detect_dataset_type(self, path: str) -> str:
        """
        Detect if dataset is single-turn or multi-turn.

        Args:
            path: Path to dataset file

        Returns:
            'single_turn' or 'multi_turn'
        """
        df = self.load(path)

        # Check if 'turns' column exists (multi-turn indicator)
        if 'turns' in df.columns:
            return 'multi_turn'

        # Check if we have conversation-related columns
        if 'messages' in df.columns or 'conversation' in df.columns:
            return 'multi_turn'

        # Check if we have single-turn columns
        # FIX: Only 'input' is required for single-turn (expected_output is optional)
        if 'input' in df.columns:
            return 'single_turn'

        # Default to single-turn (most common case)
        logger.warning("Could not definitively detect dataset type, defaulting to single_turn")
        return 'single_turn'

    def _load_jsonl(self, path: Path) -> pl.DataFrame:
        """Load JSONL file."""
        try:
            # Read JSONL using Polars
            df = pl.read_ndjson(path)
            return df
        except Exception as e:
            logger.error(f"Failed to load JSONL with Polars: {e}")
            # Fallback: manual parsing for complex nested structures
            return self._load_jsonl_manual(path)

    def _load_jsonl_manual(self, path: Path) -> pl.DataFrame:
        """Manually parse JSONL for complex structures."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

        if not data:
            raise ValueError("No valid data found in JSONL file")

        # Convert to DataFrame
        df = pl.DataFrame(data)
        return df

    def _load_csv(self, path: Path) -> pl.DataFrame:
        """Load CSV file."""
        try:
            df = pl.read_csv(path)
            return df
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")

    def load_test_cases(self, filepath: str) -> List[Union[TestCase, MultiTurnTestCase]]:
        """
        Load test cases from file.

        Args:
            filepath: Path to dataset file

        Returns:
            List of TestCase or MultiTurnTestCase objects
        """
        df = self.load(filepath)

        # Detect type
        is_multi_turn = 'turns' in df.columns

        test_cases = []
        for row in df.iter_rows(named=True):
            try:
                if is_multi_turn:
                    test_case = self._row_to_multi_turn_test_case(row)
                else:
                    test_case = self._row_to_test_case(row)
                test_cases.append(test_case)
            except Exception as e:
                logger.warning(f"Failed to convert row to test case: {e}")
                continue

        logger.info(f"✓ Loaded {len(test_cases)} test cases")
        return test_cases

    def _row_to_test_case(self, row: dict) -> TestCase:
        """Convert DataFrame row to TestCase."""
        input_text = row.get('input')
        expected_output = row.get('expected_output')

        # Handle metadata - prioritize explicit 'metadata' field
        if 'metadata' in row and row['metadata']:
            # If there's an explicit metadata field, use it
            metadata = row['metadata']
            if isinstance(metadata, str):
                # If metadata is stored as JSON string, parse it
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse metadata as JSON: {metadata}")
                    metadata = {}
        else:
            # Otherwise, extract metadata from all other fields
            metadata = {k: v for k, v in row.items()
                        if k not in ['input', 'expected_output']}

        return TestCase(
            input=input_text,
            expected_output=expected_output,
            metadata=metadata
        )

    def _row_to_multi_turn_test_case(self, row: dict) -> MultiTurnTestCase:
        """Convert DataFrame row to MultiTurnTestCase."""
        turns_data = row.get('turns', [])
        expected_final_output = row.get('expected_final_output')

        # Convert turns
        turns = [Turn.from_dict(turn_dict) for turn_dict in turns_data]

        # Extract metadata
        metadata = {k: v for k, v in row.items()
                    if k not in ['turns', 'expected_final_output']}

        return MultiTurnTestCase(
            turns=turns,
            expected_final_output=expected_final_output,
            metadata=metadata
        )

    def save(self, df: pl.DataFrame, filepath: str, format: Optional[str] = None):
        """
        Save dataset to file.

        Args:
            df: Polars DataFrame
            filepath: Output file path
            format: File format (auto-detected if None)
        """
        path = Path(filepath)

        # Auto-detect format
        if format is None:
            format = path.suffix.lower()

        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saving dataset to {filepath} (format: {format})")

        if format == '.jsonl':
            df.write_ndjson(path)
        elif format == '.csv':
            df.write_csv(path)

        logger.info(f"✓ Saved dataset: {len(df)} rows")