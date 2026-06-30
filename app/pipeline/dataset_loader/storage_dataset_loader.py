from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from app.domain.models.artifacts import RawDataset

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


class StorageDatasetLoader:
    """
    Loads CSV or XLS/XLSX datasets from Azure Blob SAS URLs or local file paths.

    Azure Blob (production): receives an HTTPS SAS URL → pandas reads directly over HTTP.
    Local path (testing): receives a filesystem path → reads from disk.
    """

    def load(self, dataset_path: str) -> RawDataset:
        if dataset_path.startswith("http://") or dataset_path.startswith("https://"):
            return self._load_from_url(dataset_path)
        return self._load_from_local(dataset_path)

    # ── URL (Azure Blob SAS) ──────────────────────────────────────────────────

    def _load_from_url(self, url: str) -> RawDataset:
        # Extract extension from the URL path (before the query string)
        path_part = urlparse(url).path
        suffix = Path(path_part).suffix.lower()
        file_name = Path(path_part).name

        logger.info("[LOADER] Reading from URL: %s (ext=%s)", path_part, suffix)

        try:
            if suffix == ".csv" or suffix not in (".xls", ".xlsx"):
                frame = pd.read_csv(url, dtype=str, encoding="utf-8-sig", low_memory=False)
            else:
                frame = pd.read_excel(url, dtype=str)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not load dataset from URL: '{url}' — {exc}"
            ) from exc

        frame.columns = [str(col).strip() for col in frame.columns]
        frame["_source_file"] = file_name
        return RawDataset(
            name="raw_dataset",
            dataframe=frame,
            source_path=url,
            source_files=[file_name],
        )

    # ── Local file / directory ────────────────────────────────────────────────

    def _load_from_local(self, dataset_path: str) -> RawDataset:
        path = Path(dataset_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: '{dataset_path}'."
            )

        if path.is_file():
            return self._load_single_file(path)

        if path.is_dir():
            return self._load_directory(path)

        raise ValueError(f"Dataset path is neither a file nor a directory: '{dataset_path}'")

    def _load_single_file(self, path: Path) -> RawDataset:
        frame = self._read_file(path)
        frame.columns = [str(col).strip() for col in frame.columns]
        frame["_source_file"] = path.name
        return RawDataset(
            name="raw_dataset",
            dataframe=frame,
            source_path=str(path),
            source_files=[path.name],
        )

    def _load_directory(self, path: Path) -> RawDataset:
        files: list[Path] = []
        for ext in sorted(_SUPPORTED_EXTENSIONS):
            files.extend(sorted(path.rglob(f"*{ext}")))

        if not files:
            raise FileNotFoundError(
                f"No CSV or XLS/XLSX files found in directory: '{path}'."
            )

        frames: list[pd.DataFrame] = []
        for file_path in files:
            frame = self._read_file(file_path)
            frame.columns = [str(col).strip() for col in frame.columns]
            frame["_source_file"] = file_path.name
            frames.append(frame)

        dataframe = pd.concat(frames, ignore_index=True, sort=False)
        return RawDataset(
            name="raw_dataset",
            dataframe=dataframe,
            source_path=str(path),
            source_files=[f.name for f in files],
        )

    @staticmethod
    def _read_file(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            delimiter = StorageDatasetLoader._detect_csv_delimiter(path)
            logger.info("Reading CSV '%s' with delimiter=%r", path.name, delimiter)
            return pd.read_csv(path, sep=delimiter, encoding="utf-8-sig", dtype=str)
        if suffix in (".xls", ".xlsx"):
            logger.info("Reading Excel file '%s'", path.name)
            return pd.read_excel(path, dtype=str)
        raise ValueError(f"Unsupported file format '{suffix}' for: '{path}'.")

    @staticmethod
    def _detect_csv_delimiter(path: Path) -> str:
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
                first_line = ""
                for raw_line in fh:
                    stripped = raw_line.strip()
                    if stripped:
                        first_line = stripped
                        break
        except Exception:
            return ","

        if not first_line:
            return ","

        def _count_unquoted(line: str, sep: str) -> int:
            count, in_quote = 0, False
            for ch in line:
                if ch == '"':
                    in_quote = not in_quote
                elif not in_quote and ch == sep:
                    count += 1
            return count

        candidates = {d: _count_unquoted(first_line, d) for d in (",", ";", "\t")}
        best = max(candidates, key=lambda k: candidates[k])
        return best if candidates[best] > 0 else ","
