from __future__ import annotations

import re
import pandas as pd


class PARSEC:
    """
    Reads in PARSEC isochrone files produced by the CMD 3.x web interface.

    File format expectations
    ------------------------
    - Comment / metadata lines begin with '#'.
    - The *last* '#'-prefixed line before the data block is the column-header
      line, e.g.::

        # Zini  MH  logAge  ...  Mass  ...  Gmag  G_BPmag  G_RPmag  ...

    - Data rows immediately follow with no extra separator lines.
    - A single file may contain multiple age/metallicity blocks; all rows are
      stored in one flat DataFrame and then split by (logAge, MH) on the fly.

    The returned dict from ``read_iso_file`` has the same shape as the SPOT
    reader: keys are ``float(logAge)`` values, values are DataFrames for that
    age slice (across all metallicities present at that age, if the file mixes
    several [M/H] values — callers should further filter by MH if needed).

    Attributes
    ----------
    filename : str
    verbose  : bool
    """

    def __init__(self, filename: str, verbose: bool = True) -> None:
        self.filename = filename
        self.verbose = verbose
        if verbose:
            print(f"Loading in {self.filename}")

    # ------------------------------------------------------------------
    # public interface (mirrors SPOT.read_iso_file)
    # ------------------------------------------------------------------

    def read_iso_file(self) -> dict[float, pd.DataFrame]:
        """
        Parse the PARSEC .dat file and return a mapping of
        ``{logAge: DataFrame}`` — one entry per unique logAge value found in
        the file.

        The DataFrames retain *all* original columns (Zini, MH, logAge, Mini,
        Mass, photometric bands, …) so that callers can filter further on MH
        if they need to.
        """
        header_line, data_lines = self._split_file()

        col_names = re.split(r"\s+", header_line.strip())

        rows: list[list[str]] = []
        for raw in data_lines:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            vals = re.split(r"\s+", stripped)
            if len(vals) >= len(col_names):
                rows.append(vals[: len(col_names)])
            else:
                if self.verbose:
                    print(
                        f"Warning: skipping short row "
                        f"({len(vals)} < {len(col_names)} cols): {stripped!r}"
                    )

        if not rows:
            if self.verbose:
                print(f"Warning: no data rows found in {self.filename}")
            return {}

        df = pd.DataFrame(rows, columns=col_names)
        df = df.apply(pd.to_numeric, errors="coerce")

        log_col = self._find_logage_col(df)
        if log_col is None:
            raise ValueError(
                f"PARSEC file {self.filename!r} has no logAge column. "
                f"Columns found: {list(df.columns)}"
            )

        sections: dict[float, pd.DataFrame] = {}
        for age_val, group in df.groupby(log_col, sort=True):
            sections[float(age_val)] = group.reset_index(drop=True)

        if self.verbose:
            print(f"Found {len(sections)} age sections")

        return sections

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _split_file(self) -> tuple[str, list[str]]:
        """
        Return (header_line, data_lines).

        The header is the *last* '#'-prefixed line whose content looks like
        column names (i.e. non-numeric tokens after stripping the '#').
        """
        with open(self.filename, "r") as fh:
            lines = fh.readlines()

        header_line: str | None = None
        header_idx: int = 0

        for idx, line in enumerate(lines):
            if not line.startswith("#"):
                break
            candidate = line.lstrip("#").strip()
            # A genuine header line has mostly non-numeric tokens
            tokens = re.split(r"\s+", candidate)
            non_numeric = sum(1 for t in tokens if not re.match(r"^-?[\d.eE+]+$", t))
            if non_numeric >= max(1, len(tokens) // 2):
                header_line = candidate
                header_idx = idx
        else:
            # File is all comments — no data
            pass

        if header_line is None:
            raise ValueError(
                f"Could not locate a column-header line in {self.filename!r}. "
                "Expected the last '#'-prefixed line before data to list column names."
            )

        data_lines = lines[header_idx + 1 :]
        return header_line, data_lines

    @staticmethod
    def _find_logage_col(df: pd.DataFrame) -> str | None:
        for col in df.columns:
            if col.lower() in ("logage", "log_age", "log(age)"):
                return col
        return None


# ---------------------------------------------------------------------------
# quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "example.dat"
    sections = PARSEC(path).read_iso_file()
    for age_key, frame in list(sections.items())[:3]:
        print(f"logAge={age_key}  shape={frame.shape}  cols={list(frame.columns[:6])}")
