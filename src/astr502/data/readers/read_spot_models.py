import pandas as pd
import re

class SPOT:
    """
    Reads in SPOT isochrone models
    Returns several DataFrames, one for each age
    """

    def __init__(self, filename, verbose=True):
        self.filename = filename
        self.verbose = verbose
        if verbose:
            print("Loading in " + self.filename)

    def read_iso_file(self):

        #age_string = "## log10 Age(yr) ="
        #age_re = re.compile(age_string)
        #age_re = re.compile(r'^##\s*log10 Age\(yr\)\s*=\s*([0-9.]+)')

        #header_string = "## logAge"
        #header_re = re.compile(header_string)
        header_re = re.compile(r'^##\s*logAge\b', re.IGNORECASE)

        with open(self.filename, "r") as f:
            lines = f.readlines()

        i = 0
        sections = {}
        n = len(lines)

        while i < n:
            m = header_re.match(lines[i])
            if not m:
                i += 1
                continue

            # age = float(m.group(1))
            # #print(age)
            # i += 1

            header_line = lines[i].lstrip('#').strip()
            col_names = re.split(r'\s+', header_line)
            i += 1

            rows = []
            while i <n and not lines[i].startswith('##'):
                line_stripped = lines[i].strip()
                i += 1
                if line_stripped == '':
                    continue

                vals = re.split(r'\s+', line_stripped)
                if len(vals) >= len(col_names):
                    rows.append(vals[:len(col_names)])
                else:
                    if self.verbose:
                        print(f"Warning: skipping line with insufficient columns: {line_stripped}")

            if rows:
                df = pd.DataFrame(rows, columns=col_names)
                df = df.apply(pd.to_numeric, errors='coerce')
                log_col = next((c for c in df.columns if c.lower() == 'logage'), None)
                if log_col is None:
                    if self.verbose:
                        print(f"Warning: no logAge column found in section starting at line {i-len(rows)}")
                    section_key = len(sections)
                else:
                    section_key = float(df.iloc[0][log_col])
            else:
                df = pd.DataFrame(columns=col_names)
                if self.verbose:
                    print(f"Warning: no data rows found for section starting at line {i} in file {self.filename}")
                section_key = len(sections)

            sections[section_key] = df

        if self.verbose:
            print(f"Found {len(sections)} age sections")
        return sections

if __name__ == "__main__":
    dfs = SPOT("/Users/archon/classes/ASTR_502/workstation/data/raw/isochrones/SPOTS/isos/f000.isoc").read_iso_file()
    for df in dfs:
        print(df)
