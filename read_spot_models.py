import pandas as pd
import re

class SPOT:
    """
    Reads in SPOT isochrone models
    Returns several DataFrames, one for each age
    """

    def __init__(self, filename, verbose=True):
        self.filename = filename

        if verbose:
            print("Loading in " + self.filename)

    def read_iso_file(self):

        #age_string = "## log10 Age(yr) ="
        #age_re = re.compile(age_string)
        age_re = re.compile(r'^##\s*log10 Age\(yr\)\s*=\s*([0-9.]+)')

        #header_string = "## logAge"
        #header_re = re.compile(header_string)
        header_re = re.compile(r'^##\s*logAge\b', re.IGNORECASE)

        with open(self.filename, "r") as f:
            lines = f.readlines()

        i = 0
        sections = {}
        n = len(lines)

        while i < n:
            m = age_re.match(lines[i])
            if not m:
                i += 1
                continue

            age = float(m.group(1))
            #print(age)
            i += 1
            while i < n and not header_re.match(lines[i]):
                i +=1

            if i >= n:
                break

            header_line = lines[i].lstrip('#').strip()
            col_names = re.split(r'\s+', header_line)
            #print(col_names)
            i +=1

            #collect data rows until next ## block
            rows = []
            while i < n:
                if lines[i].startswith('##'):
                    break
                line_stripped = lines[i].strip()
                if line_stripped == '':
                    i += 1
                    continue
                vals = re.split(r'\s+', line_stripped)
                if len(vals) >= len(col_names):
                    rows.append(vals[:len(col_names)])
                else:
                    print(f"Warning: skipping line with insufficient columns: {lines[i]}")
                    pass
                i += 1

            if rows:
                df = pd.DataFrame(rows, columns=col_names)
            else:
                df = pd.DataFrame(columns=col_names)
                print(f"Warning: no data rows found for age {age} in file {self.filename}")

            sections[age] = df
        print(f"Found {len(sections)} age sections")
        return sections

#Example usage:
#dfs = SPOT("isochrones/SPOTS/isos/f000.isoc").read_iso_file()



