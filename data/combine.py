#!/usr/bin/env python3
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

# 1. Load raw data & metadata
df = pd.read_csv('census_data_output.csv')
with open('data.json') as f:
    var_meta = json.load(f)['variables']

# 2. Prepare for area → area_km2
if 'area' in df.columns:
    df['area_km2'] = df['area']
elif 'ALAND' in df.columns:
    # ALAND in m² → km²
    df['area_km2'] = df['ALAND'] / 1e6
else:
    print("⚠️  No 'area' or 'ALAND' column found — skipping population_density")

# 3. Define your derived‐feature formulas
to_compute = {
    'population_density': {
        'num': ['B01003_001E'],
        'den': ['area_km2'],
    },
    'disposable_income': {
        'num': ['B19013_001E'],
        'den': ['B25077_001E'],
    },
    'education_index': {
        'num': ['B15003_022E','B15003_023E','B15003_024E'],
        'den': ['B01003_001E'],
    },
    'broadband_penetration': {
        'num': ['B28002_004E','B28002_007E'],
        'den': ['B28002_001E'],
    },
    'commute_index': {
        'num': ['B08303_003E'],
        'den': ['B08303_001E'],
    },
    'unemployment_ratio': {
        'num': ['B23025_005E'],
        'den': ['B23025_003E'],
    },
}

computed = []
for name, parts in to_compute.items():
    num_cols = parts['num']
    den_cols = parts['den']
    # only compute if *all* needed columns exist
    if all(c in df.columns for c in num_cols + den_cols):
        df[name] = df[num_cols].sum(axis=1) / df[den_cols].sum(axis=1)
        computed.append(name)
    else:
        missing = [c for c in num_cols + den_cols if c not in df.columns]
        print(f"⚠️  Skipping {name}: missing columns {missing}")

if not computed:
    raise RuntimeError("No derived features could be computed—check your input columns")

# 4. Normalize (z-score) only the ones we actually computed
scaler = StandardScaler()
z = scaler.fit_transform(df[computed])
for i, col in enumerate(computed):
    df[f"{col}_z"] = z[:, i]

# 5. Rename to human-readable using data.json metadata
for code in computed:
    meta = var_meta.get(code, {})
    # prefer the concept, then the label, then the code itself
    human = meta.get('concept') or meta.get('label') or code
    df.rename(columns={
        code:       human,
        f"{code}_z": f"{human} (z-score)",
    }, inplace=True)

# 6. Write out
out = 'census_derived_features.csv'
df.to_csv(out, index=False)
print(f"✅ Done — wrote {out} with derived features: {computed}")
