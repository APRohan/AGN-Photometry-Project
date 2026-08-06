# %%
import polars as pl

catalina_path = "/home/rohan/agn_photometry/photometry/results/catalina_photometry.parquet"
alerce_path   = "/home/rohan/agn_photometry/photometry/results/alerce_photometry.parquet"
atlas_path    = "/home/rohan/agn_photometry/photometry/results/atlas_photometry.parquet"

# outputs for this new analysis live under structure_function_fractional_variability/results
sf_fvar_results_dir = "/home/rohan/agn_photometry/structure_function_fractional_variability/results"

df_catalina = pl.read_parquet(catalina_path)
df_alerce   = pl.read_parquet(alerce_path)
df_atlas    = pl.read_parquet(atlas_path)

def cadence_summary(df, time_col, group_cols, survey_name):
    return (
        df.group_by(group_cols)
        .agg([
            pl.len().alias("N"),
            (pl.col(time_col).max() - pl.col(time_col).min()).alias("baseline_days"),
        ])
        .with_columns(pl.lit(survey_name).alias("survey"))
        .sort("N", descending=True)
    )

catalina_cadence = cadence_summary(
    df_catalina, "MJD", ["source_name"], "Catalina"
).with_columns(pl.lit(None, dtype=pl.Utf8).alias("band"))

alerce_cadence = cadence_summary(
    df_alerce, "mjd", ["source_name", "fid"], "ALERCE"
).rename({"fid": "band"}).with_columns(pl.col("band").cast(pl.Utf8))

atlas_cadence = cadence_summary(
    df_atlas, "MJD", ["source_name", "F"], "ATLAS"
).rename({"F": "band"}).with_columns(pl.col("band").cast(pl.Utf8))

cadence_all = pl.concat(
    [catalina_cadence, alerce_cadence, atlas_cadence],
    how="diagonal",
).select(["survey", "source_name", "band", "N", "baseline_days"])

# Tiering: N >= 50 and baseline >= 365d -> trusted; N >= 20 -> marginal; else -> excluded
cadence_all = cadence_all.with_columns(
    pl.when((pl.col("N") >= 50) & (pl.col("baseline_days") >= 365))
      .then(pl.lit("trusted"))
      .when(pl.col("N") >= 20)
      .then(pl.lit("marginal"))
      .otherwise(pl.lit("excluded"))
      .alias("tier")
)

pl.Config.set_tbl_rows(-1)
print(cadence_all)

print("\nTier counts by survey:")
print(cadence_all.group_by(["survey", "tier"]).agg(pl.len().alias("count")).sort(["survey", "tier"]))

output_path = f"{sf_fvar_results_dir}/cadence_summary.parquet"
cadence_all.write_parquet(output_path)
print(f"\nSaved to {output_path}")

# %%
import polars as pl
import numpy as np

catalina_path  = "/home/rohan/agn_photometry/photometry/results/catalina_photometry.parquet"
alerce_path    = "/home/rohan/agn_photometry/photometry/results/alerce_photometry.parquet"
atlas_path     = "/home/rohan/agn_photometry/photometry/results/atlas_photometry.parquet"
cadence_path   = "/home/rohan/agn_photometry/structure_function_fractional_variability/results/cadence_summary.parquet"

df_catalina = pl.read_parquet(catalina_path)
df_alerce   = pl.read_parquet(alerce_path)
df_atlas    = pl.read_parquet(atlas_path)
cadence     = pl.read_parquet(cadence_path)

def remove_outliers_mask(mag_vals, threshold=4.0):
    """Same MAD-based logic as Photometry.ipynb's remove_outliers, returns a boolean mask."""
    y_arr = mag_vals.to_numpy()
    n = len(y_arr)
    if n < 3:
        return np.ones(n, dtype=bool)
    median = np.median(y_arr)
    mad = np.median(np.abs(y_arr - median))
    if mad == 0:
        return np.ones(n, dtype=bool)
    modified_z = 0.6745 * (y_arr - median) / mad
    return np.abs(modified_z) < threshold

def apply_mad_filter(df, mag_col, group_cols, err_col=None):
    """Apply MAD outlier removal independently within each (source, band) group.
    If err_col is given, points that are outliers in EITHER mag or error are dropped."""
    kept_frames = []
    for _, sub in df.group_by(group_cols):
        mask = remove_outliers_mask(sub[mag_col])
        if err_col is not None:
            err_mask = remove_outliers_mask(sub[err_col])
            mask = mask & err_mask
        kept_frames.append(sub.filter(pl.Series(mask)))
    return pl.concat(kept_frames)

df_catalina = apply_mad_filter(df_catalina, "mag", ["source_name"], err_col="mag_err")
df_alerce   = apply_mad_filter(df_alerce, "magpsf_corr", ["source_name", "fid"], err_col="sigmapsf")
df_atlas    = apply_mad_filter(df_atlas, "m", ["source_name", "F"], err_col="dm")
df_atlas = df_atlas.rename({"F": "band"})

# ---- mag -> flux, per row ----
def add_flux_cols(df, mag_col, magerr_col):
    return df.with_columns([
        (10 ** (-0.4 * pl.col(mag_col))).alias("F"),
    ]).with_columns([
        (0.4 * np.log(10) * pl.col("F") * pl.col(magerr_col)).alias("sigma_F"),
    ])

df_catalina = add_flux_cols(df_catalina, "mag", "mag_err").with_columns(
    pl.lit(None, dtype=pl.Utf8).alias("band")
)
df_alerce = add_flux_cols(df_alerce, "magpsf_corr", "sigmapsf").with_columns(
    pl.col("fid").cast(pl.Utf8).alias("band")
)
df_atlas = add_flux_cols(df_atlas, "m", "dm").with_columns(
    pl.col("band").cast(pl.Utf8)
)

df_catalina = df_catalina.with_columns(pl.lit("Catalina").alias("survey"))
df_alerce   = df_alerce.with_columns(pl.lit("ALERCE").alias("survey"))
df_atlas    = df_atlas.with_columns(pl.lit("ATLAS").alias("survey"))

# ---- Fvar per (survey, source_name, band) ----
def fvar_per_group(df):
    return (
        df.group_by(["survey", "source_name", "band"])
        .agg([
            pl.len().alias("N"),
            pl.col("F").mean().alias("F_mean"),
            pl.col("F").var(ddof=1).alias("S2"),
            (pl.col("sigma_F") ** 2).mean().alias("sigma2_err_mean"),
        ])
        .with_columns(
            ((pl.col("S2") - pl.col("sigma2_err_mean")) / (pl.col("F_mean") ** 2))
            .alias("excess_var_norm")
        )
        .with_columns(
            pl.when(pl.col("excess_var_norm") > 0)
              .then(pl.col("excess_var_norm").sqrt())
              .otherwise(None)
              .alias("Fvar")
        )
    )

fvar_catalina = fvar_per_group(df_catalina)
fvar_alerce   = fvar_per_group(df_alerce)
fvar_atlas    = fvar_per_group(df_atlas)

fvar_all = pl.concat([fvar_catalina, fvar_alerce, fvar_atlas], how="diagonal")

# ---- join cadence tiers, drop excluded ----
fvar_all = fvar_all.join(
    cadence.select(["survey", "source_name", "band", "baseline_days", "tier"]),
    on=["survey", "source_name", "band"],
    how="left",
    nulls_equal=True,
)

fvar_trusted = fvar_all.filter(pl.col("tier") != "excluded")

pl.Config.set_tbl_rows(-1)
print(fvar_trusted.select([
    "survey", "source_name", "band", "N", "baseline_days", "tier", "Fvar"
]).sort(["survey", "source_name"]))

output_path = "/home/rohan/agn_photometry/structure_function_fractional_variability/results/fvar_all_surveys.parquet"
fvar_trusted.write_parquet(output_path)
print(f"\nSaved to {output_path}")

# %%
import polars as pl
import numpy as np

catalina_path = "/home/rohan/agn_photometry/photometry/results/catalina_photometry.parquet"
alerce_path   = "/home/rohan/agn_photometry/photometry/results/alerce_photometry.parquet"

df_catalina = pl.read_parquet(catalina_path)
df_alerce   = pl.read_parquet(alerce_path)

print("=" * 70)
print("CATALINA: 205601.38-062049.7 (Fvar = 3.46 outlier)")
print("=" * 70)
sub = df_catalina.filter(pl.col("source_name") == "205601.38-062049.7")
print(f"N = {sub.height}")
print(sub.select(["MJD", "mag", "mag_err"]).describe())
print("\nSorted by mag (brightest and faintest 5 points):")
print(sub.sort("mag").select(["MJD", "mag", "mag_err"]).head(5))
print(sub.sort("mag").select(["MJD", "mag", "mag_err"]).tail(5))

print("\n" + "=" * 70)
print("ALERCE: 132404.20+433407.1 (null Fvar in BOTH bands despite N=225/139)")
print("=" * 70)
sub2 = df_alerce.filter(pl.col("source_name") == "132404.20+433407.1")
print(f"Total N (both bands) = {sub2.height}")
for band in sub2["fid"].unique().sort():
    b = sub2.filter(pl.col("fid") == band)
    print(f"\n--- fid={band}, N={b.height} ---")
    print(b.select(["magpsf_corr", "sigmapsf"]).describe())

# %%
import polars as pl
import numpy as np

atlas_path = "/home/rohan/agn_photometry/photometry/results/atlas_photometry.parquet"
df_atlas = pl.read_parquet(atlas_path)

# Same MAD filter as the main pipeline, applied here for a fair comparison
def remove_outliers_mask(mag_vals, threshold=4.0):
    y_arr = mag_vals.to_numpy()
    n = len(y_arr)
    if n < 3:
        return np.ones(n, dtype=bool)
    median = np.median(y_arr)
    mad = np.median(np.abs(y_arr - median))
    if mad == 0:
        return np.ones(n, dtype=bool)
    modified_z = 0.6745 * (y_arr - median) / mad
    return np.abs(modified_z) < threshold

sources_to_check = ["001255.59+014750.9", "113618.90+125554.8", "150348.33+193941.3"]

for src in sources_to_check:
    print("=" * 70)
    print(f"{src}")
    print("=" * 70)
    for band in ["c", "o"]:
        sub = df_atlas.filter(
            (pl.col("source_name") == src) & (pl.col("F") == band)
        )
        mag_mask = remove_outliers_mask(sub["m"])
        err_mask = remove_outliers_mask(sub["dm"])
        mask = mag_mask & err_mask
        sub_clean = sub.filter(pl.Series(mask))

        m = sub_clean["m"].to_numpy()
        dm = sub_clean["dm"].to_numpy()
        flux = 10 ** (-0.4 * m)
        sigma_flux = 0.4 * np.log(10) * flux * dm

        S2 = np.var(flux, ddof=1)
        sigma2_err_mean = np.mean(sigma_flux ** 2)
        excess = S2 - sigma2_err_mean

        print(f"\n--- band={band}, N={len(m)} (after MAD filter) ---")
        print(f"  mag std      = {np.std(m):.5f}")
        print(f"  mean dm      = {np.mean(dm):.5f}")
        print(f"  median dm    = {np.median(dm):.5f}")
        print(f"  max dm       = {np.max(dm):.5f}")
        print(f"  flux S2      = {S2:.6e}")
        print(f"  mean sigma_F^2 = {sigma2_err_mean:.6e}")
        print(f"  excess (S2 - sigma2_err_mean) = {excess:.6e}  {'-> NEGATIVE (null Fvar)' if excess <= 0 else '-> positive'}")

# %%
import polars as pl
combined_targets = pl.read_csv("/home/rohan/agn_photometry/data/sdss_meeting_targets_dr18.csv")  # adjust path if needed
print(combined_targets.columns)
print(combined_targets["Name of the source"].head(10))


