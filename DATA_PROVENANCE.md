# Data provenance

## NOAA weather downloads

- Dataset: NOAA NCEI Global Historical Climatology Network hourly (GHCNh), station-year PSV files.
- Download dates reported by the researcher: **27–28 October 2026** (Europe/Amsterdam). Exact dates per file were not recorded.
- Collection scope: ATL, DFW, DEN, ORD, CLT, LAX, PHX, LAS, SEA, and LGA; station-year files for 2024, 2025, and available 2026. This note does not certify that all planned files are present.
- Storage: `data/raw/noaa/{year}/noaa_{airport}_{year}.psv`.
- Intended use: 2024–2025 weather plus the early-January 2026 observations needed for late-2025 flight prediction cutoffs.
- Source: https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly
- This records the reported download window, not observation dates or the dataset release date.

## BTS flight downloads

- Dataset: Bureau of Transportation Statistics (BTS), TranStats Reporting Carrier On-Time Performance (1987–present); monthly domestic flight records supplied by reporting carriers.
- Source: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ
- Field definitions: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ
- Confirmed local coverage: January–December 2024, twelve monthly CSV files.
- Acquisition history: the January 2024 file was supplied and inspected on **13 October 2026**. The complete set of twelve 2024 files was confirmed present on **21 October 2026**. These are documented availability/inspection dates, not verified download dates; exact download dates were not recorded.
- Storage: `data/raw/bts/2024/otr_{month:02d}_2024.csv`. Original BTS filenames were renamed locally for chronological sorting.
- Raw files remain unchanged apart from filenames; filtering and timestamp preparation are performed separately in notebooks.
- Input manifest: `outputs/2024_audit/input_manifest.csv` records filenames, file sizes, SHA-256 hashes, and row counts for the audited 2024 files.
- Intended use: 2024 for model development and validation; 2025 for final evaluation. The local `data/raw/bts/2025/` directory was empty when this note was added on **28 October 2026**.
- Airport selection is derived from full-year 2024 reported flight counts; the downloaded source files retain all reporting origins.


## BTS 2025 downloads — 17 November 2026

- Downloaded all twelve official pre-zipped Reporting Carrier On-Time Performance files from https://www.transtats.bts.gov/PREZIP/ on **17 November 2026** (Europe/Amsterdam).
- Storage: `data/raw/bts/2025/otr_{month:02d}_2025.csv`; CSV contents extracted unchanged, with filenames renamed locally.
- Verified 7,001,619 total records. Every record has the expected 2025 year and month; every calendar day is represented. All twelve CSV headers match the January 2024 raw-file header and include the required modelling and audit fields.
- Manifest: `outputs/2025_audit/download_manifest.json` records each source URL, UTC acquisition timestamp, SHA-256 checksum, file size, row count, and date coverage.
- Earlier incorrect/incomplete downloads were preserved in `data/raw/bts/2025/previous_downloads_20261117T223806Z/`. These are not experiment inputs. Load only the twelve top-level `otr_*_2025.csv` files.
- Original downloaded ZIP archives are retained in `data/raw/bts/2025/download_staging/`.
- This step checked acquisition, schema, and date coverage only. Cleaning, weather matching, and final model evaluation remain separate steps; no 2025 model performance was examined.
