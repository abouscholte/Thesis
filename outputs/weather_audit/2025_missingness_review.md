# 2025 weather missingness review

Reviewed 18 November 2026 without inspecting delay labels or model performance.

- All 2,307,766 eligible flights remain. 4,062 (0.176%) have no observation at or before the prediction cutoff within 60 minutes.
- Missing core features concentrate in August and September (approximately 8.5% and 7.6–7.7%, respectively), not uniformly across the year.
- The raw files contain actual numeric measurements with source 999 and blank quality codes around late August/early September. The established source/quality allowlist masks these values. These are not necessarily absent measurements or proven erroneous values.
- 28,590 flights match an observation but have all four core fields masked/missing; most occur August 28–September 2.
- Source 999 was not defined in the official source list inspected: https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-source-list.pdf . Its provenance/quality status remains unresolved.
- Decision: retain the established cleaning policy and identical flight populations for both variants. Use the existing training-fitted median imputation and missingness indicators. Do not fill from future observations or learn imputation values from 2025.
- Do not interpret high timestamp match coverage as equivalent to usable measurement coverage. Report the time-concentrated missingness as a limitation. Any later change to the quality policy requires documented justification and consistent treatment of both years.

Files: 2025_monthly_missingness.csv; 2025_airport_month_missingness.csv; 2025_unmatched_dates.csv; 2025_missingness_reasons.csv; 2025_raw_quality_missingness.csv.
