"""Shared BTS preparation; airport selection is fixed using 2024 data."""
from pathlib import Path
import calendar
import json
import pandas as pd

COLUMNS = [
    'Month', 'DayOfWeek', 'Reporting_Airline', 'Dest', 'CRSDepTime', 'Distance',
    'FlightDate', 'Origin', 'Flight_Number_Reporting_Airline',
    'DepDel15', 'DepDelay', 'Cancelled',
]
KEY = ['FlightDate', 'Reporting_Airline', 'Flight_Number_Reporting_Airline',
       'Origin', 'Dest', 'CRSDepTime']
TIMEZONES = {
    'ATL': 'America/New_York', 'DFW': 'America/Chicago',
    'DEN': 'America/Denver', 'ORD': 'America/Chicago',
    'CLT': 'America/New_York', 'LAX': 'America/Los_Angeles',
    'PHX': 'America/Phoenix', 'LAS': 'America/Los_Angeles',
    'SEA': 'America/Los_Angeles', 'LGA': 'America/New_York',
}

def load_flights(project_root, year):
    """Load exactly twelve monthly files and audit eligibility exclusions."""
    root = Path(project_root)
    airports = json.loads((root / 'outputs/2024_audit/airport_selection.json').read_text())['airports']
    assert len(airports) == len(set(airports)) == 10
    frames, counts = [], []
    for month in range(1, 13):
        path = root / f'data/raw/bts/{year}/otr_{month:02d}_{year}.csv'
        monthly = pd.read_csv(path, usecols=COLUMNS, dtype={'CRSDepTime': 'string'})
        dates = pd.to_datetime(monthly['FlightDate'], errors='raise')
        assert dates.notna().all(), f'{path.name}: missing dates'
        assert (dates.dt.year.eq(year) & dates.dt.month.eq(month)).all(), path.name
        selected = monthly.loc[monthly['Origin'].isin(airports)]
        assert selected['Cancelled'].isin([0, 1]).all(), f'{path.name}: invalid cancellation flag'
        operated = selected.loc[selected['Cancelled'].eq(0)]
        clean = operated.dropna(subset=['DepDel15'])
        frames.append(clean)
        counts.append(dict(file=path.name, raw=len(monthly), selected=len(selected),
                           cancelled=len(selected)-len(operated),
                           missing_label=len(operated)-len(clean), retained=len(clean)))
    flights = pd.concat(frames, ignore_index=True)
    flights['FlightDate'] = pd.to_datetime(flights['FlightDate'])
    audit = pd.DataFrame(counts)
    audit['cancelled_pct'] = 100 * audit['cancelled'] / audit['selected']
    validate_flights(flights, audit, airports, year)
    return flights, audit


def validate_flights(flights, audit, airports, year):
    """Check retained records without imputing or silently removing duplicates."""
    assert len(flights) == audit['retained'].sum()
    assert audit['selected'].eq(audit['cancelled']+audit['missing_label']+audit['retained']).all()
    assert set(flights['Origin']) == set(airports)
    assert flights['FlightDate'].dt.year.eq(year).all()
    assert flights['FlightDate'].dt.normalize().nunique() == 365 + calendar.isleap(year)
    assert flights['Cancelled'].eq(0).all()
    assert flights['DepDel15'].isin([0, 1]).all()
    assert flights['DepDelay'].notna().all()
    assert flights['DepDel15'].eq(flights['DepDelay'].ge(15)).all()
    assert not flights.duplicated(KEY).any(), 'Duplicate flight identifiers; investigate before continuing'


def prepare_timestamps(flights):
    """Interpret 2400 as next midnight; exclude ambiguous/nonexistent local times.

    Returns retained flights, excluded records, and a timestamp audit.
    Prediction cutoffs are two elapsed hours before scheduled departure.
    """
    flights = flights.copy()
    times = pd.to_numeric(flights['CRSDepTime'], errors='coerce')
    hours, minutes = times // 100, times % 100
    valid = ((times.eq(times.round()) & hours.between(0,23)
              & minutes.between(0,59)) | times.eq(2400)).fillna(False)
    assert valid.all(), f'Invalid scheduled times: {(~valid).sum()}'
    flights['scheduled_departure_local'] = (flights['FlightDate']
        + pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m'))
    flights['timezone'] = flights['Origin'].map(TIMEZONES)
    assert flights['timezone'].notna().all()
    flights['scheduled_departure_utc'] = pd.Series(pd.NaT, index=flights.index, dtype='datetime64[ns, UTC]')
    for timezone, group in flights.groupby('timezone'):
        local = group['scheduled_departure_local'].dt.tz_localize(
            timezone, ambiguous='NaT', nonexistent='NaT')
        flights.loc[group.index, 'scheduled_departure_utc'] = local.dt.tz_convert('UTC')
    unresolved = flights['scheduled_departure_utc'].isna()
    excluded = flights.loc[unresolved].copy()
    retained = flights.loc[~unresolved].copy()
    retained['prediction_cutoff_utc'] = retained['scheduled_departure_utc'] - pd.Timedelta(hours=2)
    assert retained['prediction_cutoff_utc'].notna().all()
    audit = pd.DataFrame([dict(before_timestamps=len(flights),
        times_recorded_as_2400=int(times.eq(2400).sum()),
        unresolved_timestamps=len(excluded), retained=len(retained))])
    return retained, excluded, audit


def save_flights(project_root, year, flights, audit, missingness, excluded, timestamp_audit):
    """Save processed records and audits; never modify raw files."""
    root = Path(project_root)
    processed = root / 'data/processed'
    audit_dir = root / f'outputs/{year}_audit'
    processed.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    flights.to_csv(processed / f'flights_{year}_top10.csv.gz', index=False)
    audit.to_csv(audit_dir / 'cleaning_counts.csv', index=False)
    missingness.to_csv(audit_dir / 'retained_missingness.csv')
    excluded.to_csv(audit_dir / 'excluded_unresolved_departures.csv', index=False)
    timestamp_audit.to_csv(audit_dir / 'timestamp_counts.csv', index=False)
