"""Shared, prediction-time-safe data preparation for notebook and scoring."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SEED = 42
TARGET = 'delivered_late'
CATEGORICAL = ['origin_city', 'destination_city', 'carrier', 'service_level',
               'goods_type', 'customer_priority_tier', 'weather_condition']
NUMERIC = ['shipment_month', 'distance_km', 'weight_kg', 'num_stops',
           'is_cross_border', 'scheduled_transit_hours', 'pickup_delay_minutes',
           'driver_experience_years', 'vehicle_age_years', 'traffic_index',
           'route_congestion_score', 'prior_late_deliveries_30d', 'fuel_cost_cad']
PREDICTORS = CATEGORICAL + NUMERIC
ENGINEERED = ['month_sin', 'month_cos', 'pickup_delay_hours',
              'pickup_delay_budget_fraction', 'required_speed_kmh', 'weight_suspect']

def normalize_categories(frame):
    """Normalize whitespace and observed case variants without learning from labels."""
    result = frame.copy()
    for col in CATEGORICAL:
        if col in result:
            result[col] = result[col].map(
                lambda x: ' '.join(str(x).split()) if pd.notna(x) else np.nan)
    for col in ['service_level', 'weather_condition', 'customer_priority_tier']:
        if col in result:
            result[col] = result[col].str.title()
    if 'shipment_id' in result:
        result['shipment_id'] = result['shipment_id'].astype(str).str.strip().str.upper()
    return result

def clean_historical(raw):
    """Deduplicate before splitting; quarantine unresolved duplicate references."""
    frame = normalize_categories(raw)
    before = len(frame)
    frame = frame.drop_duplicates().copy()
    normalized_duplicates = before - len(frame)
    conflicting = frame['shipment_id'].duplicated(keep=False)
    quarantine = frame.loc[conflicting].copy()
    frame = frame.loc[~conflicting].reset_index(drop=True)
    if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError('Target must be present and binary; do not impute labels.')
    if not frame['shipment_id'].is_unique:
        raise ValueError('Shipment IDs must be unique before splitting.')
    audit = {'raw_rows': before, 'exact_duplicates': int(raw.duplicated().sum()),
             'normalized_duplicates_removed': normalized_duplicates,
             'conflicting_rows_quarantined': len(quarantine),
             'conflicting_ids': quarantine['shipment_id'].unique().tolist(),
             'clean_rows': len(frame), 'late_count': int(frame[TARGET].sum()),
             'late_rate': float(frame[TARGET].mean()),
             'suspect_weights': int((pd.to_numeric(frame['weight_kg'], errors='coerce') > 50000).sum())}
    return frame, quarantine, audit

class DispatchFeatures(BaseEstimator, TransformerMixin):
    """Stateless rules available after pickup; fitted imputation occurs downstream."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing = sorted(set(PREDICTORS) - set(X.columns))
        if missing:
            raise ValueError('Missing dispatch fields: ' + ', '.join(missing))
        frame = normalize_categories(X[PREDICTORS])
        for col in NUMERIC:
            frame[col] = pd.to_numeric(frame[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        frame['weight_suspect'] = (frame['weight_kg'] > 50000).astype(int)
        frame.loc[(frame['weight_kg'] > 50000) | (frame['weight_kg'] <= 0), 'weight_kg'] = np.nan
        for col in ['distance_km', 'scheduled_transit_hours']:
            frame.loc[frame[col] <= 0, col] = np.nan
        for col in ['num_stops', 'driver_experience_years', 'vehicle_age_years',
                    'prior_late_deliveries_30d', 'fuel_cost_cad']:
            frame.loc[frame[col] < 0, col] = np.nan
        for col, low, high in [('shipment_month', 1, 12), ('traffic_index', 0, 100),
                                ('route_congestion_score', 0, 1)]:
            frame.loc[~frame[col].between(low, high) & frame[col].notna(), col] = np.nan
        frame.loc[~frame['is_cross_border'].isin([0, 1]), 'is_cross_border'] = np.nan
        frame['month_sin'] = np.sin(2 * np.pi * frame['shipment_month'] / 12)
        frame['month_cos'] = np.cos(2 * np.pi * frame['shipment_month'] / 12)
        frame['pickup_delay_hours'] = frame['pickup_delay_minutes'].clip(lower=0) / 60
        frame['pickup_delay_budget_fraction'] = frame['pickup_delay_hours'] / frame['scheduled_transit_hours']
        frame['required_speed_kmh'] = frame['distance_km'] / frame['scheduled_transit_hours']
        return frame
