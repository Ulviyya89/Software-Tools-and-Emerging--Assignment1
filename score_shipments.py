"""Score a dispatch export and optionally post a Slack summary."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from src.alerts import build_alert, post_alert, score_shipments, alert_event_key

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--send', action='store_true', help='Post if private .env also enables Slack.')
    parser.add_argument('--live', action='store_true', help='Requires shipment_status; retain only in_transit rows.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    bundle = joblib.load(root / 'models/delivery_delay_model.joblib')
    incoming = pd.read_csv(args.input)
    source = 'historical replay'
    if args.live:
        if 'shipment_status' not in incoming:
            raise ValueError('Live scoring requires shipment_status from an operational feed.')
        incoming = incoming.loc[incoming['shipment_status'].astype(str).str.strip().str.lower() == 'in_transit']
        source = 'active in-transit export'
    if incoming.empty:
        print('No eligible shipments to score.')
        return
    scored = score_shipments(bundle, incoming)
    (root / 'reports').mkdir(exist_ok=True)
    scored.to_csv(root / 'reports/latest_scored_shipments.csv', index=False)
    message, risky = build_alert(scored, bundle['threshold'], source=source)
    print(message if message else 'No high-risk shipments detected.')
    if args.send:
        print(post_alert(message, root, event_key=alert_event_key(risky, bundle['threshold'], source)))

if __name__ == '__main__':
    main()
