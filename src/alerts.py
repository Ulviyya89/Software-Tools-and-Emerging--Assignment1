"""Small Slack webhook client; never log the URL or raw network exceptions."""
import hashlib
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import pandas as pd
import requests
from dotenv import load_dotenv
from src.maplefreight import normalize_categories

def build_alert(scored, threshold, source='historical replay', top_n=5):
    risky = scored.loc[scored['risk_score'] >= threshold].sort_values('risk_score', ascending=False)
    if risky.empty:
        return None, risky
    escape = lambda value: html.escape(str(value), quote=False)
    lines = [f'*MapleFreight: {len(risky)} of {len(scored)} shipments at risk*',
             f'{source.capitalize()} | risk threshold {threshold:.0%}',
             f'Top {min(top_n, len(risky))} for dispatch review:']
    for row in risky.head(top_n).itertuples():
        lines.append(f'{escape(row.shipment_id)} — {escape(row.destination_city)} — '
                     f'{escape(row.carrier)} — risk {row.risk_score:.1%}')
    lines.append('Action: review pickup delay, route traffic and forecast; assess rerouting or customer notice.')
    if source == 'historical replay':
        lines.append('Assessment demonstration using historical data; these are not live shipments.')
    return '\n'.join(lines), risky

def alert_event_key(risky, threshold, source):
    """Include every flagged shipment, even when only five appear in the message."""
    rows = risky[['shipment_id', 'risk_score']].sort_values('shipment_id').copy()
    rows['risk_score'] = rows['risk_score'].round(8)
    content = {'source': source, 'threshold': float(threshold),
               'shipments': rows.to_dict('records')}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

def post_alert(message, project_root, channel_label='#all-lambton-college-software', event_key=None):
    """Post when explicitly enabled; suppress identical successful replay messages."""
    root = Path(project_root)
    load_dotenv(root / '.env', override=False)
    if message is None:
        return {'status': 'no_high_risk_shipments', 'sent': False}
    if os.getenv('SLACK_ENABLED', 'false').strip().lower() != 'true':
        return {'status': 'preview_only', 'sent': False,
                'detail': 'Set SLACK_ENABLED=true in private .env to send.'}
    webhook = os.getenv('SLACK_WEBHOOK_URL', '').strip()
    parts = urlsplit(webhook)
    if (parts.scheme != 'https' or parts.hostname != 'hooks.slack.com'
            or not parts.path.startswith('/services/') or parts.query or parts.fragment):
        return {'status': 'missing_or_invalid_webhook', 'sent': False,
                'detail': 'Configure a Slack Incoming Webhook privately in .env.'}
    digest = hashlib.sha256(message.encode('utf-8')).hexdigest()
    dedup_key = event_key or digest
    state_path = root / '.alert_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if dedup_key in state:
        return {'status': 'already_sent', 'sent': False, 'sent_at_utc': state[dedup_key]}
    try:
        response = requests.post(webhook, json={'text': message}, timeout=(5, 20), allow_redirects=False)
    except requests.RequestException as error:
        return {'status': 'network_error', 'sent': False,
                'error_type': type(error).__name__, 'detail': 'Webhook redacted. Check connection before retrying.'}
    if response.status_code != 200 or response.text.strip() != 'ok':
        return {'status': 'slack_rejected', 'sent': False, 'http_status': response.status_code}
    timestamp = datetime.now(timezone.utc).isoformat()
    state[dedup_key] = timestamp
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(state_path)
    receipt = {'status': 'sent', 'sent': True, 'http_status': 200,
               'slack_acknowledged': True, 'sent_at_utc': timestamp,
               'configured_channel_label': channel_label, 'message_sha256': digest,
               'event_sha256': dedup_key}
    (root / 'reports').mkdir(exist_ok=True)
    (root / 'reports/slack_delivery_receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt

def score_shipments(model_bundle, incoming):
    """Score unlabelled dispatch fields; actual_transit_hours is never consumed."""
    if 'shipment_id' not in incoming or incoming['shipment_id'].isna().any():
        raise ValueError('Every shipment must have an ID.')
    incoming = normalize_categories(incoming)
    if incoming['shipment_id'].eq('').any():
        raise ValueError('Shipment IDs cannot be blank.')
    if not incoming['shipment_id'].is_unique:
        raise ValueError('Resolve duplicate shipment IDs before scoring.')
    scored = incoming[['shipment_id', 'destination_city', 'carrier']].copy()
    scored['risk_score'] = model_bundle['model'].predict_proba(incoming)[:, 1]
    scored['high_risk'] = scored['risk_score'] >= model_bundle['threshold']
    return scored.sort_values('risk_score', ascending=False).reset_index(drop=True)
