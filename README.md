# MapleFreight delivery delay prediction and Slack alerts

AML 3303 Assessment 1 · Student ID **C0942109**

## Problem and objective

MapleFreight learns about delays too late to intervene. This project predicts `delivered_late` after pickup, prioritizes shipment review and prepares automatic Slack alerts with shipment ID, destination, carrier and risk score. Dispatch can assess rerouting or customer notification.

The supplied CSV contains historical completed shipments. The demonstration is a **historical replay**, not a live tracking feed. Operational use requires current in-transit shipment exports.

## Dataset and cleaning

The course-provided raw dataset is included unchanged at `data/maplefreight_delivery_delay_dataset.csv`: **6,035 rows, 23 columns**, 20.58% late.

- Normalize categorical whitespace and observed service-level case variants.
- Remove 34 normalized duplicates before splitting. Quarantine two conflicting rows for `SHP-101906`, leaving **5,999 unique shipments**, of which **1,234 (20.57%) are late**.
- Mask 40 weights above 50,000 kg and add a suspect-weight indicator. A factor-1,000 conversion is plausible but unverified, so no guessed conversion is applied.
- Fit numeric median imputation, missingness indicators, categorical imputation and one-hot encoding only within training folds.
- Preserve negative pickup delays (early pickup).
- Exclude `shipment_id`, `delivered_late` and **`actual_transit_hours`** from predictors. Actual transit time is unavailable before delivery.
- Preserve the supplied target despite on-time records exceeding scheduled hours. The promised-window or grace-period definition needs clarification before deployment.

The model uses all 20 dispatch predictors described in the dictionary. Features also describe cyclical month, positive pickup delay in hours, pickup delay as a fraction of scheduled time, required planned speed and suspect weight. Carrier late-history values must use events strictly before the shipment being scored; fuel cost must remain a dispatch estimate.

## Method and results

Stratified splits with seed 42: **3,599 training / 1,200 validation / 1,200 test**. IDs are disjoint across splits. All learned preprocessing and three-fold sigmoid calibration occur within training data. Models use balanced class weights. Model selection uses validation average precision (AP); the test set is reserved for final reporting.

| Candidate | Validation AP | Precision at 0.50 | Recall at 0.50 | F1 at 0.50 |
|---|---:|---:|---:|---:|
| Always on time | 0.206 | 0.000 | 0.000 | 0.000 |
| Calibrated logistic regression | **0.561** | 0.691 | 0.308 | 0.426 |
| Calibrated random forest | 0.520 | 0.625 | 0.243 | 0.350 |
| Calibrated histogram gradient boosting | 0.544 | 0.764 | 0.223 | 0.345 |

**Chosen alert threshold: 0.20.** Search 0.05–0.80 in 0.01 steps and maximize validation F2 under an **assumed 40% alert-rate ceiling**. F2 emphasizes recall with beta squared equal to four; it is not an estimated dollar-cost ratio. The capacity ceiling is a pilot assumption that dispatch must validate. The threshold and model are locked before testing. Validation recall is 76.9%, precision 40.9% and alert rate 38.8%.

| Held-out test result | Majority baseline | Selected model at 0.20 |
|---|---:|---:|
| Accuracy | 79.4% | 72.8% |
| Precision (late class) | 0.0% | **41.3%** |
| Recall (late class) | 0.0% | **76.5%** |
| F1 | 0.000 | **0.536** |
| F2 | 0.000 | 0.654 |
| Average precision | 0.206 | 0.572 |
| ROC AUC | 0.500 | 0.817 |
| Brier score (lower is better) | 0.206 | 0.123 |
| Alert rate | 0.0% | 38.2% |

Final confusion matrix, actual rows and predicted columns `[on time, late]`:

```text
                  Predicted on time   Predicted late
Actual on time            684                269
Actual late                58                189
```

Baseline matrix: `[[953, 0], [247, 0]]`. The final model detects **189 of 247 late shipments**. Of **458 alerts**, **269 are false positives**. This is a broad review queue: a flagged shipment is not necessarily more likely than not to be late. At the supplied scale of 8,000 shipments/week, the test alert rate projects about 3,053 weekly reviews if the same mix holds.

The notebook contains training-only EDA, validation permutation importance, threshold curves, precision–recall and calibration plots, confusion matrices and short findings after every section. Reports are reproducible outputs of its code.

**Key findings:** Weather, planned distance and traffic index are the strongest inputs by validation permutation importance. In training data, storm shipments are 47.2% late versus 13.8% in clear weather. Pickup delays of 31–60 minutes have a 27.6% late share versus 13.7% for early/on-time pickups. Contract owner-operators have a 36.5% late share versus 12.2% for the internal fleet; route and service mix can explain part of this association. These findings support review priorities rather than causal conclusions.

## How to run

Use **Python 3.11** and start in this repository folder.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open `MapleFreight_Delivery_Delay_Prediction_C0942109.ipynb` and choose **Restart Kernel and Run All Cells**. Save afterward so the outputs remain visible. No private credential is needed for the analysis and alert preview. Slack is disabled by default.

The notebook saves the evaluated model and threshold together as `models/delivery_delay_model.joblib`. It deliberately retains the evaluated training fit; refitting on more data changes probabilities and requires revalidation.

Replay scoring from the command line:

```powershell
python score_shipments.py --input data/replay_shipments.csv
```

## Configure and send the real Slack message

**Real delivery completed:** On **October 8, 2026 at 8:59 PM (America/Toronto)**, Slack acknowledged the historical replay alert with **HTTP 200** and `ok`. The received message is available at [the Slack message permalink](https://lambtoncolleg-y9c1808.slack.com/archives/C0C3WS3K8J3/p1791507589126649), subject to workspace access. The sanitized receipt is saved in `reports/slack_delivery_receipt.json`, and the executed notebook displays the delivery evidence.

**Screenshot evidence completed:** [`screenshots/slack_alert.png`](screenshots/slack_alert.png) is a side-by-side composite of three original captures of this same received message, assembled because the in-app Slack view was narrow. No synthetic message content was added. See [`screenshots/README.md`](screenshots/README.md) for the evidence description.

To reproduce the integration in your own workspace:

1. Sign in at [Slack app management](https://api.slack.com/apps).
2. Choose **Create New App → From scratch**, name it `MapleFreight Dispatch Alerts`, and select your workspace.
3. Select **Incoming Webhooks** and activate them.
4. Select **Add New Webhook to Workspace**, choose **#all-lambton-college-software**, and authorize posting to that channel.
5. Copy `.env.example` to `.env`. Open it in your own text editor, put the generated URL after `SLACK_WEBHOOK_URL=`, and set `SLACK_ENABLED=true`. Never paste the URL into a notebook, README or chat. The original local project is already configured privately; the public repository and ZIP contain only `.env.example`.
6. Restart the kernel and run all cells to load your new private settings and send the prepared historical replay message. A successful send requires HTTP 200 and Slack's `ok` acknowledgment. Repeat successful identical batches are suppressed.
7. Capture the actual received message with its channel name visible. Preserve unedited captures and describe any composition when updating **`screenshots/slack_alert.png`**. The text preview and delivery receipt do not replace received-message evidence.
8. Save the notebook again with the successful status visible. The local `.env` and alert state remain ignored and must not be uploaded.

An Incoming Webhook is tied to the channel chosen during setup; the code's channel label does not reroute it. Follow the [official Slack Incoming Webhooks guide](https://docs.slack.dev/messaging/sending-messages-using-incoming-webhooks/).

**Channel requirement:** The brief asks for `#dispatch-alerts`; this project follows the user's supplied `#all-lambton-college-software`. Confirm the substitution is accepted, or create `#dispatch-alerts` and connect the webhook there before submission.

Alert message preview: `reports/slack_alert_preview.txt`. The message includes total risky count and top five IDs, destinations, carriers and risk scores. Network errors report only a sanitized error type, never the URL. Fingerprints include every flagged record so a change beyond the top five can trigger a new alert.

For a real operational export, provide the same 20 dispatch columns plus `shipment_id` and `shipment_status`. Only rows with status `in_transit` are retained:

```powershell
python score_shipments.py --input active_shipments.csv --live --send
```

Run this command from the dispatch refresh process or a scheduler at the agreed feed interval. The export must contain current eligible shipments; this repository does not manufacture a live shipment source.


## Repository contents and completion status

- `MapleFreight_Delivery_Delay_Prediction_C0942109.ipynb`: notebook with executed analysis outputs.
- `data/`: unchanged raw source and an unlabelled historical replay export.
- `src/`, `score_shipments.py`: reusable preparation, inference and webhook code.
- `models/`: evaluated fitted model with locked threshold.
- `reports/`: audit, metrics, figures, split manifest, recommendations, alert preview and successful Slack delivery receipt.
- `.gitignore`, `.env.example`, `requirements.txt`: reproducibility and credential handling.
- `screenshots/slack_alert.png`: **completed received-message evidence**, comprising three genuine captures side by side; unedited originals and an explanation are included in `screenshots/`.

**Completed:** real Slack delivery on October 8, 2026 at 8:59 PM (America/Toronto), and received-message screenshot evidence. The notebook has 13 executed code cells with visible outputs and no errors.


## References

- [scikit-learn: tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html) — select the operating threshold using separate validation data.
- [Slack: sending messages using Incoming Webhooks](https://docs.slack.dev/messaging/sending-messages-using-incoming-webhooks/) — create the channel-bound webhook, send JSON and keep its URL private.
