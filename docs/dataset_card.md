# Dataset Card: TTC Bus Delay Data

## 1. Overview
| Item | Details |
|---|---|
| Dataset name | TTC Bus Delay Data |
| Publisher | Toronto Transit Commission (TTC), via the City of Toronto Open Data Portal |
| Source | https://open.toronto.ca (dataset id: `ttc-bus-delay-data`) |
| Licence | Open Government Licence – Toronto (free to use with attribution) |
| Update frequency | Monthly (rolling file for 2025 onward) |
| Personal data | None. No passenger or operator identities are included |

## 2. Purpose in this project
This dataset is used to train a model that predicts the **delay duration in minutes**
of a TTC bus disruption, from route, time, location, direction, incident type and
weather. The prediction supports a passenger information system that shows riders
an expected delay instead of just "delayed".

## 3. Data acquired
| Part | Coverage | Format | Use |
|---|---|---|---|
| Historical delay files | 2018–2024 | One XLSX per year, 12 monthly sheets each | Training, validation and test |
| Recent delay file | January 2025 onward | One rolling CSV | Simulated production data (Phase 3) |
| Code Descriptions | 46 incident codes | CSV | Lookup for incident types |
| Weather (enrichment) | Hourly, 2018–2024 | Open-Meteo historical archive API | Weather features |
| Holidays (enrichment) | 2018–2024 | Python `holidays` library | Holiday features |

## 4. Columns (after cleaning)
| Column | Description |
|---|---|
| timestamp | Date and time of the incident (Toronto local time) |
| route | Bus route number |
| day | Day of the week |
| location | Where the incident occurred |
| incident | Type of incident (e.g. mechanical, diversion, collision) |
| min_delay | **Target**: delay to the following bus, in minutes |
| min_gap | Gap to the bus ahead, in minutes. **Excluded from features (data leakage)** |
| direction | Direction of travel: N, S, E, W, B (both ways) or UNKNOWN |
| vehicle | Vehicle number |

## 5. Size
| Stage | Rows |
|---|---|
| Raw (all monthly sheets, 2018–2024) | 389,280 |
| After cleaning | 382,131 (1.8% removed) |

Rows per year after cleaning:

| Year | Rows |
|---|---|
| 2018 | 72,417 |
| 2019 | 61,471 |
| 2020 | 35,633 |
| 2021 | 41,645 |
| 2022 | 57,575 |
| 2023 | 55,070 |
| 2024 | 58,320 |

The drop in 2020 reflects reduced service during the COVID-19 pandemic.

## 6. Data quality issues found and handled
- Column names differ between years ("Report Date" vs "Date", "Delay" vs "Min Delay",
"Gap" vs "Min Gap"). These were merged into consistent columns.
- Some 2018 sheets have a duplicate column with a leading space (" Min Delay").
- Two 2021 monthly sheets use "Line"/"Bound" instead of "Route"/"Direction".
- Time values use mixed formats ("02:08" and "00:17:00"). All were standardised.
- Delays above 300 minutes were removed as likely recording errors.
- Rows with missing route, delay or timestamp, and exact duplicates, were removed.
- About 18% of rows have an unclear direction (free text or typos in the source).
These are kept as the category "UNKNOWN".

## 7. Data split plan (time-based, not random)
| Period | Purpose |
|---|---|
| 2018–2022 | Training |
| 2023 | Validation |
| 2024 | Test |
| 2025 onward | Simulated production stream for drift monitoring |

A time-based split prevents the model from learning from future incidents.

## 8. Versioning
Raw data is versioned with DVC and stored on a DagsHub remote.
- `data-v1`: historical data 2018–2024
- `data-v2`: adds the 2025+ rolling file

## 9. Known limitations
- Only incidents are recorded, not every trip. The model predicts how long a delay
lasts once an incident happens, not whether a trip will be late.
- Data is entered manually by operators, so some labels may be noisy.
- Weather is city-level, not stop-level.
- The data covers Toronto only; a different city would require retraining.

## 10. Why this dataset was chosen
- Real operational data from Canada's largest transit system.
- Official source with a clear open licence.
- Numeric target, suitable for regression.
- Multiple years of data, allowing natural drift to be studied.
- Updated monthly, so real future data can simulate production traffic.
- Contains no personal data.