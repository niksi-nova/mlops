# MLOps Governance, Compliance & Risk Management Framework

This document outlines the governance framework, change controls, data stewardship policies, and risk mitigation safeguards established for the Delhi Bus Travel-Time Delay Prediction system.

---

## 1. Team Roles & Responsibilities

The governance model establishes clear segregation of duties across development, verification, approval, and platform administration.

*(Proposed mapping for our team of four — **to be confirmed by Anika**)*

| Member | Assigned Governance Role | Primary Responsibilities | Authorization Authority |
|---|---|---|---|
| **Anika U. Bhat** | **Project Lead & Production Authority** | Project oversight, architectural decisions, production promotion review, final PR approvals | Production model promotion, PR merges into `main` |
| **Asrith Alapati** | **MLOps & Governance Lead** | Pipeline automation, quality gate enforcement, test isolation, version ledger, governance verification | Candidate and Staging promotions, quality gate validation |
| **Ajay** | **Data & Monitoring Engineer** | Data ingestion, DVC remote management, Pandera schemas, data drift monitoring (`docs/drift_monitoring.md`) | DVC data pushes, drift alert thresholds |
| **Niksi / Nova** | **Service & Platform Engineer** | FastAPI backend, Docker packaging, GitHub Actions CI/CD workflows, Render deployment | GHCR image releases, Render infrastructure configuration |

---

## 2. Core Governance Controls Matrix

| Control ID | Control Name | Risk It Mitigates | Why We Need It | Evidence in Repository | Status |
|---|---|---|---|---|---|
| **C-01** | **Automated Model Quality Gate** | Deployment of underperforming, overfitted, or unstable models | Prevents regressions by enforcing 8 empirical performance thresholds before staging promotion | [`src/models/quality_gate.py`](file:///e:/mlops/src/models/quality_gate.py), [`params.yaml`](file:///e:/mlops/params.yaml) | **Implemented (PASS)** |
| **C-02** | **Slice Fairness & Subgroup Evaluation** | Disproportionate performance degradation during night hours or cold-start scenarios | Protects commuter trust across diverse operating conditions; ensures model is reliable during off-peak and cold-start transitions | [`reports/quality_gate.json`](file:///e:/mlops/reports/quality_gate.json) (hourly and `has_prev_bus` checks) | **Implemented (PASS)** |
| **C-03** | **Interactive Human Approval Gate** | Accidental, rogue, or automated production model releases | Enforces strict human accountability; ensures no automated tool can deploy weights directly to production | `promote_to_production()` in [`src/models/registry.py`](file:///e:/mlops/src/models/registry.py) | **Implemented (PASS)** |
| **C-04** | **Immutable Audit Logging** | Unaudited model deployments, dispute over deployment authorization | Provides an unalterable forensic record of who authorized each production release, when, and why | [`reports/production_approvals.json`](file:///e:/mlops/reports/production_approvals.json), [`reports/model_registry.json`](file:///e:/mlops/reports/model_registry.json) | **Implemented (PASS)** |
| **C-05** | **Test Isolation from Production State** | Unit tests polluting production registry or approval files | Prevents mock test inputs from contaminating live audit trails | `isolate_approval_history` autouse fixture in [`tests/test_registry.py`](file:///e:/mlops/tests/test_registry.py) | **Implemented (PASS)** |
| **C-06** | **Data Contract & Schema Validation** | Schema drift, null features, invalid ranges breaking downstream models | Rejects malformed tabular inputs at the pipeline boundary before featurization | Pandera schema in `src/data/validate.py`, [`data/processed/feature_spec.json`](file:///e:/mlops/data/processed/feature_spec.json) | **Implemented (PASS)** |
| **C-07** | **Zero Hardcoded Secrets Policy** | Compromise of cloud registry, container registry, or deployment webhooks | Protects infrastructure credentials from public repository exposure | [`.github/workflows/ci.yml`](file:///e:/mlops/.github/workflows/ci.yml), Render environment variables | **Implemented (PASS)** |
| **C-08** | **Container Immutability via Git Commit SHAs** | Image drift, untraceable container binaries in production | Ensures exact 1-to-1 traceability between deployed cloud container and source code commit | Image tagging (`sha-<7>`) in [`.github/workflows/cd.yml`](file:///e:/mlops/.github/workflows/cd.yml) | **Implemented (PASS)** |
| **C-09** | **Live Health & Readiness Probing** | Traffic routed to unresponsive or degraded inference instances | Ensures Render deployment fails fast if model weights fail to load | `/health` endpoint in [`src/api/main.py`](file:///e:/mlops/src/api/main.py), CD health poll in [`.github/workflows/cd.yml`](file:///e:/mlops/.github/workflows/cd.yml) | **Implemented (PASS)** |
| **C-10** | **GitHub Branch Protection on `main`** | Direct pushes breaking production branch or bypassing CI checks | Guarantees all code merged to `main` undergoes peer review and passes automated tests | *Recommended to repository administrators* | **Planned (MANUAL)** |
| **C-11** | **Continuous Automated Data Drift Monitoring** | Concept drift and unmonitored accuracy decay over time | Detects when feature distributions or road speeds shift significantly, triggering retraining alerts | Built by Ajay; documented in `docs/drift_monitoring.md` | **Planned (PLANNED)** |

---

## 3. Change Control & Release Management

1. **Pull Request Workflow**:
   - Direct pushes to `main` are strictly prohibited.
   - All enhancements, fixes, and experiments must originate on topic branches (e.g. `phase2-*`, `phase3-*`).
   - Merging into `main` requires a Pull Request with passing CI tests.
2. **Branch Protection Recommendation**:
   - It is strongly recommended that the repository owner enable GitHub Branch Protection on `main`:
     - Require pull request reviews before merging.
     - Require status checks to pass (`tests`, `quality-gate`, `docker-smoke`).
     - Require linear history; disallow force pushes and branch deletions.
3. **CI/CD Quality Gates**:
   - Every push and PR automatically executes the full test suite (`uv run pytest -q`) and DVC integrity verification.
   - Merges to `main` trigger Docker image building and deployment to Render web service.

---

## 4. Data Governance & Ethical Stewardship

### 4.1 Data Origin and Lineage
* **Source**: Delhi public bus transit GPS tracking dataset, covering route segments across routes 425 and 534 over a 3-month operational window.
* **Licensing**: The upstream raw dataset was collected from municipal transit feeds and does not carry a formal open-source license. Use is restricted to academic evaluation, educational benchmarking, and non-commercial transit research.
* **Storage & Versioning**: Raw and processed datasets are tracked via DVC (`delhi-data-v2`) and mirrored on DagsHub cloud storage.

### 4.2 Privacy & PII Assessment
* **No Personally Identifiable Information (PII)**: The dataset contains zero passenger counts, payment records, bus driver names, or vehicle license plates.
* **No Exact Coordinates**: Latitude and longitude GPS coordinates were stripped during segment extraction; positions are represented strictly as abstract network topology nodes (`from_node`, `nodes_spanned`, `segment_id`).

### 4.3 Data Quality & Schema Enforcement
* Schema contracts are defined in `data/processed/feature_spec.json`.
* Every featurization run validates column names, numerical ranges, categorical cardinality, and missing value thresholds.

---

## 5. Model Quality Gate & Slice Fairness

The automated model quality gate (`src/models/quality_gate.py`) enforces 8 mandatory thresholds defined in `params.yaml`:

1. **Improvement over Typical Baseline**: $\ge 10.0\%$ (Achieved: **25.88%**)
2. **Improvement over Recent Buses Baseline**: $\ge 10.0\%$ (Achieved: **19.93%**)
3. **Maximum Test MAE**: $\le 35.0$ seconds (Achieved: **30.60 s**)
4. **Generalization Gap**: $\le 10.0\%$ (Achieved: **4.32%**)
5. **Within-60s Prediction Accuracy**: $\ge 85.0\%$ (Achieved: **89.35%**)
6. **Model Artifact File Size**: $\le 100.0$ MB (Achieved: **66.80 MB**)
7. **Single-Row Warm Inference Latency**: $\le 50.0$ ms (Achieved: **15.17 ms**)
8. **Prediction Sanity Range**: Finite, positive, bounded in $[1, 1800]$ seconds (Achieved: **All valid**)

### Slice Fairness Checks
* **Hourly Performance**: Night-time vs peak daytime performance must not diverge by more than a 1.8x ratio. Low-volume slices ($N < 100$, such as Hour 4 with $N=1$) are explicitly isolated as `INSUFFICIENT_DATA`.
* **Cold-Start Resilience**: Segments without a preceding bus (`has_prev_bus=0`) are evaluated independently; the ratio to normal segments remains within the 1.8 threshold (Achieved: 1.67).

---

## 6. Secrets Management & Credential Safety

* **Zero Hardcoded Secrets**: All sensitive tokens are managed exclusively through environment variables and cloud secret vaults:
  - `DAGSHUB_USER`, `DAGSHUB_TOKEN`: Used in CI to authenticate DVC data pulls.
  - `WANDB_API_KEY`: Injected into runtime containers for model registry lookups.
  - `RENDER_DEPLOY_HOOK`: GitHub Actions secret triggering cloud deployments.
* **Key Rotation Precedent**: Earlier in development, an inadvertently committed API token was immediately revoked, rotated, and purged from repository configuration. Automated secrets scanners (`scripts/check_governance.py`) run as part of the governance audit.

---

## 7. Incident Handling & Emergency Rollback Procedure

In the event of model degradation, upstream traffic anomalies, or service degradation:

```
[Incident Detected: Elevated MAE or /health Failure]
                       │
                       ▼
         [Step 1: Diagnostic Assessment]
  Check /health, /model-info, Render application logs
                       │
                       ▼
         [Step 2: Decision on Rollback Mode]
      ┌────────────────┴────────────────┐
      ▼                                 ▼
[Registry Rollback]              [Container Rollback]
Reassign 'production' alias       Redeploy previous GHCR tag
in W&B to previous version        ghcr.io/niksi-nova/mlops:sha-<old>
(No image rebuild needed)         (Render deploy hook trigger)
                       │
                       ▼
         [Step 3: Post-Rollback Verification]
  Verify /health returns 200, test sample predictions
                       │
                       ▼
         [Step 4: Post-Mortem & Incident Log]
  Document root cause, update model risk report
```

---

## 8. Audit Trail & Evidence Retention

| Artifact Type | Storage Location | Retention Policy | Audit Verification Command |
|---|---|---|---|
| **Code Changes** | GitHub Repository | Permanent Git History | `git log --oneline` |
| **Datasets** | DagsHub DVC Remote | Project Lifecycle | `uv run dvc status` |
| **Model Weights** | W&B Model Registry | Permanent Versioned | `uv run python scripts/build_version_ledger.py` |
| **Container Images**| GitHub Packages (GHCR) | Last 30 Image Builds | GitHub Packages UI / Docker pull |
| **Production Approvals** | `reports/production_approvals.json` | Permanent Repository Log | `cat reports/production_approvals.json` |
| **Governance Reports** | `reports/governance_status.json` | Generated on CI/Audit | `uv run python scripts/check_governance.py` |

---

## 9. Model Risk Register Summary

For complete risk quantification and technical mitigations, see [`docs/model_risk_report.md`](file:///e:/mlops/docs/model_risk_report.md).

| Risk ID | Category | Severity | Likelihood | Impact | Key Technical Mitigation |
|---|---|---|---|---|---|
| **R-01** | **Data Drift** | High | High | High | Daily rolling MAE monitoring; automated retraining trigger at MAE $> 35.0$s |
| **R-02** | **Target Leakage** | Critical | Low | Critical | Strict temporal boundary: `own_prev_segment_tt_s` precedes segment entry |
| **R-03** | **Long Delay Tail** | High | Medium | High | Bounded prediction sanity checks $[1, 1800]$s; L1 objective robust to outliers |
| **R-04** | **Route & Hour Bias** | High | High | Medium | Slice-based quality gate checks (`max_slice_mae_ratio <= 1.8`) |
| **R-05** | **Unauthorized Release** | Critical | Low | Critical | Interactive human terminal confirmation required for production promotion |

---

## 10. Alignment with AI Risk Management Standards (NIST AI RMF)

The platform implements controls mapped directly to the four core functions of the **NIST AI Risk Management Framework (AI RMF 1.0)**:

1. **Govern**: Established segregated roles, change control procedures, approval policies, and ethics guidelines.
2. **Map**: Categorized risks in `docs/model_risk_report.md` spanning data leakage, traffic drift, and route imbalances.
3. **Measure**: Implemented 8 automated empirical quality gate metrics and slice evaluations in `src/models/quality_gate.py`.
4. **Manage**: Integrated dynamic registry rollback, containerized image immutability, and monitoring hooks to mitigate risks post-deployment.
