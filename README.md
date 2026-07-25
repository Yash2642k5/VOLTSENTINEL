# VoltSentinel

**AI-Powered EV Battery Asset Performance & Security Intelligence Agent**

ET AI Hackathon 2026 — Problem Statement 3: _AI for Industrial EV Supply Chain & Asset Intelligence: Accelerating Net Zero_

> Team Name: `btech10364.23` · Team Members: `Yash Sinha`

---

# Live Link [VOLTSENTINEL](https://voltsentinel.streamlit.app/)

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement & Context](#problem-statement--context)
3. [Objectives](#objectives)
4. [Proposed Solution](#proposed-solution)
5. [System Architecture](#system-architecture)
6. [Data Flow](#data-flow)
7. [Agent Decision Layer](#agent-decision-layer)
8. [Security Enforcement — Quarantine Circuit Breaker](#security-enforcement--quarantine-circuit-breaker)
9. [Fleet BI Chat](#fleet-bi-chat)
10. [Anomaly & Health Detection Logic](#anomaly--health-detection-logic)
11. [Recently Shipped Roadmap Features](#recently-shipped-roadmap-features)
12. [Data Simulation Approach](#data-simulation-approach)
13. [Configuration](#configuration)
14. [Alignment with Judging Criteria](#alignment-with-judging-criteria)
15. [Expected Deliverables](#expected-deliverables)
16. [Future Roadmap](#future-roadmap)
17. [Team](#team)
18. [Conclusion](#conclusion)

---

## Executive Summary

VoltSentinel is an AI-powered **Asset Performance Management (APM) agent** for industrial and commercial EV fleets. It monitors battery state-of-health, charging behaviour, and thermal conditions across a fleet; predicts **Remaining Useful Life (RUL)** per asset; recommends charge-discharge policies to extend battery life; and generates predictive maintenance triggers — the core capabilities of an APM agent, applied to EV batteries the way APM is applied to industrial machinery.

It extends this with a **security-aware reasoning layer** that treats unauthorized battery management system (BMS) control commands as a distinct stress event on the asset — alongside degradation and thermal signals, not separate from them. This closes a real and currently unresolved vulnerability affecting low-cost EV battery packs across India.

A **decision-making agent layer** sits above the scoring models: it takes the combined health, thermal, charging-behaviour, and security signals for an asset, reasons over them together, and emits concrete actions — a maintenance trigger, a charge-policy recommendation, or an incident escalation — rather than only displaying a score. This is what makes VoltSentinel an _agent_, in line with the PS3 track, rather than a monitoring dashboard.

The project is anchored to a live, ongoing regulatory and safety gap — the **"Tirri Challenge"** incidents of mid-2026 — giving the solution concrete, verifiable, real-world grounding.

Since the original hackathon submission, four items from the project's own [Future Roadmap](#future-roadmap) — **Driver Identity & Vehicle Assignment**, **Live SoC / Range**, **Driver-Level Coaching**, and **Weather-Aware Range** — have moved from "planned" to shipped. See [Recently Shipped Roadmap Features](#recently-shipped-roadmap-features) for what changed.

---

## Problem Statement & Context

### Industry Context

India's industrial and commercial EV segment remains under-penetrated, and the primary barrier is no longer financial — it is **operational**. Fleet operators lack asset intelligence tools to manage EV battery lifecycle, maintenance, and charging infrastructure with the same rigour applied to conventional equipment. This is compounded by a supply chain built largely on low-cost, imported battery packs with inconsistent quality and security standards.

### The Security Gap: The "Tirri Challenge"

In mid-2026, a viral trend in India — dubbed the "Tirri Challenge" — saw pranksters use Chinese battery management companion apps (including BAT-BMS, Lossigy, and Epoch i-ion) to remotely disable moving e-rickshaws. The apps connected over **unauthenticated Bluetooth** to the vehicle's BMS and triggered its discharge cut-off, stopping the vehicle in live traffic.

The root cause was the **absence of authentication** — open Bluetooth access, default or no credentials, and no verification of who was issuing administrative commands. The government's response was reactive (app removal); it did not fix the underlying hardware/firmware vulnerability across the installed base. No mandatory security standard has been announced, and any similarly-capable app could reproduce the exploit.

From an APM standpoint, each unauthorized command is also an **unplanned, non-ticketed stress event** on the battery — functionally comparable to a thermal event or an abusive discharge cycle. VoltSentinel treats it as such: a health-relevant signal the agent reasons over, not a bolt-on security feature.

---

## Objectives

- Monitor battery state-of-health, charging-cycle patterns, and thermal events across a simulated EV fleet, in line with standard APM practice.
- Predict Remaining Useful Life (RUL) and degradation trajectory per asset.
- Generate predictive maintenance triggers and optimal charge-discharge recommendations per asset, not just a health score.
- Detect BMS control commands that show signs of unauthorized use, and reason over them as a health-stress signal alongside thermal and charging data.
- Distinguish legitimate maintenance shutdowns from suspected "Tirri Challenge"-style attacks in real time, using explainable, auditable logic.
- Act as an **agent**: perceive combined asset signals, reason over them, and decide/emit concrete actions — not only classify and display.
- Present health, prescriptive, and security signals in a single, unified operator view.
- Demonstrate the concept end-to-end using simulated data, given no access to real fleet or BMS telemetry.

---

## Proposed Solution

### Core Concept

VoltSentinel ingests battery telemetry and BMS command events for a simulated fleet, and routes them through analytical layers built on a shared data pipeline: a health/degradation layer (RUL), a charging-behaviour layer, a thermal-event layer, and a security layer that scores each control command against contextual signals — matching maintenance ticket, GPS location consistency, and command frequency. These merge into a single per-asset risk profile.

An **agent decision layer** then reasons over that merged profile — the way a human asset manager would weigh multiple signals together — and emits the outputs an APM agent is expected to produce: predictive maintenance triggers, charge-discharge recommendations, and, where warranted, a security escalation.

### Key Differentiator: Not Another BMS App

| Aspect                   | BMS Companion App (e.g. BAT-BMS)  | VoltSentinel                                                                               |
| ------------------------ | ---------------------------------- | -------------------------------------------------------------------------------------------- |
| User                     | Individual driver / vehicle owner  | Fleet manager / operator                                                                     |
| Scope                    | One vehicle at a time              | Entire fleet, aggregated                                                                     |
| Relationship to commands | Sends commands to the BMS          | Observes and evaluates commands sent to the BMS                                              |
| Security awareness       | None — accepts any nearby command  | Treats unauthorized commands as a health-stress signal within the agent's reasoning          |
| Predictive capability    | None — current readings only       | RUL forecasting, thermal-event detection, charging-pattern analysis, live SoC/range           |
| Prescriptive capability  | None                               | Charge-discharge recommendations and predictive maintenance triggers, emitted by the agent   |
| Agentic behaviour        | None — passive control tool        | Perceives multi-signal state, reasons, and emits/recommends actions                          |

VoltSentinel does not attempt to fix the underlying BMS authentication gap — that is a manufacturer and regulatory responsibility. Instead, it provides a **deployable APM agent** for fleet operators who cannot wait for every unit in the field to receive a firmware fix.

---

## System Architecture

The system follows a single linear data pipeline that fans out into parallel analytical layers, converges into a merged per-asset risk profile, and is then reasoned over by an agent decision layer before reaching the dashboard.

```mermaid
flowchart TB
    subgraph SIM["Data Simulation Layer"]
        A["Python · numpy / pandas<br/>Synthetic telemetry + BMS command generator<br/>(capacity, voltage, temp, SoC, charge cycles)"]
        A2["Attack Injection<br/>(no-ticket disable, GPS mismatch, freq spikes)"]
    end

    subgraph ING["Ingestion Layer"]
        B["FastAPI · REST / WebSocket<br/>Receives & normalizes telemetry + command events<br/>(future: real BLE-connected BMS hardware)"]
    end

    subgraph STORE["Storage Layer"]
        C[("SQLite<br/>Telemetry history · Maintenance tickets · Command event log<br/>Driver pool & vehicle assignments · Quarantine / rejected-command state")]
    end

    subgraph ANALYTICS["Parallel Analytical Layers"]
        D["Predictive Analytics<br/>scikit-learn regression<br/>Capacity-fade curve → RUL"]
        E["Charging Behaviour Analytics<br/>Fast-charge freq · DoD habits · charge-rate stress<br/>(per vehicle AND per driver)"]
        F["Thermal Event Detection<br/>Overheating & abnormal thermal trend flags"]
        G["Security Analytics<br/>Rule-based + IsolationForest<br/>Ticket match · GPS consistency · command frequency"]
        RG["Live Range Estimation<br/>SoC × capacity → estimated range<br/>weather-adjusted kWh/km"]
    end

    H{{"Merged Per-Asset<br/>Risk Profile"}}

    subgraph AGENT["Agent Decision Layer"]
        I["Perceive → Reason → Decide → Act<br/>agent/decision_engine.py"]
    end

    subgraph ACTIONS["Agent Actions — agent/actions.py (mixed maturity, see tiers below)"]
        J1["create_maintenance_ticket() · Tier 1 (real email)"]
        J2["recommend_charge_policy() · Tier 2 (mocked)"]
        J3["escalate_incident() · Tier 2 (mocked)"]
        J4["notify_fleet_manager() · Tier 1 (real email)"]
        J5["quarantine_vehicle() · Tier 3 (real enforcement)"]
    end

    subgraph BICHAT["Fleet BI Chat (agent/bi_chat_engine.py)"]
        L["Read-only tool-calling loop<br/>+ optional scoped web_search"]
    end

    K["Streamlit Dashboard<br/>Fleet Overview (driver + live range columns) · Fleet BI chat · health/thermal/charging charts<br/>Agent recommendations (preview → Execute & log) · alert feed · Driver Scorecard · 'Simulate Attack'"]

    A --> B
    A2 --> B
    B --> C
    C --> D
    C --> E
    C --> F
    C --> G
    C --> RG
    D --> H
    E --> H
    F --> H
    G --> H
    H --> I
    I --> J1
    I --> J2
    I --> J3
    I --> J4
    I --> J5
    J1 --> K
    J2 --> K
    J3 --> K
    J4 --> K
    J5 --> K
    J5 -.->|"tightens future unticketed<br/>commands at ingestion"| B
    C -.-> L
    L --> K
    D -.-> K
    E -.-> K
    F -.-> K
    G -.-> K
    RG -.-> K
```

### Component Overview

| Layer                                    | Technology & Role                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Data Simulation**                      | Python (numpy, pandas) — generates synthetic battery telemetry (capacity, voltage, temperature, SoC, charge-cycle behaviour) for the fleet and injects both normal maintenance events and unauthorized-command "attack" events. Also generates a mock driver pool and per-vehicle shift assignments (`simulator/driver_generator.py`).                            |
| **Ingestion**                            | FastAPI (REST / WebSocket) — receives and normalizes incoming telemetry and command events; the boundary where real BLE-connected BMS hardware would plug in. Also the enforcement point for the quarantine circuit breaker (below): an unticketed command for a currently-quarantined vehicle is rejected outright here, not merely flagged after the fact.       |
| **Storage**                              | SQLite — stores telemetry history, mock maintenance ticket records, the command event log, agent action audit trail, quarantine / rejected-command state, and (now) driver identity + vehicle-assignment records.                                                                                                                                                 |
| **Predictive Analytics**                 | scikit-learn (regression) — fits a capacity-fade curve per asset and extrapolates Remaining Useful Life (RUL).                                                                                                                                                                                                                                                     |
| **Live Range Estimation** _[NEW]_        | `models/range_estimator.py` — same-day "can this vehicle make it back right now?" signal from each vehicle's latest SoC and current capacity, adjusted for live ambient temperature per region (`models/weather_client.py`, Open-Meteo, no API key required).                                                                                                     |
| **Charging Behaviour Analytics**         | Analyzes charge-cycle patterns per asset — fast-charge frequency, depth-of-discharge habits, charge-rate stress — as an input to both RUL and the agent's charge-policy recommendations. Now also re-aggregable **per driver** (`analyze_fleet_drivers`), on top of Driver Identity & Vehicle Assignment.                                                          |
| **Thermal Event Detection**              | Flags overheating and abnormal thermal trends from telemetry as a distinct health-anomaly signal, separate from command/security anomalies.                                                                                                                                                                                                                        |
| **Security Analytics**                   | Rule-based logic, optionally supplemented with scikit-learn IsolationForest — scores each command against maintenance-ticket match, GPS consistency, and command frequency.                                                                                                                                                                                        |
| **Agent Decision Layer**                 | Perceive → reason → decide → act loop over the merged RUL + thermal + charging + security profile per asset. Emits predictive maintenance triggers, charge-discharge recommendations, security escalations, and real quarantine enforcement. Every proposed action is previewed before a human explicitly commits it.                                             |
| **Fleet BI Chat**                        | A read-only, tool-calling chat surface (`agent/bi_chat_engine.py`) — the fleet manager asks plain-English comparison/ranking/trend questions and gets back a short answer plus a chart built dynamically from whatever the agent's own tool calls returned. Optional scoped `web_search` tool, off by default. Has zero access to `agent/actions.py`'s write path. |
| **Presentation**                         | Streamlit — unified fleet dashboard: Fleet Overview (now with a Driver column and live SoC/range signal), Fleet BI chat, Asset Detail (now with weather-aware live range), Agent (live reasoning + history), Alert Feed, and Driver Scorecard tabs, plus a live "Simulate Attack" trigger for demonstration.                                                       |

---

## Data Flow

End-to-end journey of a single telemetry/command event through the system:

```mermaid
sequenceDiagram
    participant Sim as Data Simulator
    participant API as FastAPI Ingestion
    participant DB as SQLite Storage
    participant RUL as RUL Model
    participant Chg as Charging Analytics
    participant Therm as Thermal Detection
    participant Sec as Security Analytics
    participant Agent as Agent Decision Layer
    participant UI as Streamlit Dashboard

    Sim->>API: Telemetry event (capacity, voltage, temp, SoC)
    Sim->>API: BMS command event (e.g. discharge cut-off)
    alt Vehicle is quarantined AND command has no maintenance ticket
        API->>DB: Reject command outright, log to rejected_commands
    else Not quarantined, or command is ticketed
        API->>DB: Normalize & persist event
    end

    DB->>RUL: Capacity/voltage history
    DB->>Chg: Charge-cycle history
    DB->>Therm: Temperature history
    DB->>Sec: Command + ticket + GPS history

    RUL-->>Agent: RUL trend
    Chg-->>Agent: Charging-pattern stress score
    Therm-->>Agent: Thermal-anomaly flags
    Sec-->>Agent: Security/command-anomaly flags

    Note over Agent: Perceive → Reason → Decide → Act

    Agent->>Agent: Weigh signals jointly
    Agent->>UI: Propose actions (preview only — nothing written yet)
    UI->>UI: Fleet manager reviews the preview
    UI->>Agent: "Execute & log" clicked (or "Discard")
    Agent->>DB: create_maintenance_ticket() [if RUL/thermal threshold crossed] — real email if SMTP configured
    Agent->>DB: recommend_charge_policy() [if charging stress detected] — still mocked (Tier 2)
    Agent->>DB: escalate_incident() [if unticketed + GPS mismatch/freq spike]
    Agent->>DB: quarantine_vehicle() [if security severity=high, repeated/escalating pattern] — real enforcement (Tier 3)
    Agent->>DB: notify_fleet_manager() [if severity above threshold] — real email if SMTP configured

    UI-->>UI: Render per-asset health, charts, alerts, recommendations, live SoC/range
```

---

## Agent Decision Layer

This is the component that makes VoltSentinel an **agent** rather than a scoring dashboard, and it is the direct answer to the APM Agent brief's call for predictive maintenance triggers and charge-discharge recommendations, not just RUL numbers.

**Human-in-the-loop by design:** clicking "Run agent reasoning" for an asset only _previews_ the LLM's proposed actions — nothing is written yet. The fleet manager reviews the rationale for each proposed action and must explicitly click **"Execute & log"** to commit them (or "Discard" to drop the preview). This preview-vs-commit pattern is deliberate: it means the agent's own written LLM call (`agent/decision_engine.py`) is never itself the thing that touches the database — a human decision always sits between "the model recommended X" and "X actually happened," which matters more now that some actions (below) have real-world effects rather than only being logged.

### Perceive → Reason → Decide → Act

```mermaid
flowchart LR
    P["👁️ PERCEIVE<br/>Read merged risk profile per asset:<br/>RUL trend · thermal-anomaly flags ·<br/>charging-pattern stress · security flags"]
    R["🧠 REASON<br/>Weigh signals together, not independently<br/>e.g. moderate RUL decline + frequent<br/>fast-charging + unticketed disable event<br/>≠ same RUL decline alone"]
    D["⚖️ DECIDE<br/>Determine warranted action(s) & priority<br/>with explicit, auditable rationale<br/>tied to observed signals"]
    Pv["👤 PREVIEW<br/>Fleet manager reviews proposed<br/>action(s) — nothing written yet"]
    Ac["⚡ ACT<br/>'Execute & log' commits via<br/>agent/actions.py (mixed real/mocked, see tiers below)"]

    P --> R --> D --> Pv --> Ac
    Ac -.->|feedback loop: next cycle| P
```

### Emitted Actions

| Action                              | Trigger Condition (example)                                                                  | Output                                                                                                                                                                                                                                                                              |
| ------------------------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Predictive maintenance trigger**  | RUL crosses a defined threshold, or a thermal anomaly persists across cycles                   | `create_maintenance_ticket()` — logs the ticket and sends a real email if SMTP is configured                                                                                                                                                                                        |
| **Charge-discharge recommendation** | Charging-pattern analysis shows high fast-charge frequency or excessive depth-of-discharge     | `recommend_charge_policy()` — e.g. cap charge rate, limit DoD to extend RUL (still mocked)                                                                                                                                                                                          |
| **Security escalation**             | Unticketed command + GPS mismatch and/or frequency spike                                       | `escalate_incident()` — flags for fleet-manager review, distinct from routine maintenance (still mocked)                                                                                                                                                                            |
| **Vehicle quarantine** _[NEW]_      | Security severity = high **and** a repeated/escalating pattern (not a single isolated event)   | `quarantine_vehicle()` — **real enforcement**: every further unticketed BMS command for that asset is rejected outright at ingestion from this point on. Never disables or cuts off a moving vehicle directly — it only tightens which future unauthenticated commands are honored. |
| **Fleet manager notification**      | Any high-priority decision above a severity threshold                                          | `notify_fleet_manager()` — logs the notification and sends a real email if SMTP is configured                                                                                                                                                                                       |

### Action Maturity Tiers

Not every action carries the same real-world weight, so `agent/actions.py` groups them explicitly:

| Tier                                                     | Actions                                             | Behaviour                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tier 1 — fully automated, no physical/financial risk** | `create_maintenance_ticket`, `notify_fleet_manager` | Sends a real email via SMTP when `SMTP_HOST` + `ALERT_EMAIL_TO` are configured (see [Configuration](#configuration)). Degrades gracefully to logged-only if SMTP isn't set up — nothing breaks for a demo/dev environment.                                                                                                                                                                                                                                                                  |
| **Tier 2 — mocked, human-approval gated**                | `recommend_charge_policy`, `escalate_incident`      | Logged only. Deliberately kept mocked: changing actual charge behaviour (a real OCPP/BMS write) or escalating to an on-call system should stay behind an explicit human "Approve" step before ever wiring to a live integration.                                                                                                                                                                                                                                                            |
| **Tier 3 — real enforcement (the circuit breaker)**      | `quarantine_vehicle`                                | **Not mocked.** Flips `vehicle_quarantine.active` for the vehicle; enforced at the ingestion boundary (see [Security Enforcement](#security-enforcement--quarantine-circuit-breaker)). The LLM can impose a quarantine but can never lift one — `release_vehicle_quarantine` is deliberately absent from both `agent/prompts.py`'s action vocabulary and `agent/actions.py`'s dispatcher, reachable only from a dashboard control that requires an explicit human name for the audit trail. |

Every decision — including an explicit `no_action` when nothing is warranted — is logged to the `agent_actions` audit table, so the fleet view can always show "the agent reviewed this asset and found X," not just silence.

### Scoped Per-Asset Research (optional)

A separate, narrow web-search capability sits on the Agent tab, distinct from the main decision loop and from the Fleet BI chat below:

- The main Perceive→Reason→Decide loop (`decide_for_asset` / `decide_for_fleet`) stays completely network-free — every emitted action always traces back only to an already-computed `risk_engine.py` signal, never to something pulled from the open web.
- `DecisionEngine.research_asset_question()` is a **separate** client (`GeminiSearchClient`), reachable only through this one method, and disabled by default (`enable_research=False`). It answers one free-text question about **one specific asset's already-computed profile**, and only within a fixed, narrow topic allowlist: replacement vehicle/battery-pack models suited to that asset's profile, charge-policy best practices tied to signals already present, manufacturer/part-sourcing context tied to a logged maintenance trigger, or regulatory/safety context tied to a logged security escalation.
- Anything outside that list — general knowledge, unrelated topics, requests to act on other systems — the model is instructed to explicitly refuse (`in_scope: false` with a `refusal_reason`), and the engine trusts and passes that refusal straight through rather than second-guessing it.

---

## Security Enforcement — Quarantine Circuit Breaker

Detection (flagging a suspicious command) and enforcement (actually changing what a vehicle will accept) are kept as two separate, explicit steps — matching the same detect-vs-decide separation used throughout the codebase. This section covers the enforcement half, introduced alongside the agent's Tier 3 `quarantine_vehicle` action above.

```mermaid
flowchart TB
    Q1["Agent decides quarantine_vehicle<br/>is warranted (repeated/escalating<br/>security pattern) → human clicks<br/>'Execute & log'"]
    Q2[("vehicle_quarantine table<br/>active = 1, reason, action_id,<br/>quarantined_at")]
    Q3{"Next incoming BMS command<br/>for this vehicle"}
    Q4["Has a matching<br/>maintenance ticket?"]
    Q5["✅ Accepted into `commands`<br/>(a real technician visit<br/>always still goes through)"]
    Q6["❌ Rejected outright<br/>logged to `rejected_commands`<br/>(never even reaches `commands`)"]
    Q7["👤 Human-only release<br/>(dashboard button, requires a name)<br/>release_vehicle_quarantine()"]

    Q1 --> Q2
    Q2 --> Q3
    Q3 --> Q4
    Q4 -->|Yes| Q5
    Q4 -->|No, and vehicle quarantined| Q6
    Q2 -.-> Q7
    Q7 -.->|active = 0| Q2
```

Key properties:

- **Real enforcement, not another log line.** `ingestion/db.py`'s `insert_command` / `insert_command_batch` check `vehicle_quarantine.active` on every incoming command; an unticketed command for a quarantined vehicle is rejected at the ingestion boundary itself and recorded in a dedicated `rejected_commands` audit table — it never lands in the `commands` table the analytics layers read from.
- **A ticketed command always still goes through**, quarantine or not — this is what keeps a real technician visit from being blocked by the same circuit breaker meant to stop unauthorized remote control.
- **Never touches a moving vehicle directly.** Quarantine doesn't issue a discharge-cutoff or disable command itself — the exact mechanism the "Tirri Challenge" exploited. It only tightens which _future_ unauthenticated commands are honored, deliberately avoiding recreating that same failure mode (a remote party autonomously affecting a moving vehicle) in the other direction.
- **Asymmetric by design.** The LLM can impose a quarantine (`quarantine_vehicle` is a valid action the model can choose), but it can never lift one. `release_vehicle_quarantine` is intentionally absent from both `agent/prompts.py`'s action vocabulary and `agent/actions.py`'s dispatch table — the only legitimate caller in the codebase is a dashboard control that already requires an explicit human click and a name, recorded for the audit trail (`released_by`).

---

## Fleet BI Chat

Alongside the per-asset Agent tab, the dashboard has a second, fleet-wide conversational surface: **"ask the fleet in plain English."**

- A few always-on default charts (fleet risk mix, RUL distribution, charge-stress vs. thermal scatter) sit above a chat box wired to `agent/bi_chat_engine.py`'s tool-calling loop.
- The fleet manager can ask comparison, ranking, or trend questions in plain English (e.g. _"compare EVR-0001 and EVR-0007's charging stress"_ or _"which vehicles are lowest on RUL?"_) and get back a short text answer plus — for comparative/trend questions — a chart. The chart type and fields come from the model's own dynamic chart spec, not a fixed chart per query type.
- **Strictly read-only.** This engine has zero access to `agent/actions.py`'s write path — by construction, not just by prompt instruction — so a question asked here can never accidentally trigger a maintenance ticket, a quarantine, or any other side effect. The Agent tab's write path is a deliberately separate engine and tool registry.
- **Optional web search.** A `web_search` tool (backed by a separate Gemini client with Google Search grounding) can be opted into per session, for questions the fleet database genuinely can't answer (e.g. _"what EV models should replace EVR-0012?"_). It's off by default (`enable_web_search=False`); the model is instructed to always prefer the fleet-data tools first and only fall back to the web when a question truly needs outside information.
- Conversation history is threaded per session, so follow-up questions like _"what about EVR-0002 too?"_ work, while every turn still re-queries live data rather than a stale snapshot from when the conversation started.

---

## Anomaly & Health Detection Logic

### Security / Command Anomalies

A BMS control command (e.g. discharge cut-off, disable) is flagged as suspicious when it exhibits one or more of the following:

- No matching maintenance ticket exists for the command at the time it was issued.
- The vehicle's GPS location at the time of the command is inconsistent with a known depot or service location (e.g. the vehicle is in motion on a public road).
- The frequency of control commands for that asset spikes well above its historical baseline within a short time window.

Commands matching a valid maintenance ticket and an expected service location are treated as legitimate and excluded from alerting.

### Thermal Event Detection

Temperature telemetry is analyzed as its own signal, independent of command anomalies:

- Sustained temperature readings above asset-specific safe operating thresholds.
- Abnormal rate-of-change in temperature relative to charge/discharge state.
- Repeated thermal flags correlated with specific charge-cycle behaviour, feeding both the RUL model and the agent's charge-policy recommendations.

### Charging Pattern Analysis

Charging behaviour is analyzed as a distinct signal contributing to both degradation modeling and prescriptive output:

- Fast-charge frequency relative to fleet baseline.
- Depth-of-discharge (DoD) habits per asset.
- Charge-rate stress trends over time, used to generate charge-discharge recommendations via the agent layer.

This keeps all detection and recommendation logic explainable and auditable — an important property for a fleet-operator-facing safety and asset-management tool.

---

## Recently Shipped Roadmap Features

Four of the eight gaps identified in the project's [Future Roadmap](#future-roadmap) have since moved from "planned" to **implemented and live in the dashboard**: **Feature 1** (Driver Identity & Vehicle Assignment), **Feature 2** (Live SoC / Range Tile), **Feature 5** (Driver-Level Coaching), and **Feature 8** (Weather-Aware Range Estimate). The [Future Roadmap](#future-roadmap) section below has been updated to mark these as shipped rather than re-describing them there; this section documents what actually landed.

### Feature 1 — Driver Identity & Vehicle Assignment ✅

Every signal in VoltSentinel used to be attributed to a `vehicle_id` only. Two new tables now attach a "who was driving this vehicle, and when" dimension on top, without changing any existing signal:

- **`drivers`** (`driver_id`, `name`, `license_id`, `depot_home`) and **`vehicle_assignments`** (`vehicle_id`, `driver_id`, `shift_start`, `shift_end`) in `ingestion/db.py`, using the same `INSERT OR IGNORE` idempotency pattern as every other table.
- New `Driver` / `VehicleAssignment` pydantic models in `ingestion/schemas.py`, matching the existing `extra: forbid` convention.
- `simulator/driver_generator.py` generates a mock driver pool (independent of fleet size — drivers rotate across vehicles, they don't own one each) and carves each vehicle's active time window into contiguous, non-overlapping shift assignments.
- **Dashboard:** `fleet_map.py`'s sortable asset table now has a **Driver** column (showing "Unassigned" for any vehicle with no assignment history) right after the Vehicle column.

### Feature 2 — Live SoC / Range Tile ✅

Answers the fleet manager's most operationally urgent question — *"can this vehicle make it back right now?"* — as a same-day signal, distinct from RUL's longer-horizon degradation projection:

- `models/range_estimator.py`: `estimated_range_km = capacity_kwh_remaining / kwh_per_km`, computed from each vehicle's **latest** telemetry row (not full history).
- A vehicle is flagged **at risk of stranding** if its current SoC or estimated range drops below configurable thresholds (`simulator/config.py`'s `low_soc_threshold_pct` / `low_range_threshold_km`).
- **Dashboard:** a fifth "⚠️ Stranding Risk" tile in `fleet_map.py`'s summary row, a distinct red **ring** around any at-risk vehicle's map marker (kept separate from the existing risk-level fill color, so "high risk" and "stranding risk" never mask each other), and SoC / Est. Range columns in the fleet table. `health_chart.py`'s Asset Detail tab shows the same signal per-asset as a "Live SoC / Range" metric tile, with a warning banner when that vehicle is currently at risk.

### Feature 5 — Driver-Level Coaching ✅

`charging_analyzer.py` already computed exactly the behaviours that matter for coaching — but scored the *vehicle*, blending together every driver who ever used it. Built on top of Feature 1:

- `ChargingAnalyzer.analyze_fleet_drivers()` re-aggregates the same fast-charge-frequency / DoD / stress-trend math per `driver_id`, using `vehicle_assignments` to attribute each telemetry row to whoever was actually driving at the time — across however many different vehicles that driver used.
- `agent/prompts.py` gains a dedicated `DRIVER_COACHING_SYSTEM_PROMPT` + `build_driver_coaching_prompt()`, producing a short, signal-grounded coaching note (e.g. "fast-charge frequency 40% above fleet baseline across three assigned vehicles") in the same explainable style as the main decision loop.
- **Dashboard:** a new **🧑‍✈️ Driver Scorecard** tab (`dashboard/components/driver_scorecard.py`), sorted worst-charge-stress-first, showing vehicles driven, charge cycles observed, fast-charge frequency vs. fleet baseline, mean DoD, stress trend, and a coaching suggestion per driver.

### Feature 8 — Weather-Aware Range Estimate ✅

Battery efficiency (kWh/km) is materially affected by ambient temperature — worse in the cold, and to a smaller degree in extreme heat. This closes the gap on Feature 2's range estimate by adjusting it for live weather:

- `models/weather_client.py`: a thin, network-isolated client using **Open-Meteo** (no API key, no signup required), with a built-in 10-minute per-coordinate cache so the dashboard's few-second auto-refresh doesn't hammer the API.
- `models/range_estimator.py` applies a plain, explainable multiplier on top of `kwh_per_km` — a cold penalty below a reference temperature, a smaller heat penalty above another, capped so a bad reading can't blow up the estimate. This is a correction term on the existing baseline, not a replacement model, matching `rul_model.py`'s "transparent curve fit over a black-box number" philosophy elsewhere in `models/`.
- If the live weather lookup fails for any reason (network down, vehicle has no known location yet, etc.), the estimate silently degrades to exactly the original weather-agnostic number — nothing breaks.

**Why it's called out separately here:** the on-screen difference is genuinely subtle — a couple of km of range and one small caption — so it's easy to miss during a demo unless you know to look for it:

<p align="center">
  <img src="images/live-soc-range-weather.png" alt="Asset Detail 'Live SoC / Range' metric tile showing 17% · 5.4 km, with a small weather caption reading '30°C ambient' underneath" width="360">
</p>

That `🌤️ 30°C ambient` caption underneath the **Live SoC / Range** tile (Asset Detail tab, `health_chart.py`) is the only visible sign that the `5.4 km` figure above it has already been adjusted for live temperature at that vehicle's last known location — without it, the range number would look identical whether weather adjustment ran or not. The same weather-adjusted number also flows into `fleet_map.py`'s fleet-wide stranding-risk banner text when it materially changes the figure.

---

## Data Simulation Approach

With no access to real fleet or BMS data, the simulator is the foundation the rest of the system depends on. It generates, for a configurable fleet size and cycle count, per-vehicle battery telemetry — capacity, voltage, temperature, state of charge, and charge-cycle behaviour — following an exponential degradation trend with realistic noise, alongside a corresponding stream of BMS command events.

A separate injection process introduces "attack" events into the command stream at controlled points: unauthorized discharge/disable commands with no maintenance ticket, a GPS position away from any depot, or a burst of repeated commands — mirroring the mechanics of the real Tirri Challenge incidents. This allows the detection layer to be validated against known-injected events, and allows the live demo to trigger a realistic scenario on demand and observe the agent's resulting decision.

```mermaid
flowchart LR
    Base["Baseline Generator<br/>Exponential capacity-fade trend<br/>+ realistic noise, per vehicle"]
    Norm["Normal Event Stream<br/>Routine charge cycles ·<br/>ticketed maintenance events"]
    Attack["Attack Injection<br/>No-ticket disable · GPS mismatch ·<br/>command-frequency burst"]
    Merge(["Combined Telemetry +<br/>Command Event Stream"])
    Demo["'Simulate Attack' button<br/>(on-demand demo trigger)"]

    Base --> Merge
    Norm --> Merge
    Attack --> Merge
    Demo -.-> Attack
    Merge --> API["→ FastAPI Ingestion"]
```

---

## Configuration

VoltSentinel runs fully with no configuration at all (every real integration below degrades gracefully to its original mocked/logged-only behaviour if unset) — these environment variables opt into live behaviour on top of that baseline.

| Variable                       | Required for                                                            | Notes                                                                                                                                                                          |
| ------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GEMINI_API_KEY`               | Agent decision loop, Fleet BI chat, per-asset research, driver coaching  | Without it, `agent_recommendations.py` and `bi_chat.py` both show an explicit "set this to enable" message rather than crashing — default charts and history views still work. |
| `SMTP_HOST` + `ALERT_EMAIL_TO` | Tier 1 real email (`create_maintenance_ticket`, `notify_fleet_manager`) | Both must be set for email to actually send. If either is missing, these actions fall back to the original logged-only behaviour — no error, no partial send.                  |
| `SMTP_PORT`                    | Tier 1 real email (optional)                                            | Defaults to `587`.                                                                                                                                                             |
| `SMTP_USER` / `SMTP_PASSWORD`  | Tier 1 real email (optional)                                            | Only needed if your SMTP server requires authentication.                                                                                                                       |
| `SMTP_FROM`                    | Tier 1 real email (optional)                                            | Defaults to `SMTP_USER`, or `voltsentinel@localhost` if that's also unset.                                                                                                     |

Scoped per-asset research (`enable_research=True`) and the Fleet BI chat's web-search tool (`enable_web_search=True`) are both additionally **opt-in at the call site**, on top of `GEMINI_API_KEY` being set — neither reaches the open web unless a caller explicitly requests it.

The **Live SoC / Range** tile (Feature 2) and its weather adjustment (Feature 8) need **no configuration at all** — `models/weather_client.py` uses Open-Meteo's key-free API, and both degrade automatically (to a weather-agnostic estimate) if the network is unavailable.

---

## Alignment with Judging Criteria

| Criterion                | Weight | How VoltSentinel Addresses It                                                                                                                                                                                  |
| ------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Innovation**           | 25%    | Agentic reasoning over combined health, thermal, charging, and security signals, tied to a real, unresolved regulatory gap — rather than a generic battery-health dashboard.                                   |
| **Business Impact**      | 25%    | Protects fleet revenue and driver safety, reduces unplanned downtime via predictive triggers and charge-policy guidance, in a segment with a live public-safety incident and no deployed detection tooling.    |
| **Technical Excellence** | 20%    | Multi-signal pipeline (RUL regression, thermal detection, charging-pattern analysis, security anomaly detection, live weather-adjusted range) converging into an agent decision layer that reasons and acts on a shared real-time pipeline. |
| **Scalability**          | 15%    | Modular pipeline (simulation, ingestion, storage, models, agent, dashboard) designed to accept real BLE-connected BMS input in place of the simulator, and to swap mocked actions for live integrations.       |
| **User Experience**      | 15%    | Single unified dashboard combining health, prescriptive, and security signals per asset, driver-aware views, live range/weather awareness, and agent-generated recommendations, with a live, on-demand attack simulation for a clear demo narrative. |

---
