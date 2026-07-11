# ⚡ VOLTSENTINEL
**AI-Powered EV Battery Asset Performance & Security Intelligence Agent**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)

---

## 📌 Project Overview
**ET AI Hackathon 2026**  
**Problem Statement 3:** AI for Industrial EV Supply Chain & Asset Intelligence: Accelerating Net Zero  
**Team Name:** Team btech10364.23
**Team Members:** YASH SINHA  

---

## 🚀 1. Executive Summary
**VoltSentinel** is an AI-powered Asset Performance Management (APM) agent for industrial and commercial EV fleets. It extends traditional battery health monitoring with a **security-aware layer** that distinguishes legitimate maintenance activity from unauthorized Battery Management System (BMS) control commands. 

This directly addresses a real and unresolved vulnerability affecting low-cost EV battery packs across India, giving this solution a concrete, verifiable, and highly differentiated real-world grounding.

---

## ⚠️ 2. Problem Statement & Context

### 2.1 Industry Context
India's industrial and commercial EV segment is heavily under-penetrated due to operational barriers rather than financial ones. Fleet operators lack the intelligence tools needed to manage EV battery lifecycles, maintenance, and charging infrastructure. This issue is compounded by a supply chain relying on low-cost imported battery packs with inconsistent quality and security standards.

### 2.2 The Security Gap: The 'Tirri Challenge'
In mid-2026, a viral trend in India dubbed the **'Tirri Challenge'** saw pranksters use companion apps (BAT-BMS, Lossigy, Epoch i-ion) to remotely disable moving e-rickshaws. Using unauthenticated Bluetooth connections, they triggered discharge cut-offs in live traffic. 

The root cause was the **absence of authentication**. 
* Low-cost battery packs shipped with open Bluetooth access.
* The government reactively banned the apps, but the underlying hardware/firmware vulnerabilities remain.
* **The Gap:** There is currently no deployed detection or response capability at the fleet-operator level to protect against this live vulnerability.

---

## 🎯 3. Objectives
1. **Health Monitoring:** Monitor battery health and predict Remaining Useful Life (RUL) across a simulated EV fleet.
2. **Security Detection:** Detect unauthorized BMS control commands (treating them with skepticism rather than as trusted signals).
3. **Contextual Intelligence:** Distinguish legitimate maintenance shutdowns from 'Tirri Challenge'-style attacks using explainable, auditable logic.
4. **Unified Dashboard:** Present both health and security signals in a single operator view.
5. **End-to-End Simulation:** Demonstrate the concept using synthesized data representing realistic fleet and BMS telemetry.

---

## 💡 4. Proposed Solution

### 4.1 Core Concept
VoltSentinel ingests battery telemetry and BMS command events for a simulated fleet and routes them through two parallel intelligence layers on a shared pipeline:
* **Health/Degradation Layer:** Predicts Remaining Useful Life (RUL).
* **Security Layer:** Scores control commands against contextual signals (maintenance tickets, GPS location, command frequency) to flag unauthorized activity.

### 4.2 Key Differentiator: Not Another BMS App
VoltSentinel operates a layer above individual BMS companion apps. It is a **fleet-operator oversight tool** that watches the command traffic and judges it.

| Aspect | 📱 BMS Companion App (e.g. BAT-BMS) | ⚡ VoltSentinel |
| :--- | :--- | :--- |
| **User** | Individual driver / vehicle owner | Fleet manager / operator |
| **Scope** | One vehicle at a time | Entire fleet, aggregated |
| **Relationship** | Sends commands to the BMS | Observes and evaluates commands |
| **Security** | None — accepts any nearby command | Flags commands lacking tickets, location match, or normal frequency |
| **Prediction** | None — current readings only | Battery degradation trend and RUL forecasting |

---

## 🏗️ 5. System Architecture

The system follows a linear data pipeline that branches into two analytical consumers before converging into one unified Streamlit output.

```mermaid
%%{init: {'themeVariables': { 'textColor': '#F08080', 'edgeLabelText': '#F08080'}}}%%
graph TD
    A[Python Simulator<br>numpy / pandas] -->|Generates telemetry & tickets| B[FastAPI Ingestion Service<br>REST / WebSocket]
    B --> C[(SQLite Data Store)]
    C -->|Telemetry Data| D[scikit-learn RUL Model<br>Health Analytics Branch]
    C -->|Event Logs| E[Anomaly & Security Detector<br>Security Detection Branch]
    D --> F[Streamlit Dashboard<br>Unified Fleet View]
    E --> F
    
    classDef default color:#F08080;
    style A fill:#e0e0e0,stroke:#333,stroke-width:2px,color:#F08080
    style B fill:#bbdefb,stroke:#333,stroke-width:2px,color:#F08080
    style C fill:#e0e0e0,stroke:#333,stroke-width:2px,color:#F08080
    style D fill:#b2dfdb,stroke:#333,stroke-width:2px,color:#F08080
    style E fill:#ffcdd2,stroke:#333,stroke-width:2px,color:#F08080
    style F fill:#e1bee7,stroke:#333,stroke-width:2px,color:#F08080
    linkStyle default stroke:#333;
```

### Component Overview
* **Data Simulation (Python/numpy/pandas):** Generates synthetic battery telemetry, normal maintenance events, and injected unauthorized 'attack' events.
* **Ingestion (FastAPI):** Receives and normalizes incoming telemetry and commands.
* **Storage (SQLite):** Stores telemetry history, mock maintenance tickets, and event logs.
* **Predictive Analytics (scikit-learn):** Fits a capacity-fade curve per asset to extrapolate Remaining Useful Life (RUL).
* **Security Analytics (IsolationForest / Rules):** Scores commands against contextual parameters.
* **Presentation (Streamlit):** Unified fleet dashboard with health charts, live alerts, and a 'Simulate Attack' trigger.

---

## 🛡️ 6. Anomaly & Security Detection Logic
A BMS control command (e.g., discharge cut-off) is flagged as suspicious when it exhibits:
1. **Missing Ticket:** No matching maintenance ticket exists.
2. **GPS Mismatch:** Vehicle GPS is inconsistent with known depots (e.g., in motion on a public road).
3. **Frequency Spike:** Control commands spike above the historical baseline in a short window.

Commands matching valid tickets and expected locations are explicitly excluded from alerting to maintain explainable and auditable logic.

---

## 📊 7. Data Simulation Approach
To overcome the lack of real fleet data, a robust simulator generates:
* **Per-Vehicle Telemetry:** Capacity, voltage, temperature, state of charge (following an exponential degradation trend with noise).
* **BMS Command Stream:** Normal operation commands.
* **Attack Injection:** Introduces unauthorized commands (no-ticket, GPS mismatch, burst frequency) mirroring the Tirri Challenge to validate the detection layer.

---

## 🗓️ 8. Implementation Timeline (11 Days)

| Day | Focus |
| :---: | :--- |
| **1** | Project setup; begin telemetry simulator. |
| **2-3** | Complete simulator; build attack injection logic. |
| **4** | FastAPI ingestion service and SQLite schema setup. |
| **5-6** | RUL regression model and rule-based anomaly detector implementation. |
| **7-8** | Streamlit dashboard (fleet view, alerts, 'Simulate Attack'). |
| **9** | End-to-end integration testing across the full pipeline. |
| **10** | Finalize architecture diagram, presentation deck, and demo video. |
| **11** | Buffer, rehearsal, and hackathon submission. |

---

## 🏆 9. Alignment with Judging Criteria

| Criterion | Weight | How VoltSentinel Addresses It |
| :--- | :---: | :--- |
| **Innovation** | 25% | Security-aware asset intelligence tied to a real, unresolved regulatory gap. |
| **Business Impact** | 25% | Protects fleet revenue and driver safety in a segment lacking detection tooling. |
| **Technical Excellence** | 20% | Dual-model pipeline combining regression RUL with rule/ML-based anomaly detection. |
| **Scalability** | 15% | Modular pipeline designed to accept real BLE-connected BMS input eventually. |
| **User Experience** | 15% | Unified dashboard combining health and security with a live attack simulation. |

---

## 📦 10. Expected Deliverables
- [x] **Architecture Diagram:** Complete (See Section 5)
- [ ] **Working Prototype:** In progress
- [ ] **Presentation Deck:** Pending
- [ ] **Demo Video:** Pending (Backup to live demo)

---

## 🏁 11. Conclusion
VoltSentinel reframes EV battery asset management as both a **health problem** and a **security problem**. By grounding the project in a current, unresolved vulnerability and keeping detection logic explainable, VoltSentinel delivers a differentiated, credible, and scalable solution for modern EV fleet operators.