# NeuroStroke-XAI — Project Roadmap

## Phase 1 — CT pipeline
- TEKNOFEST 2021 dataset audit
- Patient/study-level split if identifiers permit
- CT preprocessing
- 3-class model: Normal / Ischemic / Hemorrhagic
- Evaluation and reproducible baseline

## Phase 2 — MRI pipeline
- MRI dataset research and selection
- MRI preprocessing
- 3-class MRI pipeline where supported by the selected dataset

## Phase 3 — Core imaging engine
- CT + MRI modality routing
- Common prediction/result schema

## Phase 4 — XAI + application
- Grad-CAM / SHAP as experimentally justified
- Risk assessment
- FastAPI backend
- PostgreSQL
- React frontend
- Patient/case workflow

## Phase 5 — Subtype classification
- León: ischemic subtype extension
- PhysioNet ICH: hemorrhage subtype extension
- Validate subtype models separately

## Phase 6 — Final integration
- End-to-end testing
- Generalization analysis
- Explanation evaluation
- Documentation and demo
