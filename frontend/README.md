# EconoCausal — Causal Pricing Intelligence

A distinctive dark, animated React + Vite frontend for the EconoCausal Dynamic Pricing via Double Machine Learning project.

## Features
- Animated 3D hero scene using Three.js (rotating causal-orbit model)
- Responsive dashboard and navigation
- Local CSV selection, header preview, column mapping, basic validation
- Analysis configuration for treatment, outcome and covariates
- Demo analysis results clearly marked as illustrative
- Budget scenario calculator and downloadable report
- API service layer ready for backend integration

## Run
1. Install Node.js LTS.
2. Open this folder in VS Code.
3. Terminal: `npm install`
4. Terminal: `npm run dev`
5. Open the local URL printed by Vite.

## Backend integration
Create a `.env` file using `.env.example` and set `VITE_API_BASE_URL=http://localhost:5000/api` (or your backend API base URL). The API service in `src/services/api.js` contains clearly named functions to connect:
- `POST /datasets` for multipart CSV upload
- `POST /analysis` for causal analysis
- `GET /datasets/:id` for dataset status
- `GET /reports/:id` for reports

Adjust endpoint paths and response shapes to match the backend team's actual API. The frontend does not claim to perform real Double ML estimation by itself. Without a connected backend, the app runs in demo mode and labels example results.

## Suggested CSV fields
At minimum, your backend/model should document the outcome, treatment, and confounder/covariate columns. The frontend lets users map columns rather than assuming all datasets have identical headers.
