# Detector de Supresión Expresiva en Español

Automated detector of **expressive suppression** in Spanish text, developed as part of a Master's 
dissertation in Artificial Intelligence and Behavioural Sciences (UNIR, 2026).

Expressive suppression is a transdiagnostic emotional regulation process — the voluntary inhibition of 
emotional expression while the emotion is still being experienced. This project builds the first 
automated detector of this construct in Spanish.

**Live demo:** [huggingface.co/spaces/daniguillen/detector_supresion_expresiva](https://huggingface.co/spaces/daniguillen/detector_supresion_expresiva)

---

## What this project does

Given a Spanish text message, the system predicts whether it contains expressive suppression and 
returns:
- A binary label (suppression present / not detected)
- A confidence probability (0–100%)
- The specific linguistic markers detected (e.g. *"me lo trago"*, *"por dentro"*, *"no digo nada"*)
- A conversational module (LLaMA 3 via Groq API) to answer questions about the analysis in natural
  language

---

## Project overview

| Stage | Description |
|---|---|
| Corpus | MentalRiskES — 46,384 messages from 1,334 users (anxiety, depression, ED) |
| Preprocessing | Emoji removal, URL filtering, minimum length threshold (≥15 words) → 10,745 messages |
| Manual annotation | Exhaustive review of 10,745 messages → 25 real positives (0.23%) |
| Synthetic data | Corpus expanded to 4,034 messages via LLM-guided generation |
| Bias correction | Length bias detected and corrected → final corpus: 2,195 messages (50.8% positive) |
| Models | Fine-tuning of BETO and RoBERTa on GPU (Google Colab) |
| Results | F1 = 0.990 on held-out test set (317 messages) |
| Deployment | FastAPI REST API + Hugging Face Spaces frontend |

---

## Tech stack

- **NLP & ML:** Hugging Face Transformers, PyTorch, scikit-learn
- **Models:** BETO (`dccuchile/bert-base-spanish-wwm-cased`), RoBERTa (`projecte-aina/roberta-base-ca-v2`)
- **Data processing:** pandas, regex, numpy
- **API:** FastAPI, uvicorn
- **Frontend:** v0.dev (React/Next.js), Hugging Face Spaces
- **Conversational module:** LLaMA 3 via Groq API
- **Environment:** Jupyter Notebooks, Google Colab (GPU T4), Anaconda

---

## Repository structure

detector_supresion/
├── api.py              # FastAPI REST API — loads model and serves predictions
├── requirements.txt    # Python dependencies
├── Procfile            # Deployment configuration
└── modelo_beto/        # Fine-tuned BETO model weights (via Git LFS)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    └── vocab.txt

---

## How to run locally

**1. Clone the repository:**
```bash
git clone https://github.com/danielguilleng/detector_supresion.git
cd detector_supresion
```

**2. Create and activate environment:**
```bash
conda create -n detector python=3.11 -y
conda activate detector
pip install -r requirements.txt
```

**3. Start the API:**
```bash
uvicorn api:app --reload
```

**4. Test it:**
```bash
curl -X POST http://127.0.0.1:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{"texto": "me lo trago todo, por dentro estoy destrozada pero no digo nada"}'
```

Expected response:
```json
{
  "supresion": 1,
  "probabilidad": 99.8,
  "etiqueta": "Supresion presente",
  "marcadores": ["me lo trago", "por dentro", "no digo nada"]
}
```

---

## Key findings

- Only **25 out of 10,745** real messages contained expressive suppression (0.23%) — a theoretically
  meaningful result: people who suppress tend not to verbalise their emotions in public support groups
- A classifier using **only message length** achieved F1 = 0.727 without reading a single word,
  revealing a structural bias that was detected and corrected before training
- Fine-tuned transformers achieved **F1 = 0.990** on the held-out test set with zero false positives

---

## Authors

- **Marta Ramón Elipe** — Clinical psychology, construct operationalisation, manual annotation
- **Paula María Rodríguez Martín** — Linguistics, annotation manual design, ethical framework
- **Daniel Guillén Guerrero** — ML pipeline, fine-tuning, API & deployment

**Supervisor:** Miguel Ángel Navarro Arenas — UNIR, 2026

---

## Ethical note

This tool is a research instrument, not a clinical diagnostic system. Expressive suppression is a 
transdiagnostic process, not a disorder. Any clinical application would require validation on real 
clinical data and professional supervision.
