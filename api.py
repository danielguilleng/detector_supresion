from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

tokenizer = AutoTokenizer.from_pretrained("./modelo_beto")
model = AutoModelForSequenceClassification.from_pretrained("./modelo_beto")
model.eval()

MARCADORES = [
    "me lo trago", "me lo guardo", "me callo", "me calle", "me reprimo",
    "me contengo", "lo oculto", "disimulo", "finjo", "aparento",
    "me hago el fuerte", "me hago la fuerte", "poner buena cara",
    "por dentro", "aunque por dentro", "no digo nada", "no lo digo",
    "no quiero que sepan", "no quiero que vean",
    "intento no llorar", "aguanto las lagrimas", "me muerdo la lengua",
    "me lo como", "lo llevo dentro", "me lo guardo para mi",
    "me esfuerzo por no", "procuro que no se note", "trato de no mostrar",
    "nunca digo lo que siento", "nadie sabe lo que siento",
    "quiero pero no puedo", "me gustaria pero", "quisiera pero",
    "me cuesta expresar", "no soy capaz de decir", "no encuentro las palabras",
    "me atraganto", "reprimir", "suprimir", "contener", "inhibir", "callar"
]

def detectar_marcadores(texto):
    texto_lower = texto.lower()
    return [m for m in MARCADORES if m in texto_lower] or ["No se detectaron marcadores explicitos"]

class Mensaje(BaseModel):
    texto: str

@app.post("/predecir")
async def predecir(msg: Mensaje, request: Request):
    inputs = tokenizer(
        msg.texto,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    proba = torch.softmax(logits, dim=-1)[0][1].item()
    return {
        "supresion": int(proba >= 0.5),
        "probabilidad": round(proba * 100, 1),
        "etiqueta": "Supresion presente" if proba >= 0.5 else "No supresion detectada",
        "marcadores": detectar_marcadores(msg.texto)
    }

@app.get("/")
def root():
    return {"estado": "API funcionando correctamente"}
