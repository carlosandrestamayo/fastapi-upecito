from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from typing import Optional

# Cargar variables de entorno
load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_id = os.getenv("FINE_TUNED_MODEL")

app = FastAPI()

# Modelo de entrada
class ChatRequest(BaseModel):
    message: str

# Modelo de salida mejorado
class UpecitoResponse(BaseModel):
    success: bool
    message: str
    response: Optional[str] = None

@app.post("/upecito", response_model=UpecitoResponse)
async def chat_upecito(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "Eres Upecito, el asistente formal de la Universidad Popular del Cesar, Seccional Aguachica."},
                {"role": "user", "content": request.message}
            ]
        )
        return UpecitoResponse(
            success=True,
            message="Respuesta generada correctamente.",
            response=response.choices[0].message.content
        )
    except Exception as e:
        return UpecitoResponse(
            success=False,
            message=f"Error al generar respuesta: {str(e)}",
            response=None
        )
