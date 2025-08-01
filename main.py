from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import google.generativeai as genai
from google.generativeai import types
from typing import Optional
from dotenv import load_dotenv
import logging
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="API de Asistente Legal Virtual",
    description="API que conecta tu abogado virtual con el modelo Gemini para consultas legales en México.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes poner la URL de tu frontend en vez de "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LegalQuestion(BaseModel):
    question: str
    language_style: Optional[str] = "normal"  # "tecnico" o "normal"

def get_gemini_response(user_question: str, style: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.error("La variable de entorno GEMINI_API_KEY no está configurada.")
        raise HTTPException(status_code=500, detail="La API Key de Gemini no está configurada.")

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"

    system_prompt = (
        "Eres un asistente legal virtual experto en leyes mexicanas. "
        "Responde de forma clara, precisa y profesional. "
        "Cuando cites leyes, incluye el nombre, artículo y fracción si aplica. "
        "Si no tienes suficiente información, indícalo con honestidad. "
        "Formato de respuesta:\n"
        "1. Respuesta breve y clara.\n"
        "2. Fundamento legal (si aplica).\n"
        "3. Recomendación adicional (si aplica).\n"
    )

    if style and style.lower() == "tecnico":
        language_instruction = "Responde en lenguaje técnico y jurídico, usando términos legales."
    else:
        language_instruction = "Responde en lenguaje claro y sencillo, fácil de entender para cualquier persona."

    full_user_prompt = f"{user_question}\n\n{language_instruction}"

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)]),
        types.Content(role="model", parts=[types.Part.from_text(text="Comprendido. Estoy listo para sus consultas.")]),
        types.Content(role="user", parts=[types.Part.from_text(text=full_user_prompt)]),
    ]

    tools = [types.Tool(googleSearch=types.GoogleSearch())]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        tools=tools,
    )

    full_response_text = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            full_response_text += chunk.text
        return full_response_text
    except Exception as e:
        logging.error(f"Error al generar contenido con Gemini: {e}")
        raise HTTPException(status_code=500, detail="Error al procesar la consulta con el modelo IA.")

@app.post("/consultar-abogado")
async def consultar_abogado_virtual(question_data: LegalQuestion):
    """
    Endpoint para enviar una consulta legal al asistente virtual.
    El frontend puede cambiar el parámetro 'language_style' para alternar entre lenguaje técnico y coloquial.
    """
    try:
        response_text = get_gemini_response(question_data.question, question_data.language_style)
        return {"response": response_text}
    except HTTPException as e:
        logging.error(f"HTTPException: {e.detail}")
        raise
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
