import csv
import json
from typing import Literal

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:1b"


def call_ollama(prompt: str) -> str:
    """Llama a Ollama con el modelo gemma3."""
    payload = {"model": MODEL, "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error al llamar a Ollama: {e}")
        return ""


def translate_comment(comment: str) -> str:
    """Traduce un comentario del ingles al espanol."""
    if not comment or comment.strip() == "No Comments":
        return "Sin comentarios"

    prompt = f"""Translate the following comment from English to Spanish. 
Only return the translation, nothing else.

Comment: {comment}

Spanish translation:"""

    return call_ollama(prompt)


def classify_criticality(comment: str) -> Literal["normal", "critico", "muy critico"]:
    """Clasifica el nivel de criticidad de un comentario."""
    if not comment or comment.strip() == "No Comments":
        return "normal"

    prompt = f"""Analyze the following comment and classify its criticality level.
The comment is feedback about a professor/teacher.

Classify as:
- "normal": Neutral or positive feedback, constructive criticism
- "critico": Negative feedback, complaints, frustration expressed
- "muy critico": Very negative feedback, harsh criticism, strong negative emotions, mentions of failure, dropping class, or serious complaints

Only respond with one of these three words: normal, critico, muy critico

Comment: {comment}

Criticality level:"""

    response = call_ollama(prompt).lower().strip()

    # Normalizar la respuesta
    if "muy critico" in response or "muy crítico" in response:
        return "muy critico"
    elif "critico" in response or "crítico" in response:
        return "critico"
    else:
        return "normal"


def process_feedback(input_file: str, output_file: str):
    """Procesa el archivo de feedback, traduce y clasifica cada comentario."""
    rows = []

    # Leer el archivo CSV (usar latin-1 como fallback si utf-8 falla)
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except UnicodeDecodeError:
        with open(input_file, "r", encoding="latin-1") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    total = len(rows)
    print(f"Procesando {total} comentarios...")

    # Procesar cada fila
    for i, row in enumerate(rows, 1):
        comment = row.get("comment", "")

        print(f"[{i}/{total}] Procesando comentario...")

        # Traducir el comentario
        comment_es = translate_comment(comment)
        row["comment_es"] = comment_es

        # Clasificar criticidad
        nivel_criticidad = classify_criticality(comment)
        row["nivel_criticidad"] = nivel_criticidad

        print(f"  -> Criticidad: {nivel_criticidad}")

    # Escribir el archivo de salida
    fieldnames = ["Id", "comment", "quality", "comment_es", "nivel_criticidad"]

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nArchivo guardado en: {output_file}")


if __name__ == "__main__":
    input_file = "feedback.csv"
    output_file = "feedback_processed.csv"

    process_feedback(input_file, output_file)
