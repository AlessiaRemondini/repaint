from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import numpy as np
from pathlib import Path

# Importiamo la tua logica di colorizzazione
# Assicurati che il file si chiami esattamente colorizzazione_avanzata_hd.py
import colorizzazione_avanzata_hd as colorizer

app = FastAPI()

# --- CONFIGURAZIONE CORS (IMPORTANTE PER LOVABLE) ---
origins = [
    "*",  # "*" significa "accetta tutti". Per produzione sarebbe meglio mettere l'URL specifico di Lovable, ma per test va benissimo così.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------------------------------

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "API di Colorizzazione Attiva! Vai su /docs per testare."}

@app.post("/repaint")
async def repaint_endpoint(
    bw_image: UploadFile = File(...),
    ref_image: UploadFile = File(...)
):
    try:
        # 1. Salva i file ricevuti temporaneamente
        bw_path = f"{UPLOAD_FOLDER}/{bw_image.filename}"
        ref_path = f"{UPLOAD_FOLDER}/{ref_image.filename}"
        
        with open(bw_path, "wb") as buffer:
            shutil.copyfileobj(bw_image.file, buffer)
        with open(ref_path, "wb") as buffer:
            shutil.copyfileobj(ref_image.file, buffer)

        # 2. Carica le immagini con OpenCV
        img_bw = cv2.imread(bw_path)
        img_ref = cv2.imread(ref_path)

        if img_bw is None or img_ref is None:
            raise HTTPException(status_code=400, detail="Errore nel caricamento delle immagini. Formato non valido?")

        # Convertiamo BGR (OpenCV) -> RGB (Processing) -> Float [0,1]
        img_bw = cv2.cvtColor(img_bw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Gestione B&N: se l'immagine ha 3 canali ma è grigia, ok. Se ne ha 1, convertiamo.
        if img_bw.ndim == 2:
            img_bw = np.stack([img_bw]*3, axis=-1)

        # 3. Eseguiamo la colorizzazione
        # Chiamiamo la funzione principale del tuo script
        # params=None usa i default definiti nel tuo file
        output_dict = colorizer.colorizzazione_avanzata_hd(img_bw, img_ref, params=None)
        
        # Recuperiamo il risultato migliore (es. 'blend' o 'method5')
        # Puoi cambiare 'blend' con 'method5' se preferisci il risultato Bilateral
        result_img = output_dict.get('blend') 
        
        if result_img is None:
             # Fallback se qualcosa va storto
            result_img = img_bw

        # 4. Salviamo il risultato
        output_filename = f"colorized_{bw_image.filename}"
        output_path = f"{RESULTS_FOLDER}/{output_filename}"
        
        # Convertiamo da Float [0,1] -> RGB uint8 -> BGR per salvataggio OpenCV
        result_uint8 = (np.clip(result_img, 0, 1) * 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_path, result_bgr)

        # 5. Restituiamo l'immagine processata
        return FileResponse(output_path, media_type="image/jpeg")

    except Exception as e:
        # Se c'è un errore, lo stampiamo nei log di Render e lo diciamo al frontend
        print(f"ERRORE: {e}")
        raise HTTPException(status_code=500, detail=str(e))
