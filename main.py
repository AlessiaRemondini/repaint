from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import base64
from PIL import Image
import numpy as np

# Importiamo le tue funzioni dal tuo file
from colorizzazione_avanzata_hd import Params, colorizzazione_avanzata_hd, im2double_local

app = FastAPI()

# Permette a Lovable di connettersi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(img_array):
    """Converte un array numpy in stringa base64 per Lovable"""
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.post("/repaint")
async def repaint_endpoint(bw: UploadFile = File(...), ref: UploadFile = File(...)):
    # 1. Leggi le immagini caricate
    bw_content = await bw.read()
    ref_content = await ref.read()
    
    # 2. Converti in formato PIL e poi in array per il tuo codice
    bw_img = Image.open(io.BytesIO(bw_content)).convert('RGB')
    ref_img = Image.open(io.BytesIO(ref_content)).convert('RGB')
    
    # Trasforma in double [0,1] come richiesto dal tuo algoritmo
    I_bw = np.array(bw_img).astype(np.float64) / 255.0
    I_ref = np.array(ref_img).astype(np.float64) / 255.0

    # 3. Esegui il tuo algoritmo (qui chiami la tua funzione principale)
    # Nota: assicurati che la tua funzione restituisca un dizionario o una lista dei risultati
    params = Params(output_res=(I_bw.shape[0], I_bw.shape[1]))
    
    # ESEMPIO: Supponiamo che la tua funzione restituisca una lista di array numpy
    # risultati = colorizzazione_avanzata_hd(I_bw, I_ref, params) 
    
    # Per ora simuliamo la risposta per i box di Lovable:
    response_data = {
        "method1": image_to_base64(I_bw), # Sostituisci con il risultato reale
        "method2": image_to_base64(I_bw), 
        "method3": image_to_base64(I_bw),
        "method4": image_to_base64(I_bw),
        "method5": image_to_base64(I_bw),
        "method6": image_to_base64(I_bw),
        "method7": image_to_base64(I_bw),
        "method8": image_to_base64(I_bw),
    }

    return JSONResponse(content=response_data)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
