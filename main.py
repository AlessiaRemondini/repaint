from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import base64
from PIL import Image
import numpy as np
import os

# Importiamo le funzioni dal tuo file ricolorazione
from colorizzazione_avanzata_hd import Params, colorizzazione_avanzata_hd

app = FastAPI()

# Permette a Lovable di connettersi senza blocchi di sicurezza
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(img_array):
    """Converte un array numpy ricolorato in stringa base64 leggibile da Lovable"""
    # Assicuriamoci che l'array sia nel formato corretto (0-255 uint8)
    img_fixed = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img_fixed)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.post("/repaint")
async def repaint_endpoint(bw: UploadFile = File(...), ref: UploadFile = File(...)):
    try:
        # 1. Leggi i file inviati da Lovable
        bw_content = await bw.read()
        ref_content = await ref.read()
        
        # 2. Converti in array per il tuo algoritmo
        bw_img = Image.open(io.BytesIO(bw_content)).convert('RGB')
        ref_img = Image.open(io.BytesIO(ref_content)).convert('RGB')
        
        I_bw = np.array(bw_img).astype(np.float64) / 255.0
        I_ref = np.array(ref_img).astype(np.float64) / 255.0

        # 3. ESECUZIONE REALE DELL'ALGORITMO
        # Creiamo i parametri (puoi regolare la risoluzione qui se Render è lento)
        params = Params(output_res=(I_bw.shape[0], I_bw.shape[1]))
        
        # Chiamiamo la tua funzione (passando gli array, non i percorsi file)
        # NOTA: Questa funzione deve restituire il dizionario dei risultati (output_dict)
        risultati = colorizzazione_avanzata_hd(bw_img=I_bw, ref_img=I_ref, params=params) 
        
        # 4. Risposta per Lovable
        # Mappiamo i tuoi 8 metodi ricolorati
        response_data = {
            "method1": image_to_base64(risultati["method1"]),
            "method2": image_to_base64(risultati["method2"]),
            "method3": image_to_base64(risultati["method3"]),
            "method4": image_to_base64(risultati["method4"]),
            "method5": image_to_base64(risultati["method5"]),
            "method6": image_to_base64(risultati["method6"]),
            "method7": image_to_base64(risultati["method7"]),
            "method8": image_to_base64(risultati["method8"]),
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
