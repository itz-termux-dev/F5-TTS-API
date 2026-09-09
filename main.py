from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from melo.api import TTS
import os

app = FastAPI()

# Initialize the pre-trained native Indian accent female voice profile
# 'EN-BR' or native mapping provides the smooth conversational Hinglish style
print("Loading MeloTTS Indian Female Engine...")
model = TTS(language='EN', device='cpu') 
speaker_ids = model.hps.data.spk2id

@app.post("/predict")
async def generate_speech(text: str):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    output_path = "output.wav"
    
    try:
        # Use the native Indian speaker ID profile for natural Hinglish flow
        # Speaker ID 2 or 4 represents the clear, emotional female tone
        model.tts_to_file(text, speaker_ids['EN-IN'], output_path, speed=1.0)
        return FileResponse(output_path, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
