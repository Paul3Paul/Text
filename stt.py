from fastapi import FastAPI, WebSocket
import whisper
import numpy as np
import asyncio
import uvicorn
import logging
import warnings

# Suppress FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    model = whisper.load_model("medium")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise

BUFFER_DURATION_MS = 5000
SAMPLE_RATE = 16000
SAMPLES_PER_BUFFER = int(SAMPLE_RATE * BUFFER_DURATION_MS / 1000)
SILENCE_THRESHOLD = 0.01

@app.websocket("/transcribe")
async def transcribe_audio(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = np.array([], dtype=np.float32)
    is_closed = False

    try:
        while not is_closed:
            # Receive raw PCM audio chunk
            data = await websocket.receive_bytes()
            logger.info(f"Received audio chunk of size: {len(data)} bytes")

            # Convert bytes to NumPy array (16-bit PCM to float32)
            try:
                pcm_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer = np.append(audio_buffer, pcm_data)
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                if not is_closed:
                    await websocket.send_text(f"Error: Invalid audio data")
                continue

            # Check if buffer is full
            if len(audio_buffer) >= SAMPLES_PER_BUFFER:
                # Silence detection (skip if RMS is too low)
                rms = np.sqrt(np.mean(audio_buffer**2))
                if rms < SILENCE_THRESHOLD:
                    logger.info("Skipping silent buffer")
                    audio_buffer = audio_buffer[-int(SAMPLE_RATE * 1.0):]  # Keep 1s overlap
                    continue

                try:
                    # Normalize audio (amplify if too quiet)
                    audio_buffer = audio_buffer / max(np.max(np.abs(audio_buffer)), 1e-8)
                    # Transcribe with Whisper
                    result = model.transcribe(audio_buffer, language="en")
                    transcription = result["text"].strip()
                    logger.info(f"Transcription: {transcription}")
                    if transcription and not is_closed:
                        await websocket.send_text(transcription)
                except Exception as e:
                    logger.error(f"Error during transcription: {e}")
                    if not is_closed:
                        await websocket.send_text(f"Error: Transcription failed")

                # Reset buffer with overlap
                audio_buffer = audio_buffer[-int(SAMPLE_RATE * 1.0):]  # 1s overlap

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if not is_closed:
            try:
                await websocket.send_text(f"Error: {str(e)}")
            except:
                pass
    finally:
        is_closed = True
        try:
            await websocket.close()
            logger.info("WebSocket connection closed")
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_ping_interval=None)