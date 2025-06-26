import os
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
from pydub import AudioSegment
import io

from deepgram import (
    DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions
)
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import stream

# --- INITIALIZATION ---
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize clients
deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

app = FastAPI()

# --- THE MAIN CONVERSATIONAL LOGIC ---
class ConversationManager:
    def __init__(self, websocket: WebSocket, stream_sid: str):
        self.websocket = websocket
        self.stream_sid = stream_sid
        self.is_speaking = False # To prevent interruption

 
    async def handle_response(self, text: str):
        if self.is_speaking:
            print("[DEBUG] AI is already speaking, ignoring new request.")
            return
        
        self.is_speaking = True
        print(f"--- STARTING AI RESPONSE ---")
        print(f"1. AI Text: '{text}'")

        # Step 1: Generate high-quality MP3 audio from ElevenLabs
        try:
            print("2. Calling ElevenLabs to generate MP3 audio stream...")
            # audio_stream = elevenlabs_client.generate(
            #     text=text,
            #     voice="Rachel",
            #     model="eleven_turbo_v2",
            #     stream=True
            # )
            audio_stream = elevenlabs_client.text_to_speech.stream(
                text=text,
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Adam - a pre-made voice
                model_id="eleven_turbo_v2"
            )
            print("3. ElevenLabs stream received.")
        except Exception as e:
            print(f"[ERROR] ElevenLabs API call failed: {e}")
            self.is_speaking = False
            return

        # Step 2: Collect the MP3 audio chunks into a single byte string
        try:
            print("4. Collecting MP3 audio chunks into memory...")
            mp3_data = b"".join([chunk for chunk in audio_stream])
            print(f"5. MP3 audio collected. Total size: {len(mp3_data)} bytes.")
            if len(mp3_data) == 0:
                print("[ERROR] MP3 data from ElevenLabs is empty. Aborting.")
                self.is_speaking = False
                return
        except Exception as e:
            print(f"[ERROR] Failed to collect audio chunks: {e}")
            self.is_speaking = False
            return

        # Step 3: Use pydub to convert the audio format in memory
        try:
            print("6. Starting pydub conversion: MP3 -> 8kHz mu-law...")
            # Load the MP3 data from memory
            audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
            print("7. pydub loaded MP3 data from memory.")

            # Set the frame rate to 8000Hz (required for Twilio)
            resampled_audio = audio_segment.set_frame_rate(8000)
            print("8. pydub resampled audio to 8000Hz.")
            
            # Export as mu-law format
            output_bytes = resampled_audio.export(format="mulaw").read()
            print(f"9. pydub exported audio to mu-law. Total size: {len(output_bytes)} bytes.")

        except Exception as e:
            print(f"[ERROR] Pydub conversion failed: {e}")
            self.is_speaking = False
            return
            
        # Step 4: Send the converted mu-law audio back to Twilio
        try:
            print(f"10. Preparing to send {len(output_bytes)} bytes of audio to Twilio in chunks...")
            chunk_size = 2048 # Send in 2KB chunks
            total_chunks = (len(output_bytes) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(output_bytes), chunk_size):
                chunk_num = (i // chunk_size) + 1
                chunk = output_bytes[i:i + chunk_size]
                
                # This log can be very noisy, so it's commented out by default
                # print(f"   - Sending chunk {chunk_num}/{total_chunks} ({len(chunk)} bytes)")
                
                media_message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {
                        "payload": base64.b64encode(chunk).decode('utf-8')
                    }
                }
                await self.websocket.send_text(json.dumps(media_message))
            
            print(f"11. All {total_chunks} chunks sent to Twilio successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to send audio chunk to Twilio: {e}")
            self.is_speaking = False
            return

        # Step 5: Send a "mark" message to signal the end of speech
        try:
            mark_message = {
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": { "name": "ai_speech_ended" }
            }
            await self.websocket.send_text(json.dumps(mark_message))
            print("12. 'Mark' message sent to Twilio.")
        except Exception as e:
            print(f"[ERROR] Failed to send 'mark' message to Twilio: {e}")

        self.is_speaking = False
        print("--- AI RESPONSE COMPLETE ---")

# --- WEBSOCKET ENDPOINT ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("--- Twilio WebSocket connection accepted ---")
    
    stream_sid = None
    conversation_manager = None
    
    try:
        # Deepgram connection setup
        dg_connection = deepgram_client.listen.asynclive.v("1")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            
            print(f"User: {sentence}")

            if result.is_final and conversation_manager:
                print(f"Final transcript received: '{sentence}'.")
                # Send to OpenAI
                chat_completion = openai_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a friendly and helpful AI assistant named Leo. Keep your responses very short and conversational."},
                        {"role": "user", "content": sentence},
                    ],
                    model="gpt-3.5-turbo",
                )
                ai_response = chat_completion.choices[0].message.content
                await conversation_manager.handle_response(ai_response)

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        await dg_connection.start(LiveOptions(model="nova-2-general", language="en-IN", encoding="mulaw", sample_rate=8000, punctuate=True, interim_results=True))

        # Main loop to process messages from Twilio
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get('event')

            if event == 'start':
                stream_sid = data['start']['streamSid']
                conversation_manager = ConversationManager(websocket, stream_sid)
                print(f"Twilio stream started with SID: {stream_sid}")

            elif event == 'media':
                await dg_connection.send(base64.b64decode(data['media']['payload']))

            elif event == 'stop':
                print("--- Twilio 'stop' event received. Call has ended. ---")
                break
    
    except WebSocketDisconnect:
        print("--- Twilio WebSocket disconnected ---")
    
    finally:
        print("--- Cleaning up connection... ---")
        # Clean shutdown logic can be added here if needed