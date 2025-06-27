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
        print(f"--- STARTING AI RESPONSE (Final Pattern) ---")
        print(f"1. AI Text: '{text}'")

        try:
            print("2. Calling ElevenLabs for 8kHz mu-law audio...")
            # This returns a BLOCKING generator, which is the source of all our complexity
            audio_stream = elevenlabs_client.text_to_speech.stream(
                text=text,
                voice_id="21m00Tcm4TlvDq8ikWAM",
                model_id="eleven_turbo_v2",
                output_format="ulaw_8000"
            )
            print("3. ElevenLabs stream received.")

            # This is the correct, non-blocking pattern to handle a blocking generator in asyncio
            loop = asyncio.get_running_loop()
            stream_iterator = iter(audio_stream)

            # Wrapper function to safely get the next chunk in a separate thread
            def get_next_chunk():
                try:
                    return next(stream_iterator)
                except StopIteration:
                    return None

            print("4. Streaming non-blocking, paced chunks to Twilio...")
            chunk_count = 0
            while True:
                # Run the blocking call in the executor
                chunk = await loop.run_in_executor(None, get_next_chunk)
                
                # Stop when the generator is exhausted
                if chunk is None:
                    break

                chunk_count += 1
                media_message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(chunk).decode('utf-8')}
                }
                await self.websocket.send_text(json.dumps(media_message))
                
                # Pacing the stream is crucial for stability with Twilio
                await asyncio.sleep(0.02) # Increased sleep slightly for more stability
            
            print(f"5. All {chunk_count} chunks sent successfully.")

            # Send the mark message to signal the end of the TTS
            mark_message = {
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {"name": "ai_speech_ended"}
            }
            await self.websocket.send_text(json.dumps(mark_message))
            print("6. 'Mark' message sent.")

        except Exception as e:
            print(f"[ERROR] During AI response generation/streaming: {e}")

        finally:
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
                if not conversation_manager.is_speaking:
                  await dg_connection.send(base64.b64decode(data['media']['payload']))
                # await dg_connection.send(base64.b64decode(data['media']['payload']))

            elif event == 'stop':
                print("--- Twilio 'stop' event received. Call has ended. ---")
                break
    
    except WebSocketDisconnect:
        print("--- Twilio WebSocket disconnected ---")
    
    finally:
        print("--- Cleaning up connection... ---")
        # Clean shutdown logic can be added here if needed