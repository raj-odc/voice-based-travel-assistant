import os
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
from openai import OpenAI
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Setup Deepgram client
# The config is optional, but useful for things like logging
config = DeepgramClientOptions(verbose=0) # Set to 1 for more logs
deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY, config)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Setup FastAPI app
app = FastAPI()

# This function sets up the connection to Deepgram and defines event handlers
async def get_deepgram_connection():
    # These options are specific to the audio format Twilio sends
    options = LiveOptions(
        model="nova-2-general",
        language="en-IN",  # Using Indian English as discussed
        encoding="mulaw",  # Mu-law is the audio encoding from phone lines
        sample_rate=8000,  # 8000Hz is the sample rate for phone audio
        interim_results=True,
        punctuate=True,
    )
    
    try:
        dg_connection = deepgram.listen.asynclive.v("1")

        # --- Define Event Handlers ---
        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            
            print(f"User: {sentence}")

            # --- THIS IS THE NEW PART ---
            # Check if the transcript is final
            if result.is_final:
                print(f"Final transcript received: '{sentence}'. Sending to OpenAI...")
                
                try:
                    # Call OpenAI's chat completion API
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a friendly and helpful AI assistant. Keep your responses short and conversational.",
                            },
                            {
                                "role": "user",
                                "content": sentence,
                            },
                        ],
                        model="gpt-3.5-turbo", # Use gpt-4o for better quality
                    )
                    
                    ai_response = chat_completion.choices[0].message.content
                    print(f"AI Response: {ai_response}")
                    
                    # For now, we just print the response. We'll add the voice in the next step.

                except Exception as e:
                    print(f"Error calling OpenAI: {e}")

        async def on_error(self, error, **kwargs):
            print(f"Deepgram Error: {error}")

        # Register the event handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        # Start the connection
        await dg_connection.start(options)
        return dg_connection

    except Exception as e:
        print(f"Could not open socket to Deepgram: {e}")
        return None

# This is the main WebSocket endpoint that Twilio connects to
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("--- Twilio WebSocket connection accepted ---")
    
    dg_connection = await get_deepgram_connection()
    if dg_connection is None:
        await websocket.close(code=1011, reason="Failed to connect to Deepgram")
        return

    try:
        # This is the main loop that receives audio from Twilio
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                event = data.get('event')

                if event == 'media':
                    payload = data['media']['payload']
                    audio_data = base64.b64decode(payload)
                    await dg_connection.send(audio_data)

                elif event == 'stop':
                    print("--- Twilio 'stop' event received. Call has ended. ---")
                    break # Exit the loop

            except WebSocketDisconnect:
                print("--- Twilio WebSocket disconnected ---")
                break # Exit the loop if Twilio hangs up

    except Exception as e:
        print(f"An error occurred in the WebSocket loop: {e}")
    
    finally:
        # This is the crucial part to prevent the "tasks cancelled" error
        print("--- Cleaning up connection... ---")
        try:
            # We shield the finish() call to protect it from being cancelled
            # by FastAPI when the client disconnects.
            await asyncio.shield(dg_connection.finish())
            print("Deepgram connection finished.")
        except asyncio.CancelledError:
            # This is now the expected behavior. The shield is cancelled,
            # but the underlying finish() call is allowed to complete.
            print("Cleanup shield was cancelled, which is normal on hangup.")
        
        # Ensure the WebSocket is closed from our side as well
        if websocket.client_state.name != "DISCONNECTED":
            await websocket.close()
        print("--- WebSocket cleanup complete ---")