import os
import base64
import json
import asyncio
from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
from fastapi.responses import PlainTextResponse


# Load environment variables from .env file or Replit secrets
load_dotenv()

# Get API keys from environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Setup Deepgram client
config = DeepgramClientOptions(
    verbose=1,  # Set to 1 for detailed logging
)
deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY, config)

# Setup FastAPI app
app = FastAPI()


async def get_deepgram_connection():
    """Establishes a connection to Deepgram and sets up event handlers."""
    options = LiveOptions(
        model="nova-2-general",
        language="en-US",
        encoding="mulaw",
        sample_rate=8000,
        # To get interim results, set interim_results=True
        interim_results=True,
        # To get transcripts with punctuation, set punctuate=True
        punctuate=True,
    )

    try:
        dg_connection = deepgram.listen.asynclive.v("1")

        async def on_message(self, result, **kwargs):
            """Handles transcription results from Deepgram."""
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return

            # This is where we print the transcript to the console
            print(f"User: {sentence}")

            if result.is_final:
                # This is where you would trigger the LLM in the next step
                print("--- Final Transcript Received ---")

        async def on_error(self, error, **kwargs):
            """Handles errors from Deepgram."""
            print(f"Deepgram Error: {error}")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        await dg_connection.start(options)
        return dg_connection

    except Exception as e:
        print(f"Could not open socket to Deepgram: {e}")
        return None


#
# REPLACE your old websocket_endpoint function with this entire block
#
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the WebSocket connection from Twilio and streams audio to Deepgram."""
    await websocket.accept()
    print("Twilio connection accepted.")
    
    dg_connection = await get_deepgram_connection()
    if dg_connection is None:
        print("Failed to connect to Deepgram. Closing WebSocket.")
        await websocket.close(code=1011, reason="Failed to connect to Deepgram")
        return

    try:
        while True:
            # Receive message from Twilio
            message = await websocket.receive_text()
            data = json.loads(message)
            
            # Handle different event types from Twilio
            event = data.get('event')

            if event == 'connected':
                print("Twilio 'connected' event received.")
            
            elif event == 'start':
                print("Twilio 'start' event received. Audio stream beginning.")

            elif event == 'media':
                # This is the actual audio data
                payload = data['media']['payload']
                # The payload is base64 encoded, so we decode it
                audio_data = base64.b64decode(payload)
                # Send the raw mulaw audio data to Deepgram
                await dg_connection.send(audio_data)

            elif event == 'stop':
                print("Twilio 'stop' event received. Call has ended.")
                break # Exit the loop to close the connection

            else:
                print(f"Received unknown event: {event}")

    except Exception as e:
        # This provides a much more detailed error message
        print(f"An error occurred in the WebSocket loop: {type(e).__name__} - {e}")
    
    finally:
            print("Attempting to close Deepgram connection...")
            try:
                await dg_connection.finish()
                print("Deepgram connection closed gracefully.")
            except asyncio.CancelledError:
                # This is expected if the client (Twilio) hangs up abruptly
                print("Deepgram finish() was cancelled, which is normal on hangup.")
            except Exception as e:
                print(f"Error closing Deepgram connection: {e}")

# A simple root endpoint to check if the server is running
@app.get("/")
def read_root():
    return {"status": "Server is running"}


if __name__ == "__main__":
    import uvicorn
    
    # Check if required environment variables are set
    if not DEEPGRAM_API_KEY:
        print("ERROR: DEEPGRAM_API_KEY environment variable is not set!")
        print("Please add your Deepgram API key to the Secrets tab.")
        exit(1)
    
    print("Starting FastAPI server...")
    print("WebSocket endpoint available at: ws://localhost:8000/ws")
    
    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
