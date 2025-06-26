import os
import json
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    BufferSource,
)

# Load environment variables from .env file
load_dotenv()

# The path to your audio file
AUDIO_FILE = "test_audio.wav"

def main():
    """
    Transcribes a local audio file using Deepgram's Pre-recorded API.
    """
    try:
        # STEP 1: Create a Deepgram client using your API key
        # You can optionally configure the client with a config object
        config = DeepgramClientOptions(verbose=0) # Change to 1 for detailed logs
        deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)

        # STEP 2: Read the audio file
        with open(AUDIO_FILE, "rb") as file:
            buffer_data = file.read()

        payload: BufferSource = {
            "buffer": buffer_data,
        }

        # STEP 3: Configure Deepgram options
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )

        # STEP 4: Call the API
        print("Sending audio file to Deepgram for transcription...")
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        
        # STEP 5: Print the results
        print("\n--- TRANSCRIPTION RESULTS ---")
        print(json.dumps(response.to_dict(), indent=4))
        print("---------------------------\n")


    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(f"Error: {e}")
        print("---------------------------\n")

if __name__ == "__main__":
    main()