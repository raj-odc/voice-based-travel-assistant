import os
import io
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment

# --- INITIALIZATION ---
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Check if the API key is loaded
if not ELEVENLABS_API_KEY:
    print("[FATAL ERROR] ELEVENLABS_API_KEY not found in environment variables.")
    exit()

# Initialize the ElevenLabs client
try:
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    print("Successfully initialized ElevenLabs client.")
except Exception as e:
    print(f"[FATAL ERROR] Could not initialize ElevenLabs client: {e}")
    exit()

# The text we want to convert to speech
TEXT_TO_SPEAK = "Hello, this is a test of the audio generation and conversion pipeline."

# The name of the final output file
OUTPUT_FILENAME = "output.ulaw"

# Get available voices
try:
    voices = elevenlabs_client.voices.search()
    print("Available voices:")
    for voice in voices.voices:
        print(f"Name: {voice.name}, ID: {voice.voice_id}")
except Exception as e:
    print(f"[ERROR] Could not fetch voices: {e}")

def test_pipeline():
    """
    Simulates the full audio generation and conversion process.
    """
    print(f"--- STARTING AUDIO GENERATION TEST ---")
    print(f"1. AI Text: '{TEXT_TO_SPEAK}'")

    # Step 1: Generate high-quality MP3 audio from ElevenLabs
    try:
        print("2. Calling ElevenLabs to generate MP3 audio stream...")
        audio_stream = elevenlabs_client.text_to_speech.stream(
            text=TEXT_TO_SPEAK,
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Adam - a pre-made voice
            model_id="eleven_turbo_v2"
        )
        print("3. ElevenLabs stream received.")
    except Exception as e:
        print(f"[ERROR] ElevenLabs API call failed: {e}")
        return

    # Step 2: Collect the MP3 audio chunks into a single byte string
    try:
        print("4. Collecting MP3 audio chunks into memory...")
        mp3_data = b"".join([chunk for chunk in audio_stream])
        print(f"5. MP3 audio collected. Total size: {len(mp3_data)} bytes.")
        if len(mp3_data) == 0:
            print("[ERROR] MP3 data from ElevenLabs is empty. Aborting.")
            return
    except Exception as e:
        print(f"[ERROR] Failed to collect audio chunks: {e}")
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
        # This is the most likely point of failure if ffmpeg is not installed
        print(f"[ERROR] Pydub conversion failed: {e}")
        print("\n[HINT] This error often means 'ffmpeg' is not installed or not in your system's PATH.")
        print("[HINT] On Mac: 'brew install ffmpeg'. On Linux: 'sudo apt-get install ffmpeg'.")
        return
        
    # Step 4: Save the final converted audio to a file
    try:
        with open(OUTPUT_FILENAME, "wb") as f:
            f.write(output_bytes)
        print(f"10. Successfully saved converted audio to '{OUTPUT_FILENAME}'.")
    except Exception as e:
        print(f"[ERROR] Failed to write output file: {e}")
        return

    print("--- TEST COMPLETE: SUCCESS! ---")

# Run the main function
if __name__ == "__main__":
    test_pipeline()