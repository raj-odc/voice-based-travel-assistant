import os
import httpx
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv

# --- INITIALIZATION ---
load_dotenv()
app = FastAPI()

# Environment variables
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
VAPI_API_KEY = os.getenv("VAPI_API_KEY")

# Constants
DEFAULT_CITY = "Singapore"
WEATHER_API_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# --- TOOL/FUNCTION DEFINITION ---
async def get_weather(city: str = DEFAULT_CITY) -> str:
    """Fetches the current weather for a given city."""
    if not WEATHER_API_KEY:
        print("[ERROR] WEATHER_API_KEY is not set.")
        return "I can't fetch the weather right now; my API key is missing."

    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(WEATHER_API_BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            description = data['weather'][0]['description']
            temp = data['main']['temp']
            
            result = f"The current weather in {city} is {temp} degrees Celsius with {description}."
            print(f"Function result: '{result}'")
            return result
            
    except httpx.HTTPStatusError as e:
        error_msg = f"Sorry, I couldn't find the city {city}. Please check the spelling." if e.response.status_code == 404 else "Sorry, I had a problem fetching the weather."
        print(f"Weather API error: {str(e)}")
        return error_msg
    except Exception as e:
        print(f"Unexpected error fetching weather: {str(e)}")
        return "Sorry, an unexpected error occurred while fetching the weather."

# Function registry
TOOL_FUNCTIONS = {
    "getWeather": get_weather
}

# --- VAPI WEBHOOK ENDPOINT ---
@app.post("/")
async def vapi_webhook(request: Request):
    try:
        payload = await request.json()
        message = payload.get("message", {})
        results = []

        if message.get("type") != "tool-calls" or "toolCalls" not in message:
            return {"results": results}

        if not VAPI_API_KEY:
            raise HTTPException(status_code=500, detail="VAPI_API_KEY not found in environment")

        for tool_call in message["toolCalls"]:
            if tool_call.get("type") != "function":
                continue

            function_name = tool_call.get("function", {}).get("name")
            if function_name not in TOOL_FUNCTIONS:
                print(f"Warning: Unknown function {function_name}")
                continue

            try:
                args = tool_call["function"].get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                
                function = TOOL_FUNCTIONS[function_name]
                result = await function(**args) if args else await function()

                results.append({
                    "toolCallId": tool_call["id"],
                    "result": result
                })

            except Exception as e:
                print(f"Error executing {function_name}: {str(e)}")
                results.append({
                    "toolCallId": tool_call["id"],
                    "error": str(e)
                })

        return {"results": results}

    except Exception as e:
        print(f"Webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))