from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import nest_asyncio
import os

if os.getenv("USE_NEST_ASYNCIO"):  # Use an environment variable to toggle
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())  # Disable uvloop
    nest_asyncio.apply()

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Hello World"}


# nest_asyncio.apply()
if __name__ == "__main__":
    uvicorn.run("app_api:app",host="127.0.0.1", port=8000)