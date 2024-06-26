from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn 

from routes import router

app = FastAPI()

app.include_router(router=router)

@app.get("/")
async def return_root():
    message = "I'm alive!"
    return JSONResponse(content=message, status_code=200)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)