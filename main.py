import json
from typing import Union
from fastapi import FastAPI
from routers import predict, chat, news, auth, profile


#uvicorn main:app --reload
#uvicorn main:app --host 0.0.0.0 --port 8000
#hcsp_1_5g
#http://192.168.50.110:8000/docs

#cloud
#uvicorn main:app --host 127.0.0.1 --port 8000
#cloudflared tunnel run fastapi-laptop

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Fake News Detection API"}

app.include_router(predict.router)
app.include_router(chat.router)
app.include_router(news.router)
app.include_router(auth.router)
app.include_router(profile.router)




