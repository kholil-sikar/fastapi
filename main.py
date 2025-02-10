from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI + TensorFlow!"}
