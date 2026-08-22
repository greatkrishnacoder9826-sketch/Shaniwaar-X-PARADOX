from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import re
import string
from pydantic import BaseModel, Field
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os
import pickle
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np

model_path = "Shaniwaar/BiGRU_Model.keras"
tokenizer_path = "Shaniwaar/tokenizer.pkl"
max_sequence_length = 100
emotions_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
emotions_emogies = {
    "sadness": "😢",
    "joy": "😁",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😯"
}


def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join(text.split(' '))
    return text


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="The Sentence to Analyse",
                       json_schema_extra={"Example": "Appne Bhavnnaye Batao"})


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


dl_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer")
    dl_model['BiGRU'] = load_model(model_path)
    with open(tokenizer_path, 'rb') as file:
        dl_model['Tokenizer'] = pickle.load(file)
    print("Models are loaded successfully")

    yield

    dl_model.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/static', StaticFiles(directory='static'), name="static")


@app.get('/', include_in_schema=False)
def server_ui():
    return FileResponse('static/index.html')


@app.get('/health', response_model=HealthResponse)
def health_check():
    return HealthResponse(status="server is running", model_loaded=bool(dl_model))


@app.post('/predict', response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again")

    cleaned_text = clean(text_input.text)
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])

    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    probabilities = BiGRU_model.predict(padded_sequence)[0]
    all_probabilities = {
        label: float(prob) for prob, label in zip(probabilities, emotions_labels)
    }
    top_emotion_index = int(np.argmax(probabilities))

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=emotions_labels[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        all_probabilities=all_probabilities
    )