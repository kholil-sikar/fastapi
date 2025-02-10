from fastapi import FastAPI, HTTPException, Request # type: ignore
from pydantic import BaseModel # type: ignore

# Import Bot
import json
import string
import random
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pickle

# adding check ulang similarity 
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import numpy as np # type: ignore
import re

import json
import random
import asyncio

# image chat
import tensorflow as tf # type: ignore
import base64
from io import BytesIO
from PIL import Image # type: ignore
from tensorflow import keras # type: ignore

# FastAPI instance
app = FastAPI()

# Load models and tokenizers
model1 = load_model('models/chatbot1_model1_non_sfa.h5')
le1 = pickle.load(open('knowledge/nltk/nltk1_non_sfa/le1_non_sfa.pkl', 'rb'))
tokenizer1 = pickle.load(open('knowledge/nltk/nltk1_non_sfa/tokenizers1_non_sfa.pkl', 'rb'))
with open('knowledge/brain1_non_sfa.json') as content:
    data1 = json.load(content)
responses1 = {intent['tag']: intent['responses'] for intent in data1['intents']}

model2 = load_model('models/chatbot2_model2_sfa.h5')
le2 = pickle.load(open('knowledge/nltk/nltk2_sfa/le2_sfa.pkl', 'rb'))
tokenizer2 = pickle.load(open('knowledge/nltk/nltk2_sfa/tokenizers2_sfa.pkl', 'rb'))
with open('knowledge/brain2_sfa.json') as content:
    data2 = json.load(content)
responses2 = {intent['tag']: intent['responses'] for intent in data2['intents']}

input_shape1 = model1.input_shape[1]
input_shape2 = model2.input_shape[1]

# ====================== MODEL IMAGES =======================
modelImages = keras.models.load_model('models/chatbot3_model3_sfa_images.keras')  # or 'my_model.keras'

class_names = [
    'input_setoran_bank_muncul_pesan_internal_server_error',
    'ketika_login_hh_terdapat_info_salesman_belum_eod',
    'ketika_salesman_print_nota_penjualan_muncul_keluar_nota_introdeal',
    'lupa_loading_stock',
    'salesman_lupa_input_km_akhir',
    'user_teritorial_tidak_ditemukan',
    'userid_belum_terdaftar_di_handheld'
]

# Request model for image classification
class ImageRequest(BaseModel):
    image: str  # base64 encoded image

# Home route
@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI TensorFlow Image Classifier"}

def remove_specific_words(question):
    # Daftar kata yang ingin dihapus (dalam bentuk regex)
    words_to_remove = r"\b(IT|pa it|Mas IT|Bpk IT|Bpkit|ms it|mas it|pk it|pakit|Mas IT,|Mas IT\.|Pak IT|Pak IT,|Pak IT\.|Bapak IT|Bapak IT,|Bapak IT\.|Tim IT)\b"
    
    # Menghapus kata-kata tersebut dari pertanyaan
    cleaned_question = re.sub(words_to_remove, "", question, flags=re.IGNORECASE)
    
    # Menghapus spasi berlebih yang mungkin tertinggal
    cleaned_question = " ".join(cleaned_question.split())
    
    return cleaned_question

def check_pattern_similarity(question, patterns):
    all_texts = [question] + patterns
    
    vectorizer = CountVectorizer().fit_transform(all_texts)
    vectors = vectorizer.toarray()
    
    similarities = cosine_similarity(vectors[0:1], vectors[1:])
    
    max_similarity = np.max(similarities)
    return max_similarity * 100

def predict_response(question):
    # Preprocess input
    clean_question = ''.join([letters.lower() for letters in question if letters not in string.punctuation])
    
    # === Model 1 Prediction ===
    seq_model1 = tokenizer1.texts_to_sequences([clean_question])
    seq_model1 = pad_sequences(seq_model1, maxlen=input_shape1)
    output_model1 = model1.predict(seq_model1)
    output_tag_model1 = le1.inverse_transform([output_model1.argmax()])[0]
    prediction_probability_model1 = output_model1[0][output_model1.argmax()]
    prediction_percentage_model1 = round(prediction_probability_model1 * 100, 2)

    print(f"Model 1: {output_tag_model1}, {prediction_percentage_model1}% confidence")

    # === Model 2 Prediction ===
    seq_model2 = tokenizer2.texts_to_sequences([clean_question])
    seq_model2 = pad_sequences(seq_model2, maxlen=input_shape2)
    output_model2 = model2.predict(seq_model2)
    output_tag_model2 = le2.inverse_transform([output_model2.argmax()])[0]
    prediction_probability_model2 = output_model2[0][output_model2.argmax()]
    prediction_percentage_model2 = round(prediction_probability_model2 * 100, 2)

    print(f"Model 2: {output_tag_model2}, {prediction_percentage_model2}% confidence")

    # Ensure valid responses
    response1 = responses1.get(output_tag_model1, ["Maaf, saya tidak mengerti."])
    response2 = responses2.get(output_tag_model2, ["Maaf, saya tidak mengerti."])

    selected_tag = ''
    selected_response = ''
    cleaned_question = remove_specific_words(question)
    croshcek_patterns = []
    similarity = 0

    if prediction_percentage_model1 > prediction_percentage_model2:
        selected_response = random.choice(response1)
        selected_tag = output_tag_model1
        croshcek_patterns = next(intent["patterns"] for intent in data1["intents"] if intent["tag"] == selected_tag)
        similarity = int(check_pattern_similarity(cleaned_question, croshcek_patterns))

        if prediction_percentage_model2 > 95:
            if selected_tag == 'kendala_printer_bluetoth_simcard_jaringan_ke_tim_ts':
                selected_response = random.choice(response2)
            else:
                selected_response = random.choice(response2) + '\nhttps://sfa.wismilak.com/faq/preview'

            selected_tag = output_tag_model2
        elif prediction_percentage_model2 < 95 and similarity < 90: 
            selected_response = "Mohon Maaf, saya belum paham apa yang dimaksud, apakah ada pertanyaan lain?"
            selected_tag = "Diluar Knowledge"
        else:
            print('')

    else:
        selected_response = random.choice(response2) + '\nhttps://sfa.wismilak.com/faq/preview'
        selected_tag = output_tag_model2
        croshcek_patterns = next(intent["patterns"] for intent in data2["intents"] if intent["tag"] == selected_tag)
        similarity = int(check_pattern_similarity(cleaned_question, croshcek_patterns))

        if prediction_percentage_model2 > 95:
            if selected_tag == 'kendala_printer_bluetoth_simcard_jaringan_ke_tim_ts':
                selected_response = random.choice(response2)
            else:
                if prediction_percentage_model2 > 97:
                    selected_response = random.choice(response2) + '\nhttps://sfa.wismilak.com/faq/preview'
                elif similarity > 50:
                    selected_response = random.choice(response2) + '\nhttps://sfa.wismilak.com/faq/preview'
                else: 
                    selected_response = "Mohon Maaf, saya belum paham apa yang dimaksud, apakah ada pertanyaan lain?"
                    selected_tag = "Diluar Knowledge"

            selected_tag = output_tag_model2
        elif prediction_percentage_model2 < 95 and similarity < 90: 
            selected_response = "Mohon Maaf, saya belum paham apa yang dimaksud, apakah ada pertanyaan lain?"
            selected_tag = "Diluar Knowledge"
        else:
            print('')
    
    print(f"🤖 Citra-Ai: : \n{selected_response},  ({selected_tag}),  {similarity}%")

    if selected_tag == 'kendala_printer_bluetoth_simcard_jaringan_ke_tim_ts':
        return {
            'tag': 'ts',
            'status': 'sfa',
            'respons': selected_response
        }
    else:
        if (output_tag_model1 == selected_tag):
            return {
                'tag': 'appdev',
                'status': 'non_sfa',
                'respons': selected_response
            }
        elif (selected_tag == 'Diluar Knowledge'):
            return {
                'tag': 'other',
                'status': 'non_sfa',
                'respons': selected_response
            }
        else:
            return {
                'tag': 'appdev',
                'status': 'sfa',
                'respons': selected_response
            }

async def predict_mobile(question):    
    print('')
    print('==================== Start Conversation ====================')
    print(f"User 😐 : {question}")

    try:
        answering_predict = ''
        if re.search(r'\bhelp\b', question, re.IGNORECASE):
            answering_predict = (
                'Halo! Saya Citra AI 🤖✨ \n'
                '(Chatbot Informatif dan Respon Aktif) \n\n'
                'Saya siap membantu Anda, Salesman/Admin, dalam menjawab pertanyaan seputar informasi FAQ di SFA. '
                'Cukup ketik pertanyaan Anda, dengan keyword "Pak IT/Bapak IT, Mas IT, IT" atau Tag Saya beserta langsung pertanyaannya '
                'nantinya saya akan memberikan jawaban yang cepat dan akurat 😊 \n\n'
                'Contoh: Pak it saya lupa loading stock hari kemarin, gimana ya pak solusinya? '
            )
            return {
                'tag': 'rule',
                'status': 'non_sfa',
                'respons': answering_predict
            }
        else:
            answering_predict = predict_response(question)

        return answering_predict
    
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Maaf, saya tidak bisa menjawab saat ini."

    
async def predict_images(question):
    answering_predict = predict_response(question)

    return {
        'tag': 'images',
        'status': 'sfa',
        'respons': answering_predict['respons']
    }


# =============== API ===============
@app.post("/api-txt")
async def handle_message_txt(request: Request):
    data = await request.json()  # Ambil JSON dari request
    received_text = data.get('text', '')

    result = await predict_mobile(received_text)
    return result

# Image classification route
@app.post("/api-img")
async def handle_message_img(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(BytesIO(image_data))
        img = tf.keras.preprocessing.image.img_to_array(image.resize((150, 150)))
        img = np.expand_dims(img, axis=0)

        # Predict the class
        predictions = modelImages.predict(img)
        score = tf.nn.softmax(predictions[0])
        result_predict = class_names[np.argmax(score)]
        result_replace = result_predict.replace("_", " ")

        result = await (predict_images(result_replace))
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server using uvicorn:
# uvicorn main:app --host 127.0.0.1 --port 8000 --reload
