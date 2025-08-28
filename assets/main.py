from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import joblib
import PyPDF2
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model=joblib.load("resume_role_model.pkl")
tfidf=joblib.load("tfidf_vectorizer.pkl")
le=joblib.load("label_encoder.pkl")

stop_words=set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()

#preprocessing
def clean(txt:str):
    txt = txt.lower()
    txt = re.sub(r'http\S+\s', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub(r'#\S+', ' ', txt)
    txt = re.sub(r'[^A-Za-z\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def final_words(txt:str):
    words=txt.split()
    words=[lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


#extract pdf text
def extract_text(file):
    pdf_reader= PyPDF2.PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text= text+page.extract_text()
    return text

#endpoint
app=FastAPI()
@app.get("/")
def home():
    return {"message": "Resume Role Screening API"}

@app.post("/predict_role/")

async def predict_role(file: UploadFile=File(...)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse(content={"error":"Only PDF files are allowed"},status_code=400)
    
    try:
        text=extract_text(file.file)

        text=clean(text)
        text=final_words(text)

        vectorized=tfidf.transform([text])
        prediction=model.predict(vectorized)[0]
        role=le.inverse_transform([prediction])[0]

        return {"predicted_role":role}
    
    except Exception as e:
        return JSONResponse(content={"error":str(e)},status_code=500)
    