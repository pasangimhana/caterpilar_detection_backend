from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from pydantic import BaseModel, EmailStr
import bcrypt
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import uvicorn 
from datetime import datetime
from sqlalchemy import DateTime
import pytz




# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
#DATABASE_URL = "mysql+mysqlconnector://root:1234@localhost:3306/butterfly"
DATABASE_URL = "mysql+mysqlconnector://butterflydb:1080#mike@butterflydb.mysql.database.azure.com:3306/butterfly?ssl_ca=./model/DigiCertGlobalRootCA.crt.pem"
#DATABASE_URL = "mysql+mysqlconnector://butterflydb:123%40gmail.com@butterflydb.mysql.database.azure.com:3306/butterfly?ssl_ca=./path_to_your_cert/BaltimoreCyberTrustRoot.crt.pem"





Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_sri_jayawardenapura_kotte_time():
    utc_now = datetime.now(pytz.utc)
    sri_jayawardenapura_kotte_time = utc_now.astimezone(pytz.timezone('Asia/Colombo'))
    return sri_jayawardenapura_kotte_time

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

Base.metadata.create_all(bind=engine)

class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    location = Column(String, index=True)
    detection_class = Column(String, index=True)
    time = Column(DateTime, default=get_sri_jayawardenapura_kotte_time)



class HistoryCreate(BaseModel):
    location: str
    detection_class: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utility functions for password hashing
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Pydantic models for API requests/responses
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    confirm_password: str

# API Routes
@app.post("/register")
def register(user: UserCreate, db: SessionLocal = Depends(get_db)):
    # Check if passwords match
    if user.password != user.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    # Check if user already exists
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash the password and store the user
    hashed_password = hash_password(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()

    return {"message": "User successfully registered"}

# TensorFlow and Image processing setup
CLASS_NAMES = ["Common_Indian_Crow", "Common_Jay", "Common_Mime_Swallowtail", 
               "Common_Rose", "Cylon_Blue_Glass_Tiger","Great_eggfly","Lemon_Pansy","Tailed_Jay"]
MODEL = tf.keras.models.load_model('./model/trained_model2.h5')

@app.get("/ping")
def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        image = image.resize((100, 100))
        image = np.array(image)
        image = image.astype('float32') / 255.0
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # Debug information
    print(f"Input data shape: {img_batch.shape}, dtype: {img_batch.dtype}")
    print(f"Model's expected input shape: {MODEL.input_shape}")

    predictions = MODEL.predict(img_batch)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Check against the confidence_threshold
    if confidence < 0.7:
        predicted_class = "Unknown Object"
    else:
        predicted_class = CLASS_NAMES[predicted_class_index]

    response = {
        'class': predicted_class,
        'confidence': float(confidence),
    }

    return response

@app.get("/history")
def get_history(skip: int = 0, limit: int = 10, db: SessionLocal = Depends(get_db)):
    # Retrieve the latest `limit` history entries, skipping the first `skip` entries.
    history = db.query(History).offset(skip).limit(limit).all()
    return history

@app.delete("/history/{history_id}")
def delete_history(history_id: int, db: SessionLocal = Depends(get_db)):
    history_entry = db.query(History).filter(History.id == history_id).first()

    if not history_entry:
        raise HTTPException(status_code=404, detail="History entry not found")

    db.delete(history_entry)
    db.commit()
    return {"message": "History entry successfully deleted"}


@app.post("/login")
def login(user_login: UserLogin, db: SessionLocal = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user_login.email).first()
    
    if not db_user or not verify_password(user_login.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    return {"message": "Login successful!"}

@app.post("/save-history")
def save_history(history: HistoryCreate, db: SessionLocal = Depends(get_db)):
    try:
        history_entry = History(location=history.location, detection_class=history.detection_class)
        db.add(history_entry)
        db.commit()
        return {"message": "History successfully saved"}
    except Exception as e:
        print(f"Error saving history: {e}")
        raise HTTPException(status_code=500, detail="Error saving history")
    



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)
