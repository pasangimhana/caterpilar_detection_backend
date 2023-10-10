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
DATABASE_URL = "mysql+mysqlconnector://root:1234@localhost:3306/butterfly"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

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
    time = Column(DateTime, default=datetime.utcnow)

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
CLASS_NAMES = ["Common_Indian_Crow", "Crimson Rose", "Common Mormon", 
               "Common Mime Swallowtail", "Ceylon Blue Glassy Tiger"]
MODEL = tf.keras.models.load_model('./model/Model.h5')

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
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

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

@app.post("/login")
def login(user_login: UserLogin, db: SessionLocal = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user_login.email).first()
    
    if not db_user or not verify_password(user_login.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    return {"message": "Login successful!"}

@app.post("/save-history")
def save_history(history: HistoryCreate, db: SessionLocal = Depends(get_db)):
    history_entry = History(location=history.location, detection_class=history.detection_class)
    db.add(history_entry)
    db.commit()
    return {"message": "History successfully saved"}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)
