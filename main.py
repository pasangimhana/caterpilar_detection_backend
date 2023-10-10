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

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    response = {
        'class': predicted_class,
        'confidence': float(confidence),
    }
    return response

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)
