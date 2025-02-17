from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import uvicorn

# Load pre-trained model (Assuming a pickle file `model.pkl`)
MODEL_PATH = "model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Model Inference API is running"}

@app.post("/predict/")
def predict(features: list):
    try:
        # Convert input to NumPy array
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run API Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)