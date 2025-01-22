from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Global variables
model = None
data = None
le = LabelEncoder()

# Define schema for prediction input
class PredictionInput(BaseModel):
    Machine_ID: str
    Coolant_Pressure: float
    Coolant_Temperature: float
    Hydraulic_Oil_Temperature: float
    Spindle_Bearing_Temperature: float
    Spindle_Speed: int


app = FastAPI()


#uploading dataset
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global data
    if file.filename.endswith(".csv"):
        data = pd.read_csv(file.file)
        return {"message": "upload successful", "columns": data.columns.tolist()}
    else:
        return {"file not uploaded"}

#training model
@app.post("/train")
async def train_model():
    global model, data, le
    if data is None:
        return {"no dataset uploaded"}
    
    data['Machine_ID'] = le.fit_transform(data['Machine_ID'])

    x= data.drop(columns=["Downtime"])
    y = data["Downtime"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test) #evaluating
    accuracy = accuracy_score(y_test, y_pred)
    import joblib
    joblib.dump(model, "model.pkl")  #saving the model
    joblib.dump(le, "label_encoder.pkl")

    return {"message": "Model trained successfully", "accuracy": accuracy}

# prediction 
import joblib
@app.post("/predict")
async def predict(input: PredictionInput):
    global model, le
    if model is None:
        if os.path.exists("model.pkl"):
            model = joblib.load("model.pkl")
            le = joblib.load("label_encoder.pkl")
        else:
            return {"error": "Model not trained. Please train the model first."}

    
    input_data = pd.DataFrame([{ #converting to pandas dataframe
        "Machine_ID": input.Machine_ID,
        "Coolant_Pressure": input.Coolant_Pressure,
        "Coolant_Temperature": input.Coolant_Temperature,
        "Hydraulic_Oil_Temperature": input.Hydraulic_Oil_Temperature,
        "Spindle_Bearing_Temperature": input.Spindle_Bearing_Temperature,
        "Spindle_Speed": input.Spindle_Speed
    }])

    input_data["Machine_ID"] = le.transform(input_data["Machine_ID"])
    prediction = model.predict(input_data)[0]
    confidence = max(model.predict_proba(input_data)[0])

    return {"Downtime": "Yes" if prediction == 1 else "No", "Confidence": round(confidence, 2)}

from fastapi.responses import RedirectResponse
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


if __name__ == "__main__": #running the fastapi app
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
