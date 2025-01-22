# FASTAPI-machine-downtime-pred-API
# FastAPI Machine Downtime Prediction API

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies**
   Ensure Python 3.8+ is installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   Start the server with:
   ```bash
   uvicorn app:app --reload
   ```

4. **Access Swagger UI**
   Open your browser and go to:
   ```
   http://127.0.0.1:8000/docs
   ```

---

## Example API Requests

### 1. **Upload Dataset**
- **Endpoint:** `POST /upload`
- **Request:**
  Upload a CSV file containing sensor data.

- **Response:**
  ```json
  {
      "message": "CSV file uploaded successfully",
      "columns": ["Machine_ID", "Coolant_Pressure", "Coolant_Temperature", "Hydraulic_Oil_Temperature", "Spindle_Bearing_Temperature", "Spindle_Speed", "Downtime"]
  }
  ```

### 2. **Train Model**
- **Endpoint:** `POST /train`
- **Request:**
  No additional input required.

- **Response:**
  ```json
  {
      "message": "Model trained successfully",
      "accuracy": 0.95
  }
  ```

### 3. **Predict Downtime**
- **Endpoint:** `POST /predict`
- **Request:**
  ```json
  {
      "Machine_ID": "Machine_01",
      "Coolant_Pressure": 1.5,
      "Coolant_Temperature": 35.2,
      "Hydraulic_Oil_Temperature": 50.1,
      "Spindle_Bearing_Temperature": 45.3,
      "Spindle_Speed": 1200
  }
  ```

- **Response:**
  ```json
  {
      "Downtime": "Yes",
      "Confidence": 0.92
  }
  ```


    "Downtime": "Yes",
    "Confidence": 0.92
}


