from fastapi import FastAPI, HTTPException
import database
import model_loader, utils, schemas

# ✅ Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# ✅ Load the latest production model
model = model_loader.load_production_model()

@app.get("/")
def home():
    """🏠 Welcome message."""
    return {"message": "Welcome to the Customer Churn Prediction API!"}

@app.get("/health")
def health_check():
    """💡 Health check endpoint."""
    return {"status": "API is running!"}

@app.post("/predict")
def predict_churn(data: schemas.ChurnInput):
    """🔮 Predicts if a customer will churn and stores in DB."""
    try:
        input_data = data.dict()

        # ✅ Preprocess input
        processed_data = utils.preprocess_input(input_data)

        # ✅ Predict churn
        prediction = model.predict(processed_data)
        churn_prediction = "Yes" if prediction[0] == 1 else "No"

        # ✅ Store data in DB
        database.save_customer_data(input_data, churn_prediction)

        return {"prediction": churn_prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ LLM-Based Churn Prediction from Customer Reviews
@app.post("/predict_review")
def predict_churn_from_review(data: schemas.LLMInput):
    """🔮 Predicts churn using customer feedback analyzed by an LLM."""
    try:
        user_feedback = data.user_feedback
        if not user_feedback:
            return {"llm_prediction": "❌ Please enter some feedback!"}

        user_feedback = data.user_feedback.strip()
        llm_response = model_loader.predict_churn_with_llm(user_feedback)
        
        # ✅ Extract Prediction & Reasoning
        if "Customer likely to leave" in llm_response:
            llm_prediction = "Customer likely to leave"
        else:
            llm_prediction = "Customer will stay"
        
        llm_reasoning = llm_response.split("\n")[-1].split("**")[-1].strip()  # Extract reasoning from the response
        # llm_reasoning = llm_reasoning.split("**")[-1].strip()
        # ✅ Store feedback in database
        utils.log_llm_response(user_feedback, llm_prediction, llm_reasoning)

        return {"llm_prediction": llm_prediction, "llm_reasoning": llm_reasoning}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))