from pydantic import BaseModel

# ✅ Define request schema for structured input
class ChurnInput(BaseModel):
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    OnlineSecurity: int
    TechSupport: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

# ✅ Define LLM Schema
class LLMInput(BaseModel):
    user_feedback: str