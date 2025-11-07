from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
app = FastAPI(title="Interview AI Assistant", version="1.0.0")

# Load model
generator = pipeline("text2text-generation", model="google/flan-t5-base")

class InterviewRequest(BaseModel):
    question: str
    difficulty: str = "medium"  # easy, medium, hard
    max_tokens: int = 150

class CodeReviewRequest(BaseModel):
    code: str
    language: str
    max_tokens: int = 100

@app.get("/")
def home():
    return {
        "message": "🚀 Interview AI Assistant Ready!", 
        "endpoints": [
            "/coding", "/technical", "/system-design", "/behavioral", 
            "/hr", "/database", "/frontend", "/backend", "/code-review"
        ]
    }

@app.post("/coding/")
def coding_round(data: InterviewRequest):
    prompt = f"Coding Interview Question ({data.difficulty}): {data.question}. Provide solution approach, algorithm, and code example."
    
    result = generator(prompt, max_length=data.max_tokens, do_sample=True, temperature=0.7)
    return {"type": "coding", "response": result[0]['generated_text']}

@app.post("/technical/")
def technical_round(data: InterviewRequest):
    prompt = f"Technical Interview Question: {data.question}. Explain concepts, provide examples, and discuss best practices."
    
    result = generator(prompt, max_length=data.max_tokens, do_sample=True, temperature=0.6)
    return {"type": "technical", "response": result[0]['generated_text']}

@app.post("/system-design/")
def system_design_round(data: InterviewRequest):
    prompt = f"System Design Question: {data.question}. Discuss architecture, scalability, database design, and trade-offs."
    
    result = generator(prompt, max_length=data.max_tokens, do_sample=True, temperature=0.8)
    return {"type": "system_design", "response": result[0]['generated_text']}

@app.post("/code-review/")
def code_review(data: CodeReviewRequest):
    prompt = f"Review this {data.language} code and suggest improvements: {data.code}"
    
    result = generator(prompt, max_length=data.max_tokens, do_sample=True, temperature=0.5)
    return {"type": "code_review", "response": result[0]['generated_text']}

@app.post("/mock-interview/")
def mock_interview(data: InterviewRequest):
    prompt = f"Act as interviewer. Ask follow-up questions for: {data.question}"
    
    result = generator(prompt, max_length=data.max_tokens, do_sample=True, temperature=0.9)
    return {"type": "mock_interview", "response": result[0]['generated_text']}

@app.post("/behavioral/")
def behavioral_round(data: InterviewRequest):
    prompt = f"Behavioral Interview Question: {data.question}. Provide STAR method approach and example answer."
    
    result = generator(prompt, max_length=data.max_tokens, do_sample=True, temperature=0.7)
    return {"type": "behavioral", "response": result[0]['generated_text']}

@app.post("/hr/")
def hr_round(data: InterviewRequest):
    prompt = f"HR Interview Question: {data.question}. Provide professional and thoughtful response."
    
    result = generator(prompt, max_length=data.max_tokens, do_sample=True, temperature=0.6)
    return {"type": "hr", "response": result[0]['generated_text']}

@app.post("/database/")
def database_round(data: InterviewRequest):
    prompt = f"Database Interview Question: {data.question}. Explain SQL concepts, optimization, and best practices."
    
    result = generator(prompt, max_length=data.max_tokens, do_sample=True, temperature=0.7)
    return {"type": "database", "response": result[0]['generated_text']}

@app.post("/frontend/")
def frontend_round(data: InterviewRequest):
    prompt = f"Frontend Interview Question: {data.question}. Discuss HTML, CSS, JavaScript, React, and UI/UX concepts."
    
    result = generator(prompt, max_length=data.max_tokens, do_sample=True, temperature=0.7)
    return {"type": "frontend", "response": result[0]['generated_text']}

@app.post("/backend/")
def backend_round(data: InterviewRequest):
    prompt = f"Backend Interview Question: {data.question}. Explain server-side concepts, APIs, databases, and scalability."
    
    result = generator(prompt, max_length=data.max_tokens, do_sample=True, temperature=0.7)
    return {"type": "backend", "response": result[0]['generated_text']}