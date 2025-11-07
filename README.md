# 🚀 Interview AI Assistant

Complete AI-powered interview preparation assistant with multiple rounds support.

## 🎯 Features

### Interview Rounds:
- 💻 **Coding Round** - Algorithms, data structures, problem solving
- 🧠 **Technical Round** - Technology concepts, best practices  
- 🏗️ **System Design** - Architecture, scalability, trade-offs
- 👔 **HR Round** - Professional questions, company fit
- 🎭 **Behavioral Round** - STAR method, soft skills
- 🗄️ **Database Round** - SQL, optimization, design
- 🎨 **Frontend Round** - HTML, CSS, JavaScript, React
- ⚙️ **Backend Round** - APIs, servers, scalability
- 👨💻 **Code Review** - Best practices, optimization

## 🚀 Quick Start

### Local Development:
```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python -m uvicorn interview_app:app --port 8000

# Test all rounds
python test_all_rounds.py

# View API docs
http://localhost:8000/docs
```

### Example Usage:
```bash
# Coding Round
curl -X POST "http://localhost:8000/coding/" \\
  -H "Content-Type: application/json" \\
  -d '{"question": "Implement binary search", "difficulty": "medium", "max_tokens": 120}'

# HR Round  
curl -X POST "http://localhost:8000/hr/" \\
  -H "Content-Type: application/json" \\
  -d '{"question": "Why should we hire you?", "max_tokens": 100}'
```

## 🌐 Deploy to Render (Free)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Interview AI Assistant"
git remote add origin YOUR_GITHUB_REPO
git push -u origin main
```

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com)
2. Connect GitHub repository
3. Choose "Web Service"
4. Use these settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn interview_app:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3.11

### Step 3: Environment Variables (Optional)
- `HF_HOME=/tmp/huggingface` (for model caching)

## 📊 API Endpoints

| Endpoint | Description | Example |
|----------|-------------|---------|
| `/coding/` | Coding problems | Algorithm solutions |
| `/technical/` | Tech concepts | REST vs GraphQL |
| `/system-design/` | Architecture | Design Twitter |
| `/hr/` | HR questions | Why this company? |
| `/behavioral/` | Soft skills | Leadership examples |
| `/database/` | SQL & DB design | Indexing strategies |
| `/frontend/` | UI/UX concepts | React lifecycle |
| `/backend/` | Server concepts | Load balancing |
| `/code-review/` | Code analysis | Optimization tips |

## 🔧 Tech Stack
- **FastAPI** - Modern Python web framework
- **FLAN-T5** - Google's instruction-tuned model
- **Transformers** - Hugging Face library
- **Render** - Free cloud deployment

## 📝 Model Info
- **Model:** google/flan-t5-base (~1GB)
- **Training Data:** Up to 2021
- **Response Time:** 2-5 seconds
- **Memory Usage:** ~2GB RAM

## 🎉 Ready to Use!
Your AI interview assistant is ready to help with all rounds of technical interviews!