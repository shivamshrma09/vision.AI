from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import Dict, Optional, List
import uuid
import random
import time
import re

app = FastAPI(title="Professional Interview System", version="2.0.0")

try:
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
except Exception as e:
    print(f"Model loading failed: {e}")
    generator = None

interview_sessions = {}

class InterviewRequest(BaseModel):
    session_id: Optional[str] = None
    round_type: str
    user_answer: Optional[str] = None
    duration_minutes: Optional[int] = 45
    resume_text: Optional[str] = None

def extract_resume_info(resume_text: str) -> Dict:
    if not resume_text:
        return {}
    
    text = resume_text.lower()
    
    skills = []
    skill_patterns = [
        r'python|java|javascript|react|node|angular|vue|django|flask|spring',
        r'aws|azure|gcp|docker|kubernetes|jenkins|git|mongodb|mysql|postgresql',
        r'machine learning|ai|data science|tensorflow|pytorch|pandas|numpy'
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text)
        skills.extend(matches)
    
    projects = []
    project_indicators = ['project', 'built', 'developed', 'created', 'implemented']
    lines = resume_text.split('\n')
    
    for line in lines:
        if any(indicator in line.lower() for indicator in project_indicators):
            if len(line.strip()) > 20:
                projects.append(line.strip())
    
    experience_years = 0
    exp_match = re.search(r'(\d+)\s*(?:years?|yrs?)', text)
    if exp_match:
        experience_years = int(exp_match.group(1))
    
    return {
        "skills": list(set(skills)),
        "projects": projects[:3],
        "experience_years": experience_years,
        "has_leadership": any(word in text for word in ['lead', 'manager', 'senior', 'architect']),
        "education": 'degree' in text or 'university' in text or 'college' in text
    }

def generate_resume_questions(resume_info: Dict, round_type: str) -> List[str]:
    questions = []
    
    if not resume_info:
        return questions
    
    if round_type == 'technical' and resume_info.get('projects'):
        for project in resume_info['projects'][:2]:
            questions.append(f"I see you worked on: '{project[:100]}...'. Can you walk me through the technical architecture and challenges you faced?")
            questions.append(f"What technologies did you use in this project and why did you choose them?")
        
        if resume_info.get('skills'):
            skills_str = ', '.join(resume_info['skills'][:5])
            questions.append(f"I notice you have experience with {skills_str}. Which of these do you consider your strongest skill and why?")
            questions.append(f"How do you stay updated with the latest developments in {resume_info['skills'][0] if resume_info['skills'] else 'technology'}?")
    
    elif round_type == 'hr' and resume_info.get('experience_years', 0) > 0:
        questions.append(f"With {resume_info['experience_years']} years of experience, what would you say has been your biggest professional growth area?")
        questions.append(f"Looking at your background, why are you interested in this particular role and company?")
        questions.append(f"Based on your experience, where do you see yourself in the next 3-5 years?")
        
        if resume_info.get('has_leadership'):
            questions.append("I see you have leadership experience. What leadership style do you prefer and why?")
    
    elif round_type == 'behavioral':
        if resume_info.get('projects'):
            for project in resume_info['projects'][:1]:
                questions.append(f"Tell me about a specific challenge you encountered while working on: '{project[:80]}...' and how you overcame it.")
        
        if resume_info.get('experience_years', 0) > 2:
            questions.append(f"In your {resume_info['experience_years']} years of experience, describe a time when you had to mentor or help a junior colleague.")
        
        if resume_info.get('has_leadership'):
            questions.append("I see you have leadership experience. Tell me about a time when you had to make a difficult decision that affected your team.")
            questions.append("Describe a situation where you had to manage conflicting priorities or team disagreements.")
        
        questions.append("Tell me about a time when you had to learn a new technology or skill quickly for a project.")
    
    return questions[:5]

def analyze_candidate_data(session_data: Dict) -> Dict:
    conversation = session_data.get("conversation", [])
    total_score = session_data.get("total_score", 0)
    question_count = session_data.get("question_count", 1)
    
    avg_score = total_score / question_count
    
    response_lengths = []
    technical_depth = 0
    communication_quality = 0
    
    for qa in conversation:
        if qa.get("user_answer"):
            answer = qa["user_answer"]
            response_lengths.append(len(answer.split()))
            
            tech_indicators = ["algorithm", "architecture", "design", "performance", "scalability", "database", "api"]
            technical_depth += sum(1 for word in tech_indicators if word.lower() in answer.lower())
            
            if len(answer.split()) > 50:
                communication_quality += 1
    
    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    
    personality_traits = {
        "analytical_thinking": technical_depth > 5,
        "detailed_communicator": avg_response_length > 60,
        "concise_communicator": avg_response_length < 30,
        "experienced_professional": avg_score > 75,
        "growth_oriented": "learn" in str(conversation).lower() or "improve" in str(conversation).lower()
    }
    
    skills_mentioned = []
    skill_keywords = {
        "programming": ["python", "java", "javascript", "react", "node", "angular"],
        "databases": ["sql", "mongodb", "postgresql", "mysql", "redis"],
        "cloud": ["aws", "azure", "gcp", "docker", "kubernetes"],
        "methodologies": ["agile", "scrum", "devops", "ci/cd", "testing"]
    }
    
    conversation_text = str(conversation).lower()
    for category, keywords in skill_keywords.items():
        mentioned = [skill for skill in keywords if skill in conversation_text]
        if mentioned:
            skills_mentioned.extend(mentioned)
    
    return {
        "performance_metrics": {
            "average_score": round(avg_score, 1),
            "total_questions": question_count,
            "response_consistency": "High" if response_lengths and max(response_lengths) - min(response_lengths) < 50 else "Medium",
            "technical_depth_score": technical_depth,
            "communication_score": communication_quality
        },
        "candidate_insights": {
            "personality_traits": personality_traits,
            "communication_style": "Detailed" if avg_response_length > 60 else "Concise" if avg_response_length < 30 else "Balanced",
            "technical_competency": "Advanced" if technical_depth > 8 else "Intermediate" if technical_depth > 4 else "Basic",
            "experience_level": "Senior" if avg_score > 85 else "Mid-level" if avg_score > 70 else "Junior"
        },
        "skills_identified": {
            "mentioned_technologies": list(set(skills_mentioned)),
            "skill_categories": list(skill_keywords.keys()),
            "expertise_areas": [cat for cat, keywords in skill_keywords.items() if any(k in conversation_text for k in keywords)]
        },
        "hiring_insights": {
            "cultural_fit": "High" if personality_traits["growth_oriented"] else "Medium",
            "technical_readiness": "Ready" if avg_score > 75 else "Needs Development",
            "interview_performance": "Excellent" if avg_score > 85 else "Good" if avg_score > 70 else "Average",
            "recommendation_confidence": "High" if question_count > 8 else "Medium"
        }
    }

def evaluate_answer_quality(answer: str, question_type: str) -> Dict:
    if not answer or len(answer.strip()) < 10:
        return {"score": 20}
    
    words = answer.split()
    base_score = min(len(words) * 3, 60)
    
    technical_keywords = ["algorithm", "database", "api", "system", "performance", "architecture", "design", "scalability"]
    hr_keywords = ["experience", "team", "project", "challenge", "responsibility", "leadership", "collaboration", "growth"]
    
    if question_type in ["technical", "system_design"]:
        keyword_matches = sum(1 for kw in technical_keywords if kw.lower() in answer.lower())
    else:
        keyword_matches = sum(1 for kw in hr_keywords if kw.lower() in answer.lower())
    
    keyword_score = min(keyword_matches * 8, 40)
    total_score = min(base_score + keyword_score, 100)
    
    return {"score": total_score}

def get_next_question(session_data: Dict, round_type: str) -> str:
    start_time = session_data.get("start_time")
    duration_seconds = session_data.get("duration_minutes", 45) * 60
    
    if start_time and (time.time() - start_time) >= duration_seconds:
        return "END_INTERVIEW"
    
    question_banks = {
        "technical": [
            "Tell me about yourself and your technical background.",
            "What programming languages are you most comfortable with and why?",
            "Explain the difference between REST and GraphQL APIs.",
            "How do you approach debugging a complex issue in production?",
            "What is your experience with database design and optimization?",
            "Describe a challenging technical problem you solved recently.",
            "How do you ensure code quality in your projects?",
            "What are the principles of SOLID design patterns?",
            "Explain microservices architecture and its trade-offs.",
            "How do you handle version control and code reviews?"
        ],
        "hr": [
            "Tell me about yourself and your professional journey.",
            "Why are you interested in this position and our company?", 
            "What are your greatest strengths and how do they benefit a team?",
            "Where do you see yourself professionally in the next 3-5 years?",
            "How do you handle stress and tight deadlines?",
            "What motivates you in your work and career?",
            "Describe your ideal work environment and team culture.",
            "What areas are you looking to improve or develop further?",
            "Why are you considering leaving your current position?",
            "How do you approach learning new skills and technologies?"
        ],
        "behavioral": [
            "Tell me about a time you faced a significant technical challenge.",
            "Describe a situation where you had to work with a difficult colleague.",
            "Give an example of when you had to learn a new technology quickly.",
            "Tell me about a time you made a mistake and how you handled it.",
            "Describe a situation where you demonstrated leadership skills.",
            "Tell me about a time you had to meet an impossible deadline.",
            "Describe when you had to convince others to adopt your solution.",
            "Give an example of going above and beyond your job requirements.",
            "Tell me about handling ambiguous or changing requirements.",
            "Describe working effectively with limited resources or budget."
        ],
        "system_design": [
            "Design a URL shortener service like bit.ly with high availability.",
            "Design a real-time chat application supporting millions of users.", 
            "Design a social media feed system like Twitter or Instagram.",
            "Design a video streaming platform like YouTube or Netflix.",
            "Design an e-commerce platform like Amazon with global scale.",
            "Design a ride-sharing service like Uber with real-time matching.",
            "Design a food delivery system like DoorDash with order tracking.",
            "Design a distributed cache system like Redis for high performance.",
            "Design a search engine with web crawling and indexing.",
            "Design a content delivery network for global distribution."
        ]
    }
    
    asked_questions = [qa.get("question", "") for qa in session_data.get("conversation", [])]
    
    resume_questions = session_data.get("resume_questions", [])
    available_resume = [q for q in resume_questions if q not in asked_questions]
    
    if available_resume:
        return random.choice(available_resume)
    
    available_standard = [q for q in question_banks.get(round_type, []) if q not in asked_questions]
    
    if not available_standard:
        return "END_INTERVIEW"
    
    return random.choice(available_standard)

@app.post("/interview/")
async def conduct_interview(request: InterviewRequest):
    if not request.session_id:
        session_id = str(uuid.uuid4())
        
        resume_questions = []
        if request.resume_text:
            resume_info = extract_resume_info(request.resume_text)
            resume_questions = generate_resume_questions(resume_info, request.round_type)
        
        interview_sessions[session_id] = {
            "round_type": request.round_type,
            "conversation": [],
            "total_score": 0,
            "question_count": 0,
            "start_time": time.time(),
            "duration_minutes": request.duration_minutes,
            "resume_text": request.resume_text,
            "resume_questions": resume_questions
        }
    else:
        session_id = request.session_id
        if session_id not in interview_sessions:
            return {"error": "Session not found"}
    
    session_data = interview_sessions[session_id]
    
    if request.user_answer and len(session_data["conversation"]) > 0:
        current_qa = session_data["conversation"][-1]
        evaluation = evaluate_answer_quality(request.user_answer, request.round_type)
        
        current_qa.update({
            "user_answer": request.user_answer,
            "score": evaluation["score"]
        })
        
        session_data["total_score"] += evaluation["score"]
        session_data["question_count"] += 1
        
        elapsed = time.time() - session_data["start_time"]
        if elapsed >= (session_data.get("duration_minutes", 45) * 60):
            candidate_analysis = analyze_candidate_data(session_data)
            avg_score = session_data["total_score"] / session_data["question_count"]
            
            completion_prompt = f"""
            Professional Interview Assessment Report
            
            Candidate Performance Summary:
            - Interview Type: {request.round_type.title()}
            - Duration: {session_data.get('duration_minutes', 45)} minutes
            - Questions: {session_data['question_count']}
            - Average Score: {round(avg_score, 1)}/100
            - Technical Depth: {candidate_analysis['performance_metrics']['technical_depth_score']}
            - Communication Style: {candidate_analysis['candidate_insights']['communication_style']}
            
            Generate professional completion message with hiring recommendation.
            """
            
            try:
                if generator:
                    completion_response = generator(completion_prompt, max_length=200, temperature=0.7)
                    professional_message = completion_response[0]['generated_text']
                else:
                    raise Exception("Generator not available")
            except:
                professional_message = f"Interview completed successfully. Candidate demonstrated {candidate_analysis['candidate_insights']['experience_level'].lower()} level competency with {candidate_analysis['hiring_insights']['interview_performance'].lower()} performance. Recommendation: {candidate_analysis['hiring_insights']['technical_readiness']}."
            
            return {
                "session_id": session_id,
                "action": "interview_completed",
                "message": "Interview Assessment Complete",
                "professional_summary": professional_message,
                "candidate_analysis": candidate_analysis,
                "final_metrics": {
                    "duration_minutes": session_data.get("duration_minutes", 45),
                    "questions_answered": session_data["question_count"],
                    "average_score": round(avg_score, 1),
                    "performance_grade": "A" if avg_score > 90 else "B" if avg_score > 80 else "C" if avg_score > 70 else "D",
                    "interview_round": request.round_type.title(),
                    "completion_status": "Successfully Completed",
                    "hiring_recommendation": candidate_analysis['hiring_insights']['technical_readiness']
                }
            }
        
        next_question = get_next_question(session_data, request.round_type)
        
        if next_question == "END_INTERVIEW":
            candidate_analysis = analyze_candidate_data(session_data)
            avg_score = session_data["total_score"] / session_data["question_count"]
            return {
                "session_id": session_id,
                "action": "interview_completed",
                "message": "All questions completed",
                "candidate_analysis": candidate_analysis,
                "final_metrics": {
                    "total_questions": session_data["question_count"],
                    "average_score": round(avg_score, 1),
                    "performance_grade": "A" if avg_score > 90 else "B" if avg_score > 80 else "C" if avg_score > 70 else "D"
                }
            }
        
        session_data["conversation"].append({"question": next_question})
        
        return {
            "session_id": session_id,
            "question": next_question
        }
    
    else:
        first_question = get_next_question(session_data, request.round_type)
        session_data["conversation"].append({"question": first_question})
        
        resume_info = ""
        if request.resume_text:
            resume_info = f" I've analyzed your resume and will ask personalized questions about your experience."
        
        return {
            "session_id": session_id,
            "question": first_question,
            "round_type": request.round_type,
            "duration_minutes": request.duration_minutes,
            "resume_questions_generated": len(session_data.get("resume_questions", [])),
            "message": f"Welcome to {request.round_type} interview round. Duration: {request.duration_minutes} minutes.{resume_info} Let's begin!"
        }

@app.get("/analytics/{session_id}")
async def get_interview_analytics(session_id: str):
    if session_id not in interview_sessions:
        return {"error": "Session not found"}
    
    session_data = interview_sessions[session_id]
    candidate_analysis = analyze_candidate_data(session_data)
    
    return {
        "session_id": session_id,
        "interview_analytics": candidate_analysis,
        "raw_data": {
            "total_questions": session_data["question_count"],
            "conversation_length": len(session_data["conversation"]),
            "interview_duration": session_data.get("duration_minutes", 45),
            "round_type": session_data["round_type"],
            "resume_questions_used": len(session_data.get("resume_questions", []))
        }
    }

@app.get("/")
def home():
    return {
        "message": "Professional Interview System with Resume Analysis",
        "features": [
            "Resume-based personalized questions",
            "Industry-standard interview experience (45 minutes)",
            "Comprehensive question banks for all interview types", 
            "Advanced candidate data analysis and insights",
            "Professional HR-ready assessment reports",
            "Skills identification and personality profiling",
            "Hiring recommendation with confidence scoring"
        ],
        "usage_flow": {
            "1_start": "POST /interview/ with round_type, duration_minutes, and optional resume_text",
            "2_continue": "POST /interview/ with session_id and user_answer (repeat until time expires)",
            "3_completion": "Automatic comprehensive analysis with candidate insights",
            "4_analytics": "GET /analytics/{session_id} for detailed data analysis"
        },
        "resume_features": [
            "Automatic skill extraction from resume",
            "Project-based technical questions",
            "Experience-level appropriate questions",
            "Leadership and responsibility focused queries"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)