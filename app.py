from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import Dict, Optional, List
import uuid
import random
import time
import re

app = FastAPI(title="AI Interview System", version="2.0.0")

generator = None

def load_model():
    global generator
    if generator is None:
        try:
            generator = pipeline("text2text-generation", model="google/flan-t5-base")
        except Exception as e:
            print(f"Model loading failed: {e}")
            generator = None
    return generator

load_model()
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
            questions.append(f"I see you worked on: '{project[:100]}...'. Can you walk me through the technical architecture?")
        
        if resume_info.get('skills'):
            skills_str = ', '.join(resume_info['skills'][:5])
            questions.append(f"I notice you have experience with {skills_str}. Which is your strongest skill and why?")
    
    elif round_type == 'hr' and resume_info.get('experience_years', 0) > 0:
        questions.append(f"With {resume_info['experience_years']} years of experience, what has been your biggest growth area?")
        
        if resume_info.get('has_leadership'):
            questions.append("I see you have leadership experience. What leadership style do you prefer?")
    
    elif round_type == 'behavioral':
        if resume_info.get('projects'):
            for project in resume_info['projects'][:1]:
                questions.append(f"Tell me about a challenge you faced while working on: '{project[:80]}...'")
        
        if resume_info.get('has_leadership'):
            questions.append("Tell me about a difficult decision you made that affected your team.")
    
    return questions[:5]

def analyze_individual_answer(question: str, answer: str, question_type: str) -> Dict:
    if not answer:
        return {"error": "No answer provided"}
    
    words = answer.split()
    word_count = len(words)
    
    technical_keywords = ["algorithm", "architecture", "design", "performance", "scalability", "database", "api", "system"]
    soft_skills = ["team", "leadership", "communication", "collaboration", "problem-solving", "mentoring"]
    
    tech_mentions = [kw for kw in technical_keywords if kw.lower() in answer.lower()]
    soft_mentions = [kw for kw in soft_skills if kw.lower() in answer.lower()]
    
    has_examples = any(phrase in answer.lower() for phrase in ["for example", "such as", "like when", "instance"])
    has_metrics = any(char.isdigit() for char in answer)
    
    return {
        "question": question,
        "answer_length": word_count,
        "technical_depth": {
            "keywords_found": tech_mentions,
            "technical_score": len(tech_mentions),
            "depth_level": "High" if len(tech_mentions) > 3 else "Medium" if len(tech_mentions) > 1 else "Low"
        },
        "soft_skills": {
            "skills_mentioned": soft_mentions,
            "leadership_indicators": len(soft_mentions)
        },
        "answer_quality": {
            "has_examples": has_examples,
            "includes_metrics": has_metrics,
            "completeness": "Complete" if word_count > 50 and has_examples else "Partial" if word_count > 20 else "Brief",
            "communication_clarity": "Clear" if 10 < len(answer.split('.')) < 8 else "Complex"
        },
        "follow_up_potential": {
            "needs_clarification": word_count < 30,
            "can_dive_deeper": len(tech_mentions) > 0 or has_examples,
            "explore_experience": any(word in answer.lower() for word in ["project", "experience", "worked", "built"])
        }
    }

def decide_next_question_strategy(session_data: Dict, last_analysis: Dict) -> Dict:
    conversation = session_data.get("conversation", [])
    question_count = len([qa for qa in conversation if qa.get("user_answer")])
    
    if not last_analysis:
        return {"strategy": "new_question", "reason": "No previous analysis available"}
    
    follow_up_score = 0
    reasons = []
    
    if last_analysis.get("follow_up_potential", {}).get("needs_clarification"):
        follow_up_score += 3
        reasons.append("Answer too brief, needs elaboration")
    
    if last_analysis.get("follow_up_potential", {}).get("can_dive_deeper"):
        follow_up_score += 2
        reasons.append("Technical depth available for exploration")
    
    recent_followups = sum(1 for qa in conversation[-3:] if qa.get("is_followup", False))
    if recent_followups >= 2:
        follow_up_score -= 3
        reasons.append("Too many recent follow-ups, need variety")
    
    elapsed_time = time.time() - session_data.get("start_time", time.time())
    time_remaining = (session_data.get("duration_minutes", 45) * 60) - elapsed_time
    
    if time_remaining < 300 and question_count < 8:
        follow_up_score -= 2
        reasons.append("Limited time remaining, prioritize coverage")
    
    decision = "follow_up" if follow_up_score > 2 else "new_question"
    
    return {
        "strategy": decision,
        "confidence_score": abs(follow_up_score),
        "reasons": reasons,
        "time_remaining_minutes": round(time_remaining / 60, 1)
    }

def generate_alternative_answers(question: str, question_type: str) -> List[Dict]:
    alternatives = []
    
    if question_type == "technical":
        alternatives = [
            {
                "level": "Excellent",
                "sample_answer": "I have 5+ years of experience in full-stack development using Python, Django, and React. I've built scalable microservices handling 100K+ requests/day, implemented CI/CD pipelines, and led a team of 4 developers. For example, I architected an e-commerce platform that reduced page load time by 40% using Redis caching and database optimization.",
                "why_excellent": "Specific experience, metrics, leadership, concrete examples, technical depth"
            },
            {
                "level": "Good",
                "sample_answer": "I'm a software developer with experience in Python and web development. I've worked on several projects including APIs and databases. I enjoy solving complex problems and learning new technologies.",
                "why_good": "Shows experience and interest, but lacks specific examples and metrics"
            },
            {
                "level": "Needs Improvement",
                "sample_answer": "I know programming and have done some projects. I like coding.",
                "why_poor": "Too brief, vague, no specific technologies or examples mentioned"
            }
        ]
    
    elif question_type == "hr":
        alternatives = [
            {
                "level": "Excellent",
                "sample_answer": "I'm a passionate software engineer with 4 years of experience building scalable web applications. I thrive in collaborative environments and have successfully led cross-functional projects. I'm particularly drawn to this role because it combines my technical expertise with my interest in mentoring junior developers, and your company's focus on innovation aligns with my career goals.",
                "why_excellent": "Personal passion, specific experience, leadership examples, company research, career alignment"
            },
            {
                "level": "Good",
                "sample_answer": "I'm a software developer with several years of experience. I work well in teams and am interested in this position because it offers growth opportunities.",
                "why_good": "Shows teamwork and growth mindset, but lacks specificity and company knowledge"
            },
            {
                "level": "Needs Improvement",
                "sample_answer": "I need a job and this seems like a good opportunity.",
                "why_poor": "Shows no preparation, research, or genuine interest in the role"
            }
        ]
    
    elif question_type == "behavioral":
        alternatives = [
            {
                "level": "Excellent",
                "sample_answer": "In my previous role, we faced a critical production issue where our API was timing out, affecting 50% of users. I immediately assembled a cross-functional team, implemented monitoring to identify the root cause - a database query bottleneck. I proposed and implemented a caching solution that reduced response time by 80%. This experience taught me the importance of proactive monitoring and quick decision-making under pressure.",
                "why_excellent": "STAR method, specific situation, quantified impact, lessons learned, leadership"
            },
            {
                "level": "Good",
                "sample_answer": "I once had to fix a bug that was causing problems for users. I worked with my team to identify the issue and we fixed it. It was challenging but we managed to resolve it.",
                "why_good": "Shows problem-solving and teamwork, but lacks specific details and impact"
            },
            {
                "level": "Needs Improvement",
                "sample_answer": "I haven't really faced any major challenges. Things usually go smoothly.",
                "why_poor": "Doesn't answer the question, shows lack of experience or self-awareness"
            }
        ]
    
    return alternatives

def generate_personalized_feedback(user_answer: str, question: str, question_type: str, score: int) -> Dict:
    analysis = analyze_individual_answer(question, user_answer, question_type)
    alternatives = generate_alternative_answers(question, question_type)
    
    if score >= 85:
        performance_level = "Excellent"
        feedback_tone = "Great job!"
    elif score >= 70:
        performance_level = "Good"
        feedback_tone = "Well done, with room for improvement."
    else:
        performance_level = "Needs Improvement"
        feedback_tone = "Consider strengthening your response."
    
    strengths = []
    improvements = []
    
    if analysis["answer_length"] > 40:
        strengths.append("Good detail level in your response")
    else:
        improvements.append("Provide more detailed explanations with specific examples")
    
    if analysis["answer_quality"]["has_examples"]:
        strengths.append("Excellent use of concrete examples")
    else:
        improvements.append("Include specific examples to illustrate your points")
    
    if analysis["technical_depth"]["technical_score"] > 2:
        strengths.append("Strong technical vocabulary and depth")
    elif question_type == "technical":
        improvements.append("Demonstrate deeper technical knowledge with specific technologies")
    
    if analysis["answer_quality"]["includes_metrics"]:
        strengths.append("Great use of quantifiable metrics")
    else:
        improvements.append("Include numbers and metrics to show impact (e.g., '40% improvement', '100+ users')")
    
    ai_feedback = ""
    if generator:
        try:
            feedback_prompt = f"""
            Provide constructive feedback for this {question_type} interview answer:
            Question: "{question[:100]}..."
            Answer: "{user_answer[:200]}..."
            Score: {score}/100
            
            Give specific, actionable advice for improvement.
            """
            
            response = generator(feedback_prompt, max_length=150, temperature=0.7)
            ai_feedback = response[0]['generated_text']
        except:
            ai_feedback = f"Your answer scored {score}/100. Focus on providing more specific examples and technical details."
    
    return {
        "performance_level": performance_level,
        "score": score,
        "feedback_summary": feedback_tone,
        "strengths": strengths,
        "areas_for_improvement": improvements,
        "ai_personalized_feedback": ai_feedback,
        "alternative_answers": alternatives,
        "improvement_tips": {
            "structure": "Use STAR method (Situation, Task, Action, Result) for behavioral questions",
            "specificity": "Include specific technologies, metrics, and examples",
            "impact": "Quantify your achievements with numbers and percentages",
            "learning": "Mention what you learned or how you grew from the experience"
        }
    }

def generate_ai_followup_question(previous_answer: str, round_type: str, analysis: Dict) -> str:
    if not generator or not previous_answer:
        return None
    
    try:
        tech_keywords = analysis.get("technical_depth", {}).get("keywords_found", [])
        completeness = analysis.get("answer_quality", {}).get("completeness", "Brief")
        
        if completeness == "Brief":
            prompt_focus = "ask for more details and specific examples"
        elif tech_keywords:
            prompt_focus = f"dive deeper into {', '.join(tech_keywords[:2])} mentioned"
        else:
            prompt_focus = "ask for specific examples and practical experience"
        
        prompt = f"""
        Based on this {round_type} interview answer: "{previous_answer[:200]}..."
        Generate a follow-up question to {prompt_focus}.
        Keep it professional and specific.
        """
        
        response = generator(prompt, max_length=100, temperature=0.8)
        followup = response[0]['generated_text'].strip()
        
        if '?' in followup:
            followup = followup.split('?')[0] + '?'
        
        return followup if len(followup) > 20 else None
    except Exception as e:
        print(f"AI followup generation failed: {e}")
        return None

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
    
    conversation = session_data.get("conversation", [])
    if len(conversation) > 0 and conversation[-1].get("user_answer"):
        last_qa = conversation[-1]
        last_answer = last_qa["user_answer"]
        last_question = last_qa.get("question", "")
        
        last_analysis = analyze_individual_answer(last_question, last_answer, round_type)
        strategy_decision = decide_next_question_strategy(session_data, last_analysis)
        
        session_data["last_decision"] = strategy_decision
        session_data["last_analysis"] = last_analysis
        
        if strategy_decision["strategy"] == "follow_up":
            ai_followup = generate_ai_followup_question(last_answer, round_type, last_analysis)
            if ai_followup:
                return {"question": ai_followup, "is_followup": True}
    
    question_banks = {
        "technical": [
            "Tell me about your technical background.",
            "What programming languages are you most comfortable with?",
            "Explain REST APIs and their benefits.",
            "How do you approach debugging complex issues?",
            "Describe a challenging technical problem you solved.",
            "What is your experience with database design?",
            "How do you ensure code quality?",
            "Explain microservices architecture.",
            "How do you handle version control?",
            "What are SOLID design principles?"
        ],
        "hr": [
            "Tell me about yourself.",
            "Why are you interested in this position?",
            "What are your greatest strengths?",
            "Where do you see yourself in 5 years?",
            "How do you handle pressure?",
            "What motivates you?",
            "Describe your ideal work environment.",
            "What areas need improvement?",
            "Why are you leaving your current job?",
            "How do you learn new skills?"
        ],
        "behavioral": [
            "Tell me about a challenge you faced.",
            "Describe working with difficult people.",
            "When did you learn something quickly?",
            "Tell me about a mistake you made.",
            "Describe a leadership situation.",
            "How did you meet a tight deadline?",
            "When did you convince others?",
            "Example of going above and beyond.",
            "How do you handle changing requirements?",
            "Working with limited resources."
        ]
    }
    
    asked_questions = [qa.get("question", "") for qa in conversation]
    resume_questions = session_data.get("resume_questions", [])
    available_resume = [q for q in resume_questions if q not in asked_questions]
    
    if available_resume:
        return random.choice(available_resume)
    
    available_standard = [q for q in question_banks.get(round_type, []) if q not in asked_questions]
    
    if not available_standard:
        return "END_INTERVIEW"
    
    return random.choice(available_standard)

def generate_final_comprehensive_report(session_data: Dict) -> Dict:
    conversation = session_data.get("conversation", [])
    total_score = session_data.get("total_score", 0)
    question_count = session_data.get("question_count", 1)
    
    avg_score = total_score / question_count
    
    question_analysis = []
    response_lengths = []
    technical_depth = 0
    
    for i, qa in enumerate(conversation):
        if qa.get("user_answer"):
            answer = qa["user_answer"]
            question = qa.get("question", "")
            
            individual_analysis = analyze_individual_answer(question, answer, session_data.get("round_type", "general"))
            individual_analysis["question_number"] = i + 1
            individual_analysis["score"] = qa.get("score", 0)
            individual_analysis["is_followup"] = qa.get("is_followup", False)
            individual_analysis["detailed_feedback"] = qa.get("detailed_feedback", {})
            individual_analysis["alternative_answers"] = generate_alternative_answers(question, session_data.get("round_type", "general"))
            
            question_analysis.append(individual_analysis)
            response_lengths.append(len(answer.split()))
            technical_depth += individual_analysis["technical_depth"]["technical_score"]
    
    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    scores = [qa.get("score", 0) for qa in conversation if qa.get("score")]
    performance_trend = "Improving" if len(scores) > 2 and scores[-1] > scores[0] else "Declining" if len(scores) > 2 and scores[-1] < scores[0] else "Stable"
    
    ai_summary = ""
    if generator:
        try:
            summary_prompt = f"""
            Generate professional interview summary:
            - Questions: {question_count}, Average Score: {round(avg_score, 1)}/100
            - Technical Depth: {technical_depth}, Trend: {performance_trend}
            - Experience Level: {"Senior" if avg_score > 85 else "Mid-level" if avg_score > 70 else "Junior"}
            Write concise HR assessment with hiring recommendation.
            """
            
            response = generator(summary_prompt, max_length=200, temperature=0.7)
            ai_summary = response[0]['generated_text']
        except:
            ai_summary = f"Candidate completed {question_count} questions with {round(avg_score, 1)}/100 average score. Performance trend: {performance_trend}. Recommendation: {'Hire' if avg_score > 75 else 'Consider' if avg_score > 65 else 'No Hire'}."
    
    return {
        "interview_summary": {
            "session_id": session_data.get("session_id"),
            "interview_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session_data.get("start_time", time.time()))),
            "interview_type": session_data["round_type"].title(),
            "duration_minutes": round((time.time() - session_data.get("start_time", time.time())) / 60, 1),
            "ai_generated_summary": ai_summary
        },
        "overall_performance": {
            "average_score": round(avg_score, 1),
            "performance_grade": "A" if avg_score > 90 else "B" if avg_score > 80 else "C" if avg_score > 70 else "D",
            "performance_trend": performance_trend,
            "technical_competency": "Advanced" if technical_depth > 15 else "Intermediate" if technical_depth > 8 else "Basic",
            "communication_style": "Detailed" if avg_response_length > 60 else "Concise" if avg_response_length < 30 else "Balanced",
            "overall_recommendation": "Strong Hire" if avg_score > 85 and technical_depth > 12 else "Hire" if avg_score > 75 else "Consider" if avg_score > 65 else "No Hire"
        },
        "question_by_question_analysis": question_analysis,
        "key_insights": {
            "strengths": [],
            "areas_for_improvement": [],
            "technical_highlights": [qa for qa in question_analysis if qa.get("technical_depth", {}).get("depth_level") == "High"],
            "communication_highlights": [qa for qa in question_analysis if qa.get("answer_quality", {}).get("completeness") == "Complete"]
        },
        "interview_statistics": {
            "total_questions": question_count,
            "followup_questions": len([qa for qa in conversation if qa.get("is_followup", False)]),
            "resume_questions_used": len(session_data.get("resume_questions", [])),
            "technical_depth_score": technical_depth,
            "score_progression": scores
        }
    }

@app.post("/interview/")
async def conduct_interview(request: InterviewRequest):
    if not request.session_id:
        session_id = str(uuid.uuid4())
        
        resume_questions = []
        if request.resume_text:
            resume_info = extract_resume_info(request.resume_text)
            resume_questions = generate_resume_questions(resume_info, request.round_type)
        
        interview_sessions[session_id] = {
            "session_id": session_id,
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
        
        feedback = generate_personalized_feedback(
            request.user_answer, 
            current_qa.get("question", ""), 
            request.round_type, 
            evaluation["score"]
        )
        
        current_qa.update({
            "user_answer": request.user_answer,
            "score": evaluation["score"],
            "detailed_feedback": feedback
        })
        
        session_data["total_score"] += evaluation["score"]
        session_data["question_count"] += 1
        
        elapsed = time.time() - session_data["start_time"]
        if elapsed >= (session_data.get("duration_minutes", 45) * 60):
            final_report = generate_final_comprehensive_report(session_data)
            
            return {
                "session_id": session_id,
                "action": "interview_completed",
                "message": "Interview completed successfully",
                "comprehensive_final_report": final_report
            }
        
        next_question = get_next_question(session_data, request.round_type)
        
        if next_question == "END_INTERVIEW":
            final_report = generate_final_comprehensive_report(session_data)
            
            return {
                "session_id": session_id,
                "action": "interview_completed",
                "message": "All questions completed",
                "comprehensive_final_report": final_report
            }
        
        if isinstance(next_question, dict):
            question_data = next_question
            session_data["conversation"].append(question_data)
            next_question = question_data["question"]
        else:
            session_data["conversation"].append({"question": next_question, "is_followup": False})
        
        return {
            "session_id": session_id,
            "question": next_question,
            "previous_answer_feedback": current_qa.get("detailed_feedback", {})
        }
    
    else:
        first_question = get_next_question(session_data, request.round_type)
        
        if isinstance(first_question, dict):
            question_data = first_question
            session_data["conversation"].append(question_data)
            first_question = question_data["question"]
        else:
            session_data["conversation"].append({"question": first_question, "is_followup": False})
        
        resume_info = ""
        if request.resume_text:
            resume_info = f" I've analyzed your resume and will ask personalized questions."
        
        return {
            "session_id": session_id,
            "question": first_question,
            "round_type": request.round_type,
            "duration_minutes": request.duration_minutes,
            "resume_questions_generated": len(session_data.get("resume_questions", [])),
            "message": f"Welcome to {request.round_type} interview. Duration: {request.duration_minutes} minutes.{resume_info} Let's begin!"
        }

@app.get("/final-report/{session_id}")
async def get_final_report(session_id: str):
    if session_id not in interview_sessions:
        return {"error": "Session not found"}
    
    session_data = interview_sessions[session_id]
    final_report = generate_final_comprehensive_report(session_data)
    
    return {
        "session_id": session_id,
        "comprehensive_final_report": final_report
    }

@app.get("/")
def home():
    return {
        "message": "AI-Powered Interview System",
        "features": [
            "AI-powered intelligent follow-up questions",
            "Resume-based personalized questions",
            "Alternative answer examples for each question",
            "Personalized feedback with improvement tips",
            "Comprehensive final report on completion",
            "Question-by-question detailed analysis",
            "Real-time answer evaluation",
            "Professional hiring recommendations"
        ],
        "usage_flow": {
            "1_start": "POST /interview/ with round_type, duration_minutes, optional resume_text",
            "2_continue": "POST /interview/ with session_id and user_answer",
            "3_completion": "Automatic comprehensive final report generation",
            "4_report": "GET /final-report/{session_id} for complete analysis"
        },
        "endpoints": [
            "POST /interview/ - Start/continue interview",
            "GET /final-report/{session_id} - Get comprehensive final report"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)