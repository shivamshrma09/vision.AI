from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import List, Dict
import ast
import re
import tempfile
import subprocess
import sys
import json

app = FastAPI(title="Enhanced Simple Code Judge", version="2.0.0")

generator = pipeline("text2text-generation", model="google/flan-t5-small")

class TestCase(BaseModel):
    input_data: List
    expected_output: any
    description: str

class CodeSubmission(BaseModel):
    code: str
    language: str
    problem_statement: str
    test_cases: List[TestCase]

def analyze_complexity_enhanced(code: str) -> Dict:
    """Enhanced complexity analysis with recursion detection"""
    try:
        tree = ast.parse(code)
        
        class ComplexityAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.loop_depth = 0
                self.max_depth = 0
                self.has_recursion = False
                self.function_name = None
                self.recursive_calls = 0
                
            def visit_FunctionDef(self, node):
                if not self.function_name:
                    self.function_name = node.name
                self.generic_visit(node)
                
            def visit_For(self, node):
                self.loop_depth += 1
                self.max_depth = max(self.max_depth, self.loop_depth)
                self.generic_visit(node)
                self.loop_depth -= 1
                
            def visit_While(self, node):
                self.loop_depth += 1
                self.max_depth = max(self.max_depth, self.loop_depth)
                self.generic_visit(node)
                self.loop_depth -= 1
                
            def visit_Call(self, node):
                if (isinstance(node.func, ast.Name) and 
                    node.func.id == self.function_name):
                    self.has_recursion = True
                    self.recursive_calls += 1
                self.generic_visit(node)
        
        analyzer = ComplexityAnalyzer()
        analyzer.visit(tree)
        
        if analyzer.has_recursion:
            if "fibonacci" in code.lower() or analyzer.recursive_calls > 1:
                complexity = "O(2^n)"
                score = 40
            else:
                complexity = "O(n)"
                score = 80
        elif analyzer.max_depth >= 3:
            complexity = "O(n³)"
            score = 50
        elif analyzer.max_depth >= 2:
            complexity = "O(n²)"
            score = 70
        elif analyzer.max_depth == 1:
            complexity = "O(n)"
            score = 90
        else:
            complexity = "O(1)"
            score = 100
            
        return {
            "time_complexity": complexity,
            "loop_depth": analyzer.max_depth,
            "has_recursion": analyzer.has_recursion,
            "recursive_calls": analyzer.recursive_calls,
            "complexity_score": score,
            "analysis": f"Detected {analyzer.max_depth} nested loops, recursion: {analyzer.has_recursion}"
        }
        
    except Exception as e:
        return {
            "time_complexity": "O(n)",
            "complexity_score": 80,
            "analysis": f"AST parsing failed: {str(e)}"
        }

def analyze_style_enhanced(code: str) -> Dict:
    """Enhanced style analysis"""
    issues = []
    good_practices = []
    score = 100
    
    lines = code.split('\n')
    
    if '"""' in code or "'''" in code:
        good_practices.append("Function documentation present")
    else:
        issues.append("Missing function docstrings")
        score -= 20
    
    if re.search(r'\bdef [A-Z][a-zA-Z]*\(', code):
        issues.append("Function names should use snake_case")
        score -= 10
    
    if re.search(r'\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b', code):
        issues.append("Variable names should use snake_case")
        score -= 10
    
    long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 79]
    if long_lines:
        issues.append(f"Lines too long (>79 chars): lines {long_lines[:3]}")
        score -= 5
    
    indentations = []
    for line in lines:
        if line.strip() and line.startswith(' '):
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                indentations.append(indent)
    
    if indentations and len(set(indentations)) > 2:
        issues.append("Inconsistent indentation")
        score -= 5
    
    if 'if __name__ == "__main__"' in code:
        good_practices.append("Proper main guard")
    
    if re.search(r'def \w+\([^)]*\) -> ', code):
        good_practices.append("Type hints present")
        score += 5
    
    comment_lines = [line for line in lines if '#' in line and line.strip().startswith('#')]
    if comment_lines:
        good_practices.append(f"Inline comments ({len(comment_lines)} lines)")
    
    return {
        "style_score": max(0, min(100, score)),
        "issues": issues,
        "good_practices": good_practices,
        "pep8_compliance": "High" if score > 85 else "Medium" if score > 70 else "Low",
        "line_count": len(lines),
        "comment_ratio": len(comment_lines) / len(lines) * 100 if lines else 0
    }

def analyze_security_enhanced(code: str) -> Dict:
    """Enhanced security analysis"""
    security_issues = []
    score = 100
    risk_factors = []
    
    dangerous_patterns = [
        (r'eval\s*\(', "Use of eval() is dangerous - can execute arbitrary code", 30),
        (r'exec\s*\(', "Use of exec() is dangerous - can execute arbitrary code", 30),
        (r'__import__\s*\(', "Dynamic imports can be risky", 15),
        (r'os\.system\s*\(', "System command execution detected", 25),
        (r'subprocess\.call\s*\(', "Subprocess usage - ensure input validation", 15),
        (r'pickle\.loads?\s*\(', "Pickle deserialization can be unsafe", 20),
        (r'input\s*\([^)]*\)', "Raw input usage - validate user input", 10)
    ]
    
    for pattern, message, penalty in dangerous_patterns:
        if re.search(pattern, code):
            security_issues.append(message)
            risk_factors.append(pattern)
            score -= penalty
    
    sql_patterns = [
        r'["\'].*%.*["\'].*%',
        r'["\'].*\+.*["\']',
        r'f["\'].*{.*}.*["\']'
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, code):
            security_issues.append("Potential SQL injection vulnerability")
            score -= 25
            break
    
    if re.search(r'(password|secret|key|token)\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
        security_issues.append("Hardcoded credentials detected")
        score -= 20
    
    return {
        "security_score": max(0, score),
        "security_issues": security_issues,
        "risk_factors": risk_factors,
        "risk_level": "Low" if score > 80 else "Medium" if score > 60 else "High"
    }

def execute_code_safely(code: str, test_cases: List[TestCase]) -> Dict:
    """Safely execute code with test cases"""
    results = []
    
    for i, test_case in enumerate(test_cases):
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                test_code = f"""
import sys
import json
import time

{code}

try:
    start_time = time.time()
    
    import ast
    tree = ast.parse('''{code}''')
    func_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            break
    
    if func_name:
        result = globals()[func_name](*{test_case.input_data})
        end_time = time.time()
        
        print(json.dumps({{
            "success": True,
            "result": result,
            "expected": {test_case.expected_output},
            "passed": result == {test_case.expected_output},
            "execution_time": end_time - start_time
        }}))
    else:
        print(json.dumps({{
            "success": False,
            "error": "No function found"
        }}))
        
except Exception as e:
    print(json.dumps({{
        "success": False,
        "error": str(e)
    }}))
"""
                f.write(test_code)
                f.flush()
                
                try:
                    result = subprocess.run(
                        [sys.executable, f.name],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.stdout:
                        exec_result = json.loads(result.stdout.strip())
                        results.append({
                            "test_case": i + 1,
                            "description": test_case.description,
                            "passed": exec_result.get("passed", False),
                            "output": exec_result.get("result"),
                            "expected": test_case.expected_output,
                            "execution_time": exec_result.get("execution_time", 0),
                            "error": exec_result.get("error")
                        })
                    else:
                        results.append({
                            "test_case": i + 1,
                            "description": test_case.description,
                            "passed": False,
                            "error": result.stderr or "No output"
                        })
                        
                except subprocess.TimeoutExpired:
                    results.append({
                        "test_case": i + 1,
                        "description": test_case.description,
                        "passed": False,
                        "error": "Execution timeout (>5s)"
                    })
                except Exception as e:
                    results.append({
                        "test_case": i + 1,
                        "description": test_case.description,
                        "passed": False,
                        "error": f"Execution error: {str(e)}"
                    })
                finally:
                    try:
                        import os
                        os.unlink(f.name)
                    except:
                        pass
                        
        except Exception as e:
            results.append({
                "test_case": i + 1,
                "description": test_case.description,
                "passed": False,
                "error": f"Setup error: {str(e)}"
            })
    
    passed_tests = sum(1 for r in results if r.get("passed", False))
    total_tests = len(results)
    correctness_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    return {
        "correctness_score": correctness_score,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "test_results": results
    }

@app.post("/judge-code/")
async def judge_code_enhanced(submission: CodeSubmission):
    """Enhanced async code judge with execution"""
    
    complexity = analyze_complexity_enhanced(submission.code)
    style = analyze_style_enhanced(submission.code)
    security = analyze_security_enhanced(submission.code)
    execution = execute_code_safely(submission.code, submission.test_cases)
    
    overall_score = int((
        execution["correctness_score"] * 0.4 +
        complexity["complexity_score"] * 0.25 +
        style["style_score"] * 0.2 +
        security["security_score"] * 0.15
    ))
    
    if overall_score >= 95:
        grade = "A+"
    elif overall_score >= 90:
        grade = "A"
    elif overall_score >= 80:
        grade = "B+"
    elif overall_score >= 70:
        grade = "B"
    elif overall_score >= 60:
        grade = "C"
    else:
        grade = "D"
    
    feedback_prompt = f"""
    Code Analysis for: {submission.problem_statement}
    
    Results:
    - Correctness: {execution['correctness_score']:.1f}% ({execution['passed_tests']}/{execution['total_tests']} tests passed)
    - Complexity: {complexity['time_complexity']} (Score: {complexity['complexity_score']}/100)
    - Style: {style['pep8_compliance']} compliance (Score: {style['style_score']}/100)
    - Security: {security['risk_level']} risk (Score: {security['security_score']}/100)
    
    Provide specific, actionable feedback for improvement.
    """
    
    try:
        ai_feedback = generator(feedback_prompt, max_length=200, do_sample=True, temperature=0.7)
        feedback_text = ai_feedback[0]['generated_text']
    except Exception as e:
        feedback_text = f"AI feedback unavailable: {str(e)}"
    
    return {
        "overall_score": overall_score,
        "grade": grade,
        "pass_status": "PASS" if overall_score >= 70 else "FAIL",
        
        "execution_results": {
            "correctness_score": execution["correctness_score"],
            "passed_tests": execution["passed_tests"],
            "total_tests": execution["total_tests"],
            "test_details": execution["test_results"]
        },
        
        "complexity_analysis": {
            "time_complexity": complexity["time_complexity"],
            "complexity_score": complexity["complexity_score"],
            "loop_depth": complexity["loop_depth"],
            "has_recursion": complexity["has_recursion"],
            "analysis": complexity["analysis"]
        },
        
        "style_analysis": {
            "style_score": style["style_score"],
            "pep8_compliance": style["pep8_compliance"],
            "issues": style["issues"],
            "good_practices": style["good_practices"],
            "line_count": style["line_count"],
            "comment_ratio": f"{style['comment_ratio']:.1f}%"
        },
        
        "security_analysis": {
            "security_score": security["security_score"],
            "risk_level": security["risk_level"],
            "security_issues": security["security_issues"],
            "risk_factors": security["risk_factors"]
        },
        
        "ai_feedback": feedback_text,
        
        "recommendations": [
            f"Fix failing tests ({execution['total_tests'] - execution['passed_tests']} failed)" if execution["passed_tests"] < execution["total_tests"] else None,
            f"Optimize {complexity['time_complexity']} complexity" if complexity["complexity_score"] < 80 else None,
            "Improve code documentation and style" if style["style_score"] < 80 else None,
            f"Address {security['risk_level'].lower()} security risks" if security["security_score"] < 80 else None
        ],
        
        "summary": {
            "strengths": [
                "Excellent correctness" if execution["correctness_score"] > 90 else None,
                "Optimal complexity" if complexity["complexity_score"] > 90 else None,
                "Clean code style" if style["style_score"] > 90 else None,
                "Secure implementation" if security["security_score"] > 90 else None
            ],
            "areas_for_improvement": [
                "Test coverage" if execution["correctness_score"] < 70 else None,
                "Algorithm efficiency" if complexity["complexity_score"] < 70 else None,
                "Code quality" if style["style_score"] < 70 else None,
                "Security practices" if security["security_score"] < 70 else None
            ]
        }
    }

@app.get("/")
def home():
    return {
        "message": "🚀 Enhanced Simple Code Judge Ready!",
        "features": [
            "Real Code Execution with Test Cases",
            "Enhanced Complexity Analysis (with recursion detection)",
            "Comprehensive Style Checking",
            "Advanced Security Scanning",
            "AI-Powered Feedback",
            "Weighted Scoring System",
            "Async Processing"
        ],
        "improvements": [
            "✅ Recursion detection in complexity analysis",
            "✅ Real code execution with correctness testing",
            "✅ Enhanced style analysis with more checks",
            "✅ Better security pattern matching",
            "✅ Async endpoints for better performance",
            "✅ Weighted scoring with correctness priority",
            "✅ Detailed test case results"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)