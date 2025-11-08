from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List, Dict, Optional
import ast
import re
import tempfile
import subprocess
import sys
import json
import time
import traceback
import asyncio
import psutil
import os

app = FastAPI(title="Balanced Coding Judge", version="6.0.0")

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

class TestCase(BaseModel):
    input_data: str
    expected_output: str
    description: str
    time_limit: Optional[float] = 2.0
    memory_limit: Optional[int] = 256
    is_hidden: Optional[bool] = False
    difficulty: Optional[str] = "medium"

class CodeSubmission(BaseModel):
    code: str
    language: str
    problem_id: str
    problem_statement: str
    test_cases: List[TestCase]
    constraints: Optional[Dict] = {}
    company: Optional[str] = "generic"

def get_language_config(language: str) -> Dict:
    configs = {
        "python": {
            "extension": ".py",
            "compile_cmd": None,
            "run_cmd": [sys.executable],
            "timeout": 10
        },
        "cpp": {
            "extension": ".cpp",
            "compile_cmd": ["g++", "-std=c++17", "-O2", "-o"],
            "run_cmd": [],
            "timeout": 15
        },
        "c": {
            "extension": ".c",
            "compile_cmd": ["gcc", "-std=c11", "-O2", "-o"],
            "run_cmd": [],
            "timeout": 15
        },
        "java": {
            "extension": ".java",
            "compile_cmd": ["javac"],
            "run_cmd": ["java"],
            "timeout": 15
        },
        "javascript": {
            "extension": ".js",
            "compile_cmd": None,
            "run_cmd": ["node"],
            "timeout": 10
        }
    }
    return configs.get(language, configs["python"])

def analyze_code_structure(code: str, language: str) -> Dict:
    lines = code.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    structure_analysis = {
        "total_lines": len(lines),
        "code_lines": len(non_empty_lines),
        "comment_lines": len([line for line in lines if line.strip().startswith('#' if language == 'python' else '//')]),
        "blank_lines": len(lines) - len(non_empty_lines),
        "functions_count": len(re.findall(r'def |function |int |void ', code)),
        "variables_count": len(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\s*=', code)),
        "readability_score": calculate_readability_score(code, language)
    }
    
    return structure_analysis

def calculate_readability_score(code: str, language: str) -> int:
    score = 100
    lines = code.split('\n')
    
    # Line length penalty
    long_lines = [line for line in lines if len(line) > 80]
    score -= min(len(long_lines) * 5, 20)
    
    # Indentation consistency
    indents = []
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                indents.append(indent)
    
    if indents and len(set(indents)) > 3:
        score -= 10
    
    # Variable naming
    if language == "python":
        bad_names = re.findall(r'\b[a-z][A-Z]', code)  # camelCase in Python
        score -= len(bad_names) * 3
    
    # Comments ratio
    comment_ratio = len([l for l in lines if l.strip().startswith('#')]) / len(lines) * 100
    if comment_ratio < 5:
        score -= 10
    elif comment_ratio > 30:
        score -= 5
    
    return max(0, score)

def analyze_algorithm_patterns(code: str, language: str) -> Dict:
    patterns = {
        "sorting": bool(re.search(r'sort|Sort', code)),
        "searching": bool(re.search(r'binary_search|bsearch|find|search', code)),
        "dynamic_programming": bool(re.search(r'dp|memo|cache|@lru_cache', code)),
        "recursion": bool(re.search(r'def.*\(.*\).*:.*\1\(', code, re.DOTALL)) if language == "python" else False,
        "greedy": bool(re.search(r'min|max|optimal|greedy', code)),
        "graph_algorithms": bool(re.search(r'dfs|bfs|graph|tree|node', code)),
        "data_structures": detect_data_structures(code, language),
        "mathematical": bool(re.search(r'math|sqrt|pow|factorial|gcd', code))
    }
    
    pattern_score = sum([10 for pattern, used in patterns.items() if used and pattern != "data_structures"])
    pattern_score += len(patterns["data_structures"]) * 5
    
    return {
        "patterns_detected": patterns,
        "pattern_diversity_score": min(pattern_score, 100),
        "algorithm_sophistication": get_sophistication_level(patterns)
    }

def detect_data_structures(code: str, language: str) -> List[str]:
    structures = []
    
    if language == "python":
        if re.search(r'set\(|{.*}', code): structures.append("set")
        if re.search(r'dict\(|{.*:.*}', code): structures.append("dictionary")
        if re.search(r'list\(|\[.*\]', code): structures.append("list")
        if re.search(r'deque|queue', code): structures.append("queue")
        if re.search(r'heapq|heap', code): structures.append("heap")
    elif language in ["cpp", "c"]:
        if re.search(r'vector|array', code): structures.append("array")
        if re.search(r'map|unordered_map', code): structures.append("hash_map")
        if re.search(r'set|unordered_set', code): structures.append("set")
        if re.search(r'queue|stack', code): structures.append("queue")
    elif language == "java":
        if re.search(r'ArrayList|List', code): structures.append("list")
        if re.search(r'HashMap|Map', code): structures.append("hash_map")
        if re.search(r'HashSet|Set', code): structures.append("set")
        if re.search(r'Queue|Stack', code): structures.append("queue")
    
    return structures

def get_sophistication_level(patterns: Dict) -> str:
    advanced_patterns = ["dynamic_programming", "graph_algorithms", "mathematical"]
    intermediate_patterns = ["sorting", "searching", "recursion"]
    
    if any(patterns[p] for p in advanced_patterns):
        return "Advanced"
    elif any(patterns[p] for p in intermediate_patterns):
        return "Intermediate"
    else:
        return "Basic"

def analyze_performance_characteristics(code: str, language: str) -> Dict:
    # Time complexity analysis
    loop_count = len(re.findall(r'for|while', code))
    nested_loops = 0
    
    if language == "python":
        nested_loops = len(re.findall(r'for.*:.*for.*:', code, re.DOTALL))
    elif language in ["cpp", "c", "java"]:
        nested_loops = len(re.findall(r'for\s*\([^)]*\)\s*{[^}]*for\s*\([^)]*\)', code, re.DOTALL))
    
    # Space complexity indicators
    space_indicators = {
        "extra_arrays": len(re.findall(r'new\s+\w+\[|malloc|vector<|list<|ArrayList', code)),
        "recursive_calls": bool(re.search(r'return.*\w+\(', code)),
        "memoization": bool(re.search(r'memo|cache|dp\[', code))
    }
    
    # Estimate complexity
    if nested_loops >= 3:
        time_complexity = "O(n³)"
        efficiency_score = 40
    elif nested_loops >= 2:
        time_complexity = "O(n²)"
        efficiency_score = 60
    elif nested_loops >= 1 or loop_count >= 1:
        time_complexity = "O(n)"
        efficiency_score = 80
    else:
        time_complexity = "O(1)"
        efficiency_score = 100
    
    # Adjust for optimizations
    if space_indicators["memoization"]:
        efficiency_score += 10
    if space_indicators["extra_arrays"] > 2:
        efficiency_score -= 10
    
    return {
        "estimated_time_complexity": time_complexity,
        "estimated_space_complexity": "O(n)" if space_indicators["extra_arrays"] > 0 or space_indicators["recursive_calls"] else "O(1)",
        "efficiency_score": max(0, min(100, efficiency_score)),
        "optimization_opportunities": get_optimization_suggestions(code, language),
        "performance_bottlenecks": identify_bottlenecks(code, language)
    }

def get_optimization_suggestions(code: str, language: str) -> List[str]:
    suggestions = []
    
    if re.search(r'range\(len\(', code):
        suggestions.append("Use enumerate() instead of range(len())")
    
    if re.search(r'for.*in.*if', code) and language == "python":
        suggestions.append("Consider using list comprehension")
    
    if re.search(r'\.append\(.*\)', code) and "list" in code:
        suggestions.append("Pre-allocate list size if known")
    
    if not re.search(r'memo|cache|dp', code) and re.search(r'def.*\(.*\).*:.*return.*\1\(', code, re.DOTALL):
        suggestions.append("Add memoization to recursive functions")
    
    return suggestions

def identify_bottlenecks(code: str, language: str) -> List[str]:
    bottlenecks = []
    
    if len(re.findall(r'for.*for.*for', code)) > 0:
        bottlenecks.append("Triple nested loops detected")
    
    if re.search(r'\.sort\(\).*for.*in', code):
        bottlenecks.append("Sorting inside loop")
    
    if re.search(r'print\(|console\.log\(|cout\s*<<', code):
        bottlenecks.append("I/O operations in main logic")
    
    return bottlenecks

def calculate_code_quality_metrics(code: str, language: str) -> Dict:
    structure = analyze_code_structure(code, language)
    patterns = analyze_algorithm_patterns(code, language)
    performance = analyze_performance_characteristics(code, language)
    
    # Overall quality score
    quality_components = {
        "readability": structure["readability_score"] * 0.25,
        "algorithm_design": patterns["pattern_diversity_score"] * 0.35,
        "performance": performance["efficiency_score"] * 0.40
    }
    
    overall_quality = sum(quality_components.values())
    
    return {
        "overall_quality_score": round(overall_quality, 1),
        "quality_breakdown": quality_components,
        "code_maturity": get_code_maturity_level(overall_quality),
        "improvement_priority": get_improvement_priority(quality_components),
        "industry_readiness": assess_industry_readiness(overall_quality, patterns["algorithm_sophistication"])
    }

def get_code_maturity_level(score: float) -> str:
    if score >= 85:
        return "Production Ready"
    elif score >= 70:
        return "Good Quality"
    elif score >= 55:
        return "Acceptable"
    else:
        return "Needs Improvement"

def get_improvement_priority(components: Dict) -> str:
    min_component = min(components, key=components.get)
    return f"Focus on {min_component.replace('_', ' ').title()}"

def assess_industry_readiness(score: float, sophistication: str) -> Dict:
    readiness = {
        "junior_developer": score >= 50,
        "mid_level_developer": score >= 70 and sophistication in ["Intermediate", "Advanced"],
        "senior_developer": score >= 85 and sophistication == "Advanced",
        "tech_lead_ready": score >= 90 and sophistication == "Advanced"
    }
    
    current_level = "Entry Level"
    if readiness["tech_lead_ready"]:
        current_level = "Tech Lead Ready"
    elif readiness["senior_developer"]:
        current_level = "Senior Developer"
    elif readiness["mid_level_developer"]:
        current_level = "Mid-Level Developer"
    elif readiness["junior_developer"]:
        current_level = "Junior Developer"
    
    return {
        "current_level": current_level,
        "readiness_breakdown": readiness,
        "next_milestone": get_next_milestone(current_level)
    }

def get_next_milestone(current_level: str) -> str:
    milestones = {
        "Entry Level": "Focus on algorithm fundamentals and code structure",
        "Junior Developer": "Learn advanced data structures and design patterns",
        "Mid-Level Developer": "Master system design and optimization techniques",
        "Senior Developer": "Develop architectural thinking and mentoring skills",
        "Tech Lead Ready": "Continue excellence and explore new technologies"
    }
    return milestones.get(current_level, "Keep improving!")

async def execute_multi_language_tests(code: str, language: str, test_cases: List[TestCase]) -> Dict:
    config = get_language_config(language)
    results = []
    total_score = 0
    execution_times = []
    
    for i, test_case in enumerate(test_cases):
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=config["extension"], delete=False) as f:
                if language == "python":
                    test_code = create_python_test(code, test_case)
                elif language in ["cpp", "c"]:
                    test_code = create_cpp_test(code, test_case, language)
                elif language == "java":
                    test_code = create_java_test(code, test_case)
                elif language == "javascript":
                    test_code = create_js_test(code, test_case)
                
                f.write(test_code)
                f.flush()
                
                try:
                    # Compile if needed
                    if config["compile_cmd"]:
                        if language == "java":
                            compile_result = subprocess.run(
                                config["compile_cmd"] + [f.name],
                                capture_output=True, text=True, timeout=10
                            )
                        else:
                            executable = f.name.replace(config["extension"], "")
                            compile_result = subprocess.run(
                                config["compile_cmd"] + [executable, f.name],
                                capture_output=True, text=True, timeout=10
                            )
                        
                        if compile_result.returncode != 0:
                            results.append({
                                "test_case": i + 1,
                                "passed": False,
                                "error": f"Compilation Error: {compile_result.stderr}",
                                "verdict": "COMPILATION_ERROR"
                            })
                            continue
                    
                    # Execute
                    if language == "java":
                        class_name = extract_java_class_name(code)
                        run_cmd = config["run_cmd"] + [class_name]
                        cwd = os.path.dirname(f.name)
                    elif config["compile_cmd"] and language in ["cpp", "c"]:
                        executable = f.name.replace(config["extension"], "")
                        run_cmd = [executable]
                        cwd = None
                    else:
                        run_cmd = config["run_cmd"] + [f.name]
                        cwd = None
                    
                    start_time = time.time()
                    result = subprocess.run(
                        run_cmd,
                        input=test_case.input_data,
                        capture_output=True,
                        text=True,
                        timeout=test_case.time_limit,
                        cwd=cwd
                    )
                    end_time = time.time()
                    
                    execution_time = end_time - start_time
                    execution_times.append(execution_time)
                    output = result.stdout.strip()
                    expected = test_case.expected_output.strip()
                    
                    passed = output == expected
                    if passed:
                        total_score += 1
                    
                    results.append({
                        "test_case": i + 1,
                        "description": test_case.description,
                        "passed": passed,
                        "output": output,
                        "expected": expected,
                        "execution_time": execution_time,
                        "performance_rating": "Fast" if execution_time < 0.1 else "Average" if execution_time < 1.0 else "Slow",
                        "verdict": "ACCEPTED" if passed else "WRONG_ANSWER"
                    })
                    
                except subprocess.TimeoutExpired:
                    results.append({
                        "test_case": i + 1,
                        "passed": False,
                        "error": f"Time Limit Exceeded (>{test_case.time_limit}s)",
                        "verdict": "TIME_LIMIT_EXCEEDED"
                    })
                except Exception as e:
                    results.append({
                        "test_case": i + 1,
                        "passed": False,
                        "error": f"Runtime Error: {str(e)}",
                        "verdict": "RUNTIME_ERROR"
                    })
                finally:
                    try:
                        os.unlink(f.name)
                        if config["compile_cmd"] and language in ["cpp", "c"]:
                            executable = f.name.replace(config["extension"], "")
                            if os.path.exists(executable):
                                os.unlink(executable)
                    except:
                        pass
                        
        except Exception as e:
            results.append({
                "test_case": i + 1,
                "passed": False,
                "error": f"Setup Error: {str(e)}",
                "verdict": "COMPILATION_ERROR"
            })
    
    total_tests = len(results)
    correctness_score = (total_score / total_tests * 100) if total_tests > 0 else 0
    
    return {
        "correctness_score": correctness_score,
        "passed_tests": total_score,
        "total_tests": total_tests,
        "test_results": results,
        "performance_summary": {
            "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "fastest_test": min(execution_times) if execution_times else 0,
            "slowest_test": max(execution_times) if execution_times else 0,
            "performance_consistency": "High" if execution_times and max(execution_times) - min(execution_times) < 0.5 else "Medium"
        },
        "final_verdict": "ACCEPTED" if correctness_score == 100 else "PARTIAL" if correctness_score >= 50 else "FAILED"
    }

def create_python_test(code: str, test_case: TestCase) -> str:
    return f"""
import sys
from typing import *

{code}

try:
    input_data = {repr(test_case.input_data)}
    
    import ast
    tree = ast.parse('''{code}''')
    func_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            break
    
    if func_name:
        lines = input_data.strip().split('\\n')
        if len(lines) == 1:
            try:
                result = globals()[func_name](int(lines[0]))
            except:
                result = globals()[func_name](lines[0])
        else:
            args = []
            for line in lines:
                try:
                    if ' ' in line:
                        args.append(list(map(int, line.split())))
                    else:
                        args.append(int(line))
                except:
                    args.append(line)
            result = globals()[func_name](*args)
        
        print(result)
    
except Exception as e:
    print(f"Error: {{e}}")
"""

def create_cpp_test(code: str, test_case: TestCase, language: str) -> str:
    return f"""
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <queue>
#include <stack>
using namespace std;

{code}

int main() {{
    return 0;
}}
"""

def create_java_test(code: str, test_case: TestCase) -> str:
    return f"""
import java.util.*;
import java.io.*;

{code}
"""

def create_js_test(code: str, test_case: TestCase) -> str:
    return f"""
{code}

const input = `{test_case.input_data}`;
const lines = input.trim().split('\\n');

try {{
    let result;
    if (lines.length === 1) {{
        const arg = isNaN(lines[0]) ? lines[0] : parseInt(lines[0]);
        const funcMatch = `{code}`.match(/function\\s+(\\w+)/);
        if (funcMatch) {{
            result = eval(funcMatch[1] + '(' + JSON.stringify(arg) + ')');
        }}
    }}
    console.log(result);
}} catch (e) {{
    console.log('Error:', e.message);
}}
"""

def extract_java_class_name(code: str) -> str:
    match = re.search(r'public\s+class\s+(\w+)', code)
    return match.group(1) if match else "Solution"

@app.post("/judge-balanced/")
async def judge_balanced_code(submission: CodeSubmission):
    try:
        supported_languages = ["python", "cpp", "c", "java", "javascript"]
        if submission.language not in supported_languages:
            raise HTTPException(status_code=400, detail=f"Language {submission.language} not supported")
        
        # Comprehensive analysis
        quality_metrics = calculate_code_quality_metrics(submission.code, submission.language)
        execution_results = await execute_multi_language_tests(submission.code, submission.language, submission.test_cases)
        
        # Calculate balanced score
        overall_score = int(
            execution_results["correctness_score"] * 0.50 +
            quality_metrics["overall_quality_score"] * 0.50
        )
        
        # Balanced grading system
        if overall_score >= 85:
            grade = "EXCELLENT"
            level = "Senior Ready"
        elif overall_score >= 75:
            grade = "GOOD"
            level = "Mid-Level Ready"
        elif overall_score >= 65:
            grade = "SATISFACTORY"
            level = "Junior+ Ready"
        elif overall_score >= 50:
            grade = "NEEDS_IMPROVEMENT"
            level = "Junior Ready"
        else:
            grade = "POOR"
            level = "Entry Level"
        
        return {
            "overall_score": overall_score,
            "grade": grade,
            "experience_level": level,
            "language": submission.language.upper(),
            "final_verdict": execution_results["final_verdict"],
            
            "execution_analysis": {
                "correctness_score": execution_results["correctness_score"],
                "passed_tests": execution_results["passed_tests"],
                "total_tests": execution_results["total_tests"],
                "performance_summary": execution_results["performance_summary"],
                "test_results": execution_results["test_results"]
            },
            
            "code_quality_analysis": {
                "overall_quality_score": quality_metrics["overall_quality_score"],
                "code_maturity": quality_metrics["code_maturity"],
                "quality_breakdown": quality_metrics["quality_breakdown"],
                "improvement_priority": quality_metrics["improvement_priority"]
            },
            
            "industry_readiness_assessment": quality_metrics["industry_readiness"],
            
            "detailed_insights": {
                "algorithm_sophistication": analyze_algorithm_patterns(submission.code, submission.language)["algorithm_sophistication"],
                "performance_characteristics": analyze_performance_characteristics(submission.code, submission.language),
                "code_structure_metrics": analyze_code_structure(submission.code, submission.language)
            },
            
            "recommendations": {
                "immediate_improvements": analyze_performance_characteristics(submission.code, submission.language)["optimization_opportunities"],
                "career_guidance": quality_metrics["industry_readiness"]["next_milestone"],
                "skill_development_focus": get_skill_focus(overall_score, quality_metrics)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def get_skill_focus(score: int, quality_metrics: Dict) -> List[str]:
    focus_areas = []
    
    if quality_metrics["quality_breakdown"]["readability"] < 60:
        focus_areas.append("Code readability and documentation")
    
    if quality_metrics["quality_breakdown"]["algorithm_design"] < 60:
        focus_areas.append("Algorithm design and data structures")
    
    if quality_metrics["quality_breakdown"]["performance"] < 60:
        focus_areas.append("Performance optimization and complexity analysis")
    
    if score < 70:
        focus_areas.append("Problem-solving fundamentals")
    
    return focus_areas if focus_areas else ["Continue practicing advanced concepts"]

@app.get("/")
def home():
    return {
        "message": "🎯 Balanced Coding Judge - Mid to Upper-Intermediate Analysis",
        "analysis_level": "Balanced (Mid-Level Focus)",
        "supported_languages": ["Python", "C++", "C", "Java", "JavaScript"],
        "key_features": [
            "✅ Balanced complexity analysis (not too basic, not too advanced)",
            "✅ Code quality and structure assessment",
            "✅ Industry readiness evaluation",
            "✅ Performance characteristics analysis",
            "✅ Algorithm pattern recognition",
            "✅ Career progression guidance",
            "✅ Practical improvement suggestions"
        ],
        "grading_system": {
            "EXCELLENT": "85+ (Senior Ready)",
            "GOOD": "75-84 (Mid-Level Ready)",
            "SATISFACTORY": "65-74 (Junior+ Ready)",
            "NEEDS_IMPROVEMENT": "50-64 (Junior Ready)",
            "POOR": "<50 (Entry Level)"
        },
        "analysis_focus": [
            "Code structure and readability",
            "Algorithm design choices",
            "Performance optimization opportunities",
            "Industry best practices",
            "Career development guidance"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)