#!/usr/bin/env python3
"""
Test all interview rounds
"""
import requests

BASE_URL = "http://localhost:8002"

def test_all_rounds():
    tests = [
        # Coding Round
        {
            "endpoint": "/coding/",
            "data": {"question": "Implement quicksort algorithm", "difficulty": "medium", "max_tokens": 120},
            "name": "💻 Coding Round"
        },
        
        # Technical Round  
        {
            "endpoint": "/technical/",
            "data": {"question": "Explain Docker vs Kubernetes", "max_tokens": 100},
            "name": "🧠 Technical Round"
        },
        
        # System Design
        {
            "endpoint": "/system-design/",
            "data": {"question": "Design Instagram architecture", "max_tokens": 150},
            "name": "🏗️ System Design"
        },
        
        # HR Round
        {
            "endpoint": "/hr/",
            "data": {"question": "Why do you want to work here?", "max_tokens": 100},
            "name": "👔 HR Round"
        },
        
        # Behavioral Round
        {
            "endpoint": "/behavioral/",
            "data": {"question": "Tell me about a time you faced a challenge", "max_tokens": 120},
            "name": "🎭 Behavioral Round"
        },
        
        # Database Round
        {
            "endpoint": "/database/",
            "data": {"question": "Explain database normalization", "max_tokens": 100},
            "name": "🗄️ Database Round"
        },
        
        # Frontend Round
        {
            "endpoint": "/frontend/",
            "data": {"question": "What is React Virtual DOM?", "max_tokens": 100},
            "name": "🎨 Frontend Round"
        },
        
        # Backend Round
        {
            "endpoint": "/backend/",
            "data": {"question": "How to handle API rate limiting?", "max_tokens": 120},
            "name": "⚙️ Backend Round"
        },
        
        # Code Review
        {
            "endpoint": "/code-review/",
            "data": {"code": "for i in range(len(arr)):\\n    print(arr[i])", "language": "python", "max_tokens": 80},
            "name": "👨‍💻 Code Review"
        }
    ]
    
    print("🚀 Testing All Interview Rounds")
    print("=" * 60)
    
    for test in tests:
        try:
            print(f"\\n{test['name']}")
            print("-" * 40)
            
            response = requests.post(f"{BASE_URL}{test['endpoint']}", json=test['data'])
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {result['response'][:100]}...")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Connection Error: {e}")
    
    print("\\n" + "=" * 60)
    print("✅ All tests completed!")

if __name__ == "__main__":
    test_all_rounds()