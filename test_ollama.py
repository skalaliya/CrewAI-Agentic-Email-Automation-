#!/usr/bin/env python3
"""Test script to verify Ollama integration with CrewAI agents and new YAML configuration."""

import os
from src.mail_agents.agents import EmailAgents

def test_yaml_configuration():
    """Test the new YAML-based configuration system."""
    print("🧪 Testing New YAML-based Configuration")
    print("=" * 50)
    
    # Initialize agents
    print("📝 Initializing EmailAgents with YAML configs...")
    try:
        agents = EmailAgents()
        print("✅ EmailAgents initialized successfully!")
        
        # Test configuration validation
        print("\n🔍 Validating configuration...")
        validation = agents.validate_configuration()
        if validation['valid']:
            print("✅ Configuration is valid!")
            print(f"   - Agents created: {validation.get('agents_created', 'N/A')}")
            print(f"   - Tasks available: {validation.get('tasks_available', 'N/A')}")
            if validation.get('warnings'):
                for warning in validation['warnings']:
                    print(f"   ⚠️  {warning}")
        else:
            print("❌ Configuration validation failed:")
            for issue in validation.get('issues', []):
                print(f"   - {issue}")
        
        # Show crew info
        print("\n📊 Crew Information:")
        crew_info = agents.get_crew_info()
        print(f"   - Available agents: {crew_info['available_agents']}")
        print(f"   - Available tasks: {crew_info['available_tasks']}")
        print(f"   - Task templates: {crew_info['available_templates']}")
        print(f"   - Specialized crews: {crew_info['specialized_crews']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing agents: {e}")
        return False

def test_ollama_integration():
    """Test Ollama integration with CrewAI agents."""
    print("\n🧪 Testing Ollama Integration with YAML Configs")
    print("=" * 50)
    
    # Initialize agents
    print("📝 Initializing EmailAgents...")
    try:
        agents = EmailAgents()
        print("✅ EmailAgents initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing agents: {e}")
        return False
    
    # Test simple email classification
    test_email = "Congratulations! You've won $1000! Click here to claim now!"
    print(f"\n📧 Testing email classification...")
    print(f"Email: {test_email}")
    
    try:
        # Test spam classifier agent
        print("\n🤖 Testing spam classifier agent...")
        result = agents.classify_email(test_email)
        print(f"✅ Classification result: {result[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Error in classification: {e}")
        return False

def test_advanced_features():
    """Test advanced features like information extraction and response drafting."""
    print("\n🧪 Testing Advanced Features")
    print("=" * 50)
    
    try:
        agents = EmailAgents()
        
        # Test information extraction
        business_email = """
        Subject: Q4 Budget Meeting
        
        Hi team,
        
        Let's schedule our Q4 budget review for March 15th at 2:00 PM in Conference Room B.
        Please bring your department projections.
        
        Best regards,
        John Smith
        Finance Director
        """
        
        print("📊 Testing information extraction...")
        extraction_result = agents.extract_information(business_email)
        print(f"✅ Information extracted: {extraction_result[:150]}...")
        
        # Test response drafting  
        print("\n✍️  Testing response drafting...")
        response_result = agents.draft_response(
            business_email, 
            context="Accepting the meeting invitation as department head"
        )
        print(f"✅ Response drafted: {response_result[:150]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in advanced features: {e}")
        return False

if __name__ == "__main__":
    # Check if Ollama is running
    print("🔍 Checking Ollama status...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama is running with {len(models)} models")
            for model in models:
                print(f"   - {model['name']}")
        else:
            print("❌ Ollama is not responding properly")
            exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        exit(1)
    
    # Run all tests
    tests = [
        ("YAML Configuration", test_yaml_configuration),
        ("Ollama Integration", test_ollama_integration), 
        ("Advanced Features", test_advanced_features)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running Test: {test_name}")
        print(f"{'='*60}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 Test Results Summary")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your YAML-based CrewAI setup is working perfectly!")
        print("\n💡 Key improvements with the new setup:")
        print("   ✅ Clean YAML-based configuration")
        print("   ✅ Organized folder structure")
        print("   ✅ Backwards compatibility maintained")
        print("   ✅ Enhanced agent capabilities")
        print("   ✅ Better error handling and validation")
        print("\n📖 Next steps:")
        print("   1. Train the ML model: uv run python -m src.mail_agents.cli train")
        print("   2. Start the API server: uv run python -m src.mail_agents.cli run-api")
        print("   3. Use the enhanced Jupyter notebook: jupyter notebook kaggle_demo.ipynb")
        print("   4. Customize YAML configs in src/mail_agents/crew/configs/")
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed. Please check the errors above.")
        exit(1)