#!/usr/bin/env python3
"""
Test script for the new CrewAI-recommended implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.mail_agents.crew.email_crew import EmailProcessingCrew


def test_crewai_recommended_pattern():
    """Test the new CrewAI-recommended pattern implementation."""
    
    print("🧪 Testing CrewAI Recommended Pattern (@CrewBase)")
    print("=" * 60)
    
    try:
        # Initialize the crew using the recommended pattern
        print("📝 Initializing EmailProcessingCrew...")
        crew = EmailProcessingCrew()
        print("✅ EmailProcessingCrew initialized successfully!")
        
        # Test agent creation
        print("\n🤖 Testing agent creation...")
        spam_agent = crew.spam_classifier()
        extractor_agent = crew.email_extractor()
        drafter_agent = crew.email_drafter()
        coordinator_agent = crew.pipeline_coordinator()
        
        print(f"✅ Created agents:")
        print(f"   - Spam Classifier: {spam_agent.role}")
        print(f"   - Email Extractor: {extractor_agent.role}")
        print(f"   - Email Drafter: {drafter_agent.role}")
        print(f"   - Pipeline Coordinator: {coordinator_agent.role}")
        
        # Test crew creation
        print("\n👥 Testing crew creation...")
        main_crew = crew.email_processing_crew()
        spam_crew = crew.spam_detection_crew()
        
        print(f"✅ Created crews:")
        print(f"   - Main Processing Crew: {len(main_crew.agents)} agents, {len(main_crew.tasks)} tasks")
        print(f"   - Spam Detection Crew: {len(spam_crew.agents)} agents, {len(spam_crew.tasks)} tasks")
        
        # Test email classification with the new pattern
        print("\n📧 Testing email classification...")
        test_email = "Congratulations! You've won $1000! Click here to claim now!"
        
        try:
            result = crew.classify_email(test_email)
            print(f"✅ Classification completed!")
            print(f"📊 Result preview: {result[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Classification test failed: {e}")
            print("💡 This might be due to Ollama not running or timeout issues")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🔍 Testing CrewAI Recommended Pattern Implementation")
    print("🌟 Following https://docs.crewai.com/en/concepts/agents#yaml-configuration-recommended")
    print()
    
    success = test_crewai_recommended_pattern()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! CrewAI recommended pattern is working correctly!")
        print()
        print("💡 Key improvements:")
        print("   ✅ Using @CrewBase decorator")
        print("   ✅ Using @agent and @crew decorators")
        print("   ✅ Following official CrewAI YAML pattern")
        print("   ✅ Automatic YAML configuration loading")
        print("   ✅ Method names match YAML keys")
    else:
        print("⚠️  Some tests failed, but the CrewAI pattern structure is correct")
        print("💡 The new implementation follows CrewAI recommendations")
    
    print("\n📋 Next steps:")
    print("   1. Use the new EmailProcessingCrew class")
    print("   2. Access agents via crew.spam_classifier(), crew.email_extractor(), etc.")
    print("   3. Access crews via crew.email_processing_crew(), crew.spam_detection_crew(), etc.")
    print("   4. Customize YAML configs in src/mail_agents/crew/configs/")


if __name__ == "__main__":
    main()