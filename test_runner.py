#!/usr/bin/env python3
"""
Test Runner for Internship Selection System
Run this script to test your ML model with various scenarios
"""

import json
import sys
import os

def load_test_samples():
    """Load test samples from JSON file"""
    try:
        with open('test_samples.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Error: test_samples.json not found!")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in test_samples.json: {e}")
        return None

def run_single_test(app_data):
    """Run a single application through the system"""
    try:
        # Import your app functions
        from app import predict_for_student
        
        print(f"\n{'='*80}")
        print(f"TESTING: {app_data.get('town', 'Unknown')} - {app_data.get('identify_as', 'Unknown')}")
        print(f"{'='*80}")
        
        # Run prediction
        result = predict_for_student(app_data)
        
        # Display results
        print(f"\n🎯 DECISION: {'SELECTED' if result['selected'] else 'NOT SELECTED'}")
        print(f"📊 CONFIDENCE: {result['confidence']:.1%}")
        print(f"\n📝 DETAILED REASONING:")
        print(result['reasoning'])
        
        return result
        
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return None

def run_batch_test():
    """Run all test samples in batch mode"""
    try:
        from app import process_applications
        
        print("🔄 Running batch test...")
        
        # Load samples
        samples = load_test_samples()
        if not samples:
            return
        
        # Convert to JSON string
        json_payload = json.dumps(samples)
        
        # Process all applications
        result = process_applications(json_payload)
        
        print("\n" + "="*80)
        print("BATCH TEST RESULTS")
        print("="*80)
        print(result)
        
    except Exception as e:
        print(f"❌ Error in batch test: {e}")

def run_individual_tests():
    """Run each test sample individually"""
    samples = load_test_samples()
    if not samples:
        return
    
    print(f"🧪 Running {len(samples)} individual tests...")
    
    results = []
    for i, sample in enumerate(samples, 1):
        print(f"\n📋 Test {i}/{len(samples)}")
        result = run_single_test(sample)
        if result:
            results.append(result)
    
    # Summary
    selected_count = sum(1 for r in results if r['selected'])
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total Applications: {len(results)}")
    print(f"Selected: {selected_count}")
    print(f"Not Selected: {len(results) - selected_count}")
    print(f"Selection Rate: {selected_count/len(results):.1%}")
    print(f"Average Confidence: {avg_confidence:.1%}")

def main():
    """Main test runner"""
    print("🚀 Internship Selection System - Test Runner")
    print("="*50)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("\nSelect test mode:")
        print("1. Individual tests (detailed analysis)")
        print("2. Batch test (ranked list)")
        print("3. Both")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            mode = "individual"
        elif choice == "2":
            mode = "batch"
        elif choice == "3":
            mode = "both"
        else:
            print("Invalid choice. Running individual tests.")
            mode = "individual"
    
    if mode in ["individual", "both"]:
        run_individual_tests()
    
    if mode in ["batch", "both"]:
        run_batch_test()
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main() 