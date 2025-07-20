#!/usr/bin/env python3
"""
Quick test script for the depth estimation backend
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend server")
        return False

def test_root():
    """Test root endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint: {data['message']}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_models_endpoint():
    """Test models list endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/depth/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models endpoint: {len(data.get('models', []))} models available")
            print(f"   Default model: {data.get('default', 'unknown')}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False

def create_test_image():
    """Create a simple test image"""
    from PIL import Image, ImageDraw
    import io
    
    # Create a simple gradient image
    img = Image.new('RGB', (256, 256), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes for depth testing
    draw.ellipse([50, 50, 150, 150], fill='red')
    draw.ellipse([100, 100, 200, 200], fill='blue')
    draw.rectangle([80, 80, 180, 180], fill='green')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def test_depth_estimation():
    """Test depth estimation endpoint"""
    try:
        print("🔄 Testing depth estimation...")
        
        # Create test image
        test_image = create_test_image()
        
        files = {'file': ('test.png', test_image, 'image/png')}
        data = {'model_name': 'Intel/dpt-hybrid-midas'}
        
        response = requests.post(
            f"{BASE_URL}/api/depth/estimate",
            files=files,
            data=data,
            timeout=60  # Increase timeout for model loading
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Depth estimation successful")
                print(f"   Model used: {result.get('model_used', 'unknown')}")
                print(f"   Resolution: {result.get('resolution', 'unknown')}")
                return True
            else:
                print("❌ Depth estimation failed: result indicates failure")
                return False
        else:
            print(f"❌ Depth estimation failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error: {error_detail}")
            except:
                print(f"   Raw response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Depth estimation error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Depth Estimation Backend")
    print("=" * 40)
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Models Endpoint", test_models_endpoint),
        ("Depth Estimation", test_depth_estimation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        if test_func():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 40)
    print(f"📊 Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the backend server.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)