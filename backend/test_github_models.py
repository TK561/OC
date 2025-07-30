"""
Test script for GitHub-based model implementations
"""
import asyncio
import requests
from PIL import Image
import io
import os

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "../frontend/public/samples/sample1.jpg"

# Models to test
MODELS_TO_TEST = [
    "Intel/dpt-large",
    "Intel/dpt-hybrid-midas", 
    "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything/Depth-Anything-V2-Large-hf"
]

def test_model(model_name: str, image_path: str):
    """Test a specific model"""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Prepare the image
        with open(image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            data = {'model_name': model_name}
            
            # Make request to v2 API
            response = requests.post(
                f"{API_BASE_URL}/api/depth/v2/estimate",
                files=files,
                params=data
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"   Model used: {result['modelUsed']}")
            print(f"   Resolution: {result['resolution']}")
            print(f"   Depth map URL: {result['depthMapUrl']}")
            print(f"   Processing note: {result.get('processingNote', 'N/A')}")
            
            # Download and save depth map
            depth_url = f"{API_BASE_URL}{result['depthMapUrl']}"
            depth_response = requests.get(depth_url)
            if depth_response.status_code == 200:
                # Save depth map
                output_filename = f"depth_output_{model_name.replace('/', '_')}.png"
                with open(output_filename, 'wb') as f:
                    f.write(depth_response.content)
                print(f"   Saved depth map to: {output_filename}")
        else:
            print(f"❌ Failed with status code: {response.status_code}")
            print(f"   Error: {response.json()}")
            
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")

def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("Testing model info endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/depth/v2/models/info")
        if response.status_code == 200:
            info = response.json()
            print("✅ Model information retrieved successfully!")
            print(f"   Default model: {info['default']}")
            print(f"   Implementation: {info['implementation']}")
            print(f"   Depth convention: {info['depth_convention']}")
            print("\n   Available models:")
            for model_id, details in info['models'].items():
                print(f"   - {model_id}: {details['name']}")
                print(f"     Description: {details['description']}")
                print(f"     Architecture: {details['architecture']}")
                print(f"     Features: {', '.join(details['features'])}")
                print()
        else:
            print(f"❌ Failed to get model info: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting model info: {str(e)}")

def main():
    """Run all tests"""
    print("🚀 Starting GitHub Model Implementation Tests")
    
    # Check if test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Test image not found: {TEST_IMAGE_PATH}")
        return
    
    # Test model info endpoint
    test_model_info()
    
    # Test each model
    for model in MODELS_TO_TEST:
        test_model(model, TEST_IMAGE_PATH)
        # Small delay between tests
        import time
        time.sleep(2)
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()