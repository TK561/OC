import requests
import json
import os

# API endpoint
url = "https://web-production-a0df.up.railway.app/api/predict"

# Test image path
image_path = r"C:\Users\filqo\OneDrive\Desktop\demo\frontend\public\samples\sample1.jpg"

# Check if file exists
if not os.path.exists(image_path):
    print(f"Error: File not found at {image_path}")
    exit(1)

print(f"Testing Railway API with image: {os.path.basename(image_path)}")

# Send request
try:
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(url, files=files)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\nAPI Response:")
        print(f"- Success: {data.get('success', False)}")
        print(f"- Model: {data.get('model', 'N/A')}")
        print(f"- Resolution: {data.get('resolution', 'N/A')}")
        print(f"- Implementation: {data.get('implementation', 'N/A')}")
        print(f"- Features: {data.get('features', [])}")
        
        # Check for 3D pointcloud data
        if 'pointcloudData' in data:
            pointcloud = data['pointcloudData']
            print(f"\n3D Pointcloud Data:")
            print(f"- Points count: {pointcloud.get('count', 0)}")
            print(f"- Downsample factor: {pointcloud.get('downsample_factor', 'N/A')}")
            print(f"- Has points array: {'points' in pointcloud}")
            print(f"- Has colors array: {'colors' in pointcloud}")
            
            if 'points' in pointcloud and len(pointcloud['points']) > 0:
                print(f"- Sample point: {pointcloud['points'][0]}")
                print(f"- Sample color: {pointcloud['colors'][0]}")
        else:
            print("\n⚠️ No pointcloud data in response!")
            
        # Save full response for inspection
        with open('railway_api_response.json', 'w') as f:
            json.dump(data, f, indent=2)
        print("\nFull response saved to railway_api_response.json")
        
    else:
        print(f"\nError: {response.text}")
        
except Exception as e:
    print(f"\nError occurred: {str(e)}")