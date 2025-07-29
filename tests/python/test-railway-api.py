import requests
import json
from PIL import Image
import io

# Create a simple test image
img = Image.new('RGB', (100, 100), color='blue')
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

# Test the API
url = "https://web-production-a0df.up.railway.app/api/predict"
files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}

try:
    response = requests.post(url, files=files)
    data = response.json()
    
    # Pretty print the response
    print(json.dumps(data, indent=2))
    
    # Check for pointcloudData
    if 'pointcloudData' in data:
        print(f"\n✅ pointcloudData found! Count: {data['pointcloudData'].get('count', 'N/A')}")
    else:
        print("\n❌ pointcloudData NOT found in response")
        
except Exception as e:
    print(f"Error: {e}")