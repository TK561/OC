from PIL import Image
import io
import base64

# Create a simple test image
img = Image.new('RGB', (100, 100), color='red')

# Save it
img.save('test_image.jpg', 'JPEG')
print("Test image created: test_image.jpg")

# Also create a base64 version for testing
buffer = io.BytesIO()
img.save(buffer, format='JPEG')
img_base64 = base64.b64encode(buffer.getvalue()).decode()
print(f"Base64 length: {len(img_base64)}")

# Save base64 to file for reference
with open('test_image_base64.txt', 'w') as f:
    f.write(img_base64)