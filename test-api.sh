#!/bin/bash

# Test Railway API with a sample image
echo "Testing Railway API..."

# Create a simple test image using PowerShell
powershell -Command "
\$bitmap = New-Object System.Drawing.Bitmap 100, 100
\$graphics = [System.Drawing.Graphics]::FromImage(\$bitmap)
\$graphics.FillRectangle([System.Drawing.Brushes]::Blue, 0, 0, 100, 100)
\$bitmap.Save('test-image.jpg', [System.Drawing.Imaging.ImageFormat]::Jpeg)
\$graphics.Dispose()
\$bitmap.Dispose()
"

# Test the API
curl -X POST https://web-production-a0df.up.railway.app/api/predict \
  -F "file=@test-image.jpg" \
  -H "Accept: application/json" | python -m json.tool

# Clean up
rm test-image.jpg