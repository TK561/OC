# Create a simple test image
Add-Type -AssemblyName System.Drawing
$bitmap = New-Object System.Drawing.Bitmap 100, 100
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.FillRectangle([System.Drawing.Brushes]::Blue, 0, 0, 100, 100)
$bitmap.Save("test-image.jpg", [System.Drawing.Imaging.ImageFormat]::Jpeg)
$graphics.Dispose()
$bitmap.Dispose()

# Test the Railway API
$uri = "https://web-production-a0df.up.railway.app/api/predict"
$form = @{
    file = Get-Item -Path "test-image.jpg"
}

try {
    $response = Invoke-RestMethod -Uri $uri -Method Post -Form $form
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $_"
}

# Clean up
Remove-Item "test-image.jpg"