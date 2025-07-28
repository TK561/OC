# Test Railway API deployment
$apiUrl = "https://web-production-a0df.up.railway.app"

Write-Host "Testing Railway API Deployment Status..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Test root endpoint
Write-Host "`n1. Testing Root Endpoint:" -ForegroundColor Yellow
$rootResponse = Invoke-RestMethod -Uri "$apiUrl/" -Method Get
Write-Host "Status: Running" -ForegroundColor Green
Write-Host "Model: $($rootResponse.model)"
Write-Host "Note: $($rootResponse.note)"

# Test health endpoint
Write-Host "`n2. Testing Health Endpoint:" -ForegroundColor Yellow
$healthResponse = Invoke-RestMethod -Uri "$apiUrl/health" -Method Get
Write-Host "Status: $($healthResponse.status)" -ForegroundColor Green
Write-Host "Model Loaded: $($healthResponse.model_loaded)"
Write-Host "Algorithms: $($healthResponse.algorithms)"

# Test prediction endpoint
Write-Host "`n3. Testing Prediction Endpoint with Sample Image:" -ForegroundColor Yellow
$imagePath = "C:\Users\filqo\OneDrive\Desktop\demo\frontend\public\samples\sample1.jpg"

if (Test-Path $imagePath) {
    try {
        # Create multipart form data
        $boundary = [System.Guid]::NewGuid().ToString()
        $LF = "`r`n"
        
        $fileBytes = [System.IO.File]::ReadAllBytes($imagePath)
        $fileEnc = [System.Text.Encoding]::GetEncoding('ISO-8859-1').GetString($fileBytes)
        
        $bodyLines = (
            "--$boundary",
            "Content-Disposition: form-data; name=`"file`"; filename=`"sample1.jpg`"",
            "Content-Type: image/jpeg",
            "",
            $fileEnc,
            "--$boundary--"
        ) -join $LF
        
        $response = Invoke-RestMethod -Uri "$apiUrl/api/predict" -Method Post -ContentType "multipart/form-data; boundary=$boundary" -Body $bodyLines
        
        Write-Host "Success: $($response.success)" -ForegroundColor Green
        Write-Host "Model: $($response.model)"
        Write-Host "Resolution: $($response.resolution)"
        Write-Host "Implementation: $($response.implementation)"
        
        if ($response.pointcloudData) {
            Write-Host "`n4. 3D Pointcloud Data:" -ForegroundColor Yellow
            Write-Host "Points Count: $($response.pointcloudData.count)" -ForegroundColor Green
            Write-Host "Downsample Factor: $($response.pointcloudData.downsample_factor)"
            Write-Host "Has Points Array: $($null -ne $response.pointcloudData.points)"
            Write-Host "Has Colors Array: $($null -ne $response.pointcloudData.colors)"
            
            if ($response.pointcloudData.points -and $response.pointcloudData.points.Count -gt 0) {
                Write-Host "Sample Point: [$($response.pointcloudData.points[0][0]), $($response.pointcloudData.points[0][1]), $($response.pointcloudData.points[0][2])]"
                Write-Host "Sample Color: [$($response.pointcloudData.colors[0][0]), $($response.pointcloudData.colors[0][1]), $($response.pointcloudData.colors[0][2])]"
            }
        } else {
            Write-Host "`nWarning: No pointcloud data in response!" -ForegroundColor Red
        }
        
        # Save response
        $response | ConvertTo-Json -Depth 10 | Out-File "railway_api_test_result.json"
        Write-Host "`nFull response saved to railway_api_test_result.json" -ForegroundColor Cyan
        
    } catch {
        Write-Host "Error during prediction: $_" -ForegroundColor Red
    }
} else {
    Write-Host "Error: Test image not found at $imagePath" -ForegroundColor Red
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "API Deployment Test Complete!" -ForegroundColor Green