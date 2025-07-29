#!/bin/bash

# Development startup script for Depth Estimation App

echo "🚀 Starting Depth Estimation App Development Environment"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi

# Create backend virtual environment if it doesn't exist
if [ ! -d "backend/venv" ]; then
    echo "📦 Creating Python virtual environment..."
    cd backend
    python3 -m venv venv
    cd ..
fi

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd backend
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || {
    echo "❌ Failed to activate virtual environment"
    exit 1
}
pip install -r requirements.txt
cd ..

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Start backend server in background
echo "🔧 Starting backend server..."
cd backend
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend server
echo "🎨 Starting frontend server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ Development servers started!"
echo "📡 Backend API: http://localhost:8000"
echo "🌐 Frontend: http://localhost:3000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user interrupt
trap 'echo "🛑 Stopping servers..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait