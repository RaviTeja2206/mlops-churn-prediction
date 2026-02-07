#!/bin/bash
# Docker management script for M1 Mac

set -e

PROJECT_NAME="churn-prediction-api"
IMAGE_NAME="churn-prediction-api:latest"
CONTAINER_NAME="churn-api"

case "$1" in
    build)
        echo "🐳 Building Docker image for ARM64 (M1 Mac)..."
        docker build --platform linux/arm64 -t $IMAGE_NAME .
        echo "✅ Build complete"
        docker images $PROJECT_NAME
        ;;
    
    run)
        echo "🚀 Starting container..."
        docker run -d \
            --name $CONTAINER_NAME \
            --platform linux/arm64 \
            -p 8000:8000 \
            -v "$(pwd)/logs:/app/logs" \
            $IMAGE_NAME
        
        echo "⏳ Waiting for API to start..."
        sleep 5
        
        echo "✅ Container started"
        docker ps | grep $CONTAINER_NAME
        
        echo ""
        echo "📍 API available at: http://localhost:8000"
        echo "📖 API docs: http://localhost:8000/docs"
        ;;
    
    stop)
        echo "🛑 Stopping container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
        echo "✅ Container stopped and removed"
        ;;
    
    logs)
        echo "📋 Container logs:"
        docker logs -f $CONTAINER_NAME
        ;;
    
    test)
        echo "🧪 Testing API..."
        python3 api/test_api.py
        ;;
    
    shell)
        echo "🐚 Opening shell in container..."
        docker exec -it $CONTAINER_NAME /bin/bash
        ;;
    
    clean)
        echo "🧹 Cleaning up..."
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        docker rmi $IMAGE_NAME 2>/dev/null || true
        echo "✅ Cleanup complete"
        ;;
    
    *)
        echo "Usage: $0 {build|run|stop|logs|test|shell|clean}"
        echo ""
        echo "Commands:"
        echo "  build  - Build Docker image"
        echo "  run    - Start container"
        echo "  stop   - Stop and remove container"
        echo "  logs   - View container logs"
        echo "  test   - Run API tests"
        echo "  shell  - Open shell in container"
        echo "  clean  - Remove container and image"
        exit 1
        ;;
esac
