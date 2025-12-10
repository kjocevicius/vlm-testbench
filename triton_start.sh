#!/bin/bash

echo "=== VLM Testbench - Triton Inference Server Startup ==="
echo ""

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "Error: Docker Compose is not installed"
    exit 1
fi

if nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected, using GPU configuration"
    COMPOSE_FILE="docker-compose.yml"
else
    echo "! No NVIDIA GPU detected, using CPU configuration"
    COMPOSE_FILE="docker-compose-cpu.yml"
fi

echo ""
echo "Starting Triton Inference Server..."
docker compose -f $COMPOSE_FILE up -d

echo ""
echo "Waiting for server to be ready..."
sleep 5

if curl -s http://localhost:8000/v2/health/ready > /dev/null; then
    echo "✓ Triton server is ready!"
    echo ""
    echo "Available endpoints:"
    echo "  - HTTP: http://localhost:8000"
    echo "  - gRPC: localhost:8001"
    echo "  - Metrics: http://localhost:8002/metrics"
    echo ""
    echo "View logs: docker compose -f $COMPOSE_FILE logs -f"
    echo "Stop server: docker compose -f $COMPOSE_FILE down"
    echo ""
    echo "Next: Open notebooks/triton_inference.ipynb"
else
    echo "✗ Server not ready yet. Check logs:"
    echo "  docker compose -f $COMPOSE_FILE logs -f"
fi
