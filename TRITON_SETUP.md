# Triton Inference Server Setup

Complete guide for running VLM models via Triton Inference Server.

## Quick Start

```bash
# Option 1: Use startup script (auto-detects GPU/CPU)
./triton_start.sh

# Option 2: Manual start
docker compose up -d                           # GPU
docker compose -f docker-compose-cpu.yml up -d # CPU
```

## Architecture

```
triton_models/
├── moondream2/
│   ├── config.pbtxt          # Model configuration
│   └── 1/
│       └── model.py          # Python backend implementation
├── kosmos2/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
└── llava/
    ├── config.pbtxt
    └── 1/
        └── model.py
```

## Model Endpoints

All models accept:
- **Input:**
  - `image` (UINT8) - Image bytes (PNG/JPEG)
  - `prompt` (STRING) - Text prompt

- **Output:**
  - `generated_text` (STRING) - Model response

### Available Models

1. **moondream2** - Fast 2B parameter model
2. **kosmos2** - Grounding and detection capabilities
3. **llava** - Most capable 7B parameter model

## Usage Examples

### Python Client

```python
from triton_client import TritonVLMClient

client = TritonVLMClient(url="localhost:8000")

result = client.infer(
    model_name="moondream2",
    image="test_images/sample_image.jpg",
    prompt="Describe this image in detail."
)
print(result)
```

### HTTP API

```bash
# Check server health
curl http://localhost:8000/v2/health/ready

# Check model status
curl http://localhost:8000/v2/models/moondream2/ready

# List all models
curl http://localhost:8000/v2/models
```

## Configuration

### Model Config (config.pbtxt)

Key settings:
- `max_batch_size: 0` - Disables batching (each request processed independently)
- `kind: KIND_GPU` - GPU placement (CPU fallback available)
- `dynamic_batching` - Enabled for better throughput

### Docker Compose

**GPU Version** (`docker-compose.yml`):
- Requires NVIDIA GPU + nvidia-docker
- Uses GPU acceleration for inference
- Faster performance

**CPU Version** (`docker-compose-cpu.yml`):
- No GPU required
- Slower but more compatible
- Good for testing

### Ports

- **8000**: HTTP inference endpoint
- **8001**: gRPC inference endpoint  
- **8002**: Metrics endpoint

## Model Loading

Models are loaded lazily:
1. First request triggers model load
2. Subsequent requests use cached model
3. Loading takes 30-60 seconds per model

Monitor loading:
```bash
docker compose logs -f
```

## Performance Tuning

### Instance Groups

In `config.pbtxt`, adjust concurrent instances:
```protobuf
instance_group [
  {
    count: 2  # Number of model instances
    kind: KIND_GPU
  }
]
```

### Dynamic Batching

Configure batching delays:
```protobuf
dynamic_batching {
  max_queue_delay_microseconds: 100
}
```

## Monitoring

### Metrics

Access Prometheus metrics:
```bash
curl http://localhost:8000/v2/metrics
```

Key metrics:
- `nv_inference_request_success`
- `nv_inference_request_duration_us`
- `nv_gpu_utilization`

### Logs

```bash
# Follow logs
docker compose logs -f

# Filter by model
docker compose logs -f | grep moondream2
```

## Troubleshooting

### Server Won't Start

```bash
# Check Docker is running
docker ps

# Check logs
docker compose logs

# Restart server
docker compose down
docker compose up -d
```

### Model Loading Fails

Common issues:
1. **Out of memory** - Reduce concurrent instances or use CPU
2. **Missing dependencies** - Models download transformers packages on first load
3. **HuggingFace cache** - Ensure `~/.cache/huggingface` is accessible

### Connection Refused

```bash
# Wait for server to be ready
curl http://localhost:8000/v2/health/ready

# Check if port is available
lsof -i :8000
```

## Cleanup

```bash
# Stop server
docker compose down

# Remove container and volumes
docker compose down -v

# Clean up models cache (optional)
rm -rf triton_models/*/1/__pycache__
```

## Next Steps

1. Open `notebooks/triton_inference.ipynb`
2. Run cells to test each model
3. Compare performance with direct inference
4. Try custom prompts and images

## Resources

- [Triton Documentation](https://github.com/triton-inference-server/server)
- [Python Backend Guide](https://github.com/triton-inference-server/python_backend)
- [Model Configuration Reference](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md)
