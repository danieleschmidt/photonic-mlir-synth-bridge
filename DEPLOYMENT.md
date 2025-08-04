# Photonic MLIR Deployment Guide

This guide provides comprehensive instructions for deploying Photonic MLIR in various environments.

## Table of Contents
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Production Deployment](#production-deployment)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Configuration](#security-configuration)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites
- Python 3.9+
- LLVM/MLIR 17+
- CMake 3.18+
- Ninja build system

### Installation

1. **Install MLIR/LLVM dependencies:**
   ```bash
   ./scripts/install_mlir.sh
   ```

2. **Build the project:**
   ```bash
   make build
   ```

3. **Install Python package:**
   ```bash
   make install
   ```

4. **Verify installation:**
   ```bash
   python -c "import photonic_mlir; print('Installation successful!')"
   ```

## Docker Deployment

### Single Container

Build and run the Docker container:

```bash
# Build image
docker build -f docker/Dockerfile -t photonic-mlir:latest .

# Run container
docker run -d \
  --name photonic-mlir \
  -p 8080:8080 \
  -v $(pwd)/data:/data \
  -v $(pwd)/logs:/logs \
  -e PHOTONIC_LOG_LEVEL=INFO \
  photonic-mlir:latest
```

### Docker Compose

For a complete stack with monitoring and caching:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f photonic-mlir

# Scale the main service
docker-compose up -d --scale photonic-mlir=3
```

Services included:
- **photonic-mlir**: Main compiler service
- **redis**: Caching layer
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboards
- **nginx**: Load balancer (production profile)

### Development Environment

```bash
# Start with development profile
docker-compose --profile dev up -d

# Access Jupyter notebooks
open http://localhost:8889
```

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3.0+

### Using Helm Charts

1. **Add the Photonic MLIR Helm repository:**
   ```bash
   helm repo add photonic-mlir https://charts.photonic-mlir.org
   helm repo update
   ```

2. **Install the chart:**
   ```bash
   helm install photonic-mlir photonic-mlir/photonic-mlir \
     --namespace photonic-mlir \
     --create-namespace \
     --set image.tag=latest \
     --set resources.limits.memory=4Gi \
     --set resources.limits.cpu=2000m
   ```

3. **Access the service:**
   ```bash
   kubectl port-forward svc/photonic-mlir 8080:8080 -n photonic-mlir
   ```

### Manual Deployment

Apply Kubernetes manifests:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

## Production Deployment

### Infrastructure Requirements

**Minimum Requirements:**
- CPU: 4 cores
- Memory: 8GB RAM
- Storage: 50GB SSD
- Network: 1Gbps

**Recommended for Production:**
- CPU: 8+ cores
- Memory: 16GB+ RAM
- Storage: 100GB+ NVMe SSD
- Network: 10Gbps
- GPU: Optional (for acceleration)

### Environment Configuration

Create production configuration file:

```yaml
# config/production.yaml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 8
  max_requests: 1000

cache:
  enabled: true
  backend: "redis"
  url: "redis://redis:6379/0"
  size_mb: 2048

logging:
  level: "INFO"
  format: "json"
  file: "/logs/photonic-mlir.log"
  max_size: "100MB"
  backup_count: 5

security:
  enable_auth: true
  jwt_secret: "${JWT_SECRET}"
  cors_origins:
    - "https://app.company.com"
    - "https://api.company.com"

hardware:
  enable_gpu: true
  max_concurrent_jobs: 10
  timeout_seconds: 300

monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_interval: 30
```

### Load Balancing

Configure NGINX for load balancing:

```nginx
# nginx/nginx.conf
upstream photonic_mlir {
    least_conn;
    server photonic-mlir-1:8080;
    server photonic-mlir-2:8080;
    server photonic-mlir-3:8080;
}

server {
    listen 80;
    server_name api.photonic-mlir.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.photonic-mlir.com;
    
    ssl_certificate /etc/ssl/certs/photonic-mlir.crt;
    ssl_certificate_key /etc/ssl/private/photonic-mlir.key;
    
    location / {
        proxy_pass http://photonic_mlir;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings for long-running compilations
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://photonic_mlir/health;
    }
}
```

### Database Setup

For persistent storage of compilation results:

```bash
# Initialize PostgreSQL database
docker run -d \
  --name photonic-postgres \
  -e POSTGRES_DB=photonic_mlir \
  -e POSTGRES_USER=photonic \
  -e POSTGRES_PASSWORD=secure_password \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:15-alpine

# Run database migrations
python -m photonic_mlir.db migrate
```

## Monitoring and Observability

### Metrics Collection

Photonic MLIR exposes metrics in Prometheus format:

- **Compilation metrics**: Success rate, duration, queue depth
- **Resource metrics**: CPU, memory, GPU utilization
- **Cache metrics**: Hit rate, size, evictions
- **Error metrics**: Error types, frequency

### Grafana Dashboards

Import the provided Grafana dashboards:

1. **System Overview**: High-level system health
2. **Compilation Performance**: Compilation-specific metrics
3. **Resource Utilization**: Infrastructure metrics
4. **Cache Performance**: Caching effectiveness

### Alerting Rules

Example Prometheus alerting rules:

```yaml
# alerts/photonic-mlir.yml
groups:
  - name: photonic-mlir
    rules:
      - alert: HighCompilationFailureRate
        expr: rate(photonic_compilation_failures_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High compilation failure rate"
          
      - alert: CompilationQueueBacklog
        expr: photonic_compilation_queue_depth > 100
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Compilation queue backlog"
          
      - alert: CacheHitRateLow
        expr: photonic_cache_hit_rate < 0.8
        for: 10m
        labels:
          severity: warning  
        annotations:
          summary: "Cache hit rate below 80%"
```

### Logging

Structured logging configuration:

```json
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "json": {
      "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
      "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
    }
  },
  "handlers": {
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "filename": "/logs/photonic-mlir.log",
      "maxBytes": 104857600,
      "backupCount": 10,
      "formatter": "json"
    },
    "stdout": {
      "class": "logging.StreamHandler",
      "stream": "ext://sys.stdout",
      "formatter": "json"
    }
  },
  "loggers": {
    "photonic_mlir": {
      "level": "INFO",
      "handlers": ["file", "stdout"],
      "propagate": false
    }
  }
}
```

## Security Configuration

### Authentication and Authorization

```python
# config/security.py
SECURITY_CONFIG = {
    "authentication": {
        "enabled": True,
        "provider": "jwt",
        "jwt_secret": os.getenv("JWT_SECRET"),
        "jwt_algorithm": "HS256",
        "jwt_expiry": 3600  # 1 hour
    },
    "authorization": {
        "enabled": True,
        "roles": {
            "admin": ["compile", "simulate", "admin"],
            "user": ["compile", "simulate"],
            "readonly": ["view"]
        }
    },
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 60,
        "burst_size": 10
    },
    "input_validation": {
        "max_model_size_mb": 100,
        "max_file_size_mb": 50,
        "allowed_file_extensions": [".py", ".json", ".yaml"]
    }
}
```

### SSL/TLS Configuration

Generate SSL certificates:

```bash
# Self-signed certificate (development only)
openssl req -x509 -newkey rsa:4096 -keyout ssl/private.key -out ssl/cert.crt -days 365 -nodes

# Let's Encrypt (production)
certbot certonly --standalone -d api.photonic-mlir.com
```

### Network Security

Configure firewall rules:

```bash
# Allow HTTPS traffic
sudo ufw allow 443/tcp

# Allow SSH (if needed)
sudo ufw allow 22/tcp

# Allow internal communication
sudo ufw allow from 10.0.0.0/8 to any port 8080

# Enable firewall
sudo ufw enable
```

## Troubleshooting

### Common Issues

1. **MLIR compilation fails**
   ```bash
   # Check LLVM installation
   llvm-config --version
   which mlir-opt
   
   # Verify MLIR paths
   export MLIR_DIR=$(llvm-config --prefix)/lib/cmake/mlir
   ```

2. **Memory issues during compilation**
   ```bash
   # Increase memory limits
   docker run -m 8g photonic-mlir:latest
   
   # Or in docker-compose
   mem_limit: 8g
   ```

3. **Slow compilation performance**
   ```bash
   # Check CPU utilization
   docker stats
   
   # Increase worker count
   export PHOTONIC_MAX_WORKERS=8
   ```

### Debugging

Enable debug logging:

```bash
export PHOTONIC_LOG_LEVEL=DEBUG
export PHOTONIC_DEBUG=1
```

Use the health check endpoint:

```bash
curl http://localhost:8080/health
```

Check service status:

```bash
# Docker
docker-compose ps
docker-compose logs photonic-mlir

# Kubernetes
kubectl get pods -n photonic-mlir
kubectl logs -f deployment/photonic-mlir -n photonic-mlir
```

### Performance Tuning

1. **Optimize cache settings**:
   - Increase cache size for better hit rates
   - Use Redis for distributed caching
   - Enable cache warming

2. **Scale horizontally**:
   - Add more worker instances
   - Use load balancing
   - Implement autoscaling

3. **Hardware optimization**:
   - Use faster storage (NVMe SSDs)
   - Add more RAM for compilation
   - Consider GPU acceleration

### Support

For additional support:
- Documentation: https://photonic-mlir.readthedocs.io
- Issues: https://github.com/danieleschmidt/photonic-mlir-synth-bridge/issues
- Discussions: https://github.com/danieleschmidt/photonic-mlir-synth-bridge/discussions
- Email: support@photonic-mlir.org