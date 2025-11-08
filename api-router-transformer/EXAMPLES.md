# Usage Examples

## Quick Setup Examples

### 1. Using with Z.AI (Free Alternative to Claude)

```bash
# Start transformer
export BACKEND_URL=https://api.z.ai/api/anthropic
export BACKEND_API_KEY=665b963943b647dc9501dff942afb877.A47LrMc7sgGjyfBJ
npm start

# Test with curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.6",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 2. Using with Official Anthropic API

```bash
# Start transformer
export BACKEND_URL=https://api.anthropic.com
export BACKEND_API_KEY=sk-ant-api03-your-key-here
npm start
```

## Integration Examples

### Jinx Agent

```bash
# 1. Start transformer
docker-compose up -d

# 2. Configure Jinx
cd /path/to/jinx
export OPENAI_API_BASE=http://localhost:8000
export OPENAI_API_KEY=dummy
export OPENAI_MODEL=glm-4.6

# 3. Run Jinx
python jinx.py "Create a REST API with FastAPI"
```

### Aider

```bash
# Configure
export OPENAI_API_BASE=http://localhost:8000
export OPENAI_API_KEY=dummy

# Run
aider --model glm-4.6 --edit-format whole
```

### Continue.dev (VS Code Extension)

Add to `~/.continue/config.json`:
```json
{
  "models": [{
    "title": "Z.AI GLM-4.6",
    "provider": "openai",
    "model": "glm-4.6",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "dummy"
  }]
}
```

### LangChain (Python)

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(
    model="glm-4.6",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="dummy",
    temperature=0.7
)

response = llm([HumanMessage(content="Explain RAG in simple terms")])
print(response.content)
```

### LlamaIndex (Python)

```python
from llama_index.llms import OpenAI
from llama_index import ServiceContext, VectorStoreIndex

llm = OpenAI(
    model="glm-4.6",
    api_base="http://localhost:8000/v1",
    api_key="dummy"
)

service_context = ServiceContext.from_defaults(llm=llm)
```

## Advanced Examples

### Streaming with Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

stream = client.chat.completions.create(
    model="glm-4.6",
    messages=[{"role": "user", "content": "Count from 1 to 10"}],
    stream=True,
    max_tokens=100
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

### Streaming with Node.js

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'dummy',
});

const stream = await client.chat.completions.create({
  model: 'glm-4.6',
  messages: [{ role: 'user', content: 'Write a short poem' }],
  stream: true,
  max_tokens: 200,
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content || '';
  process.stdout.write(content);
}
```

### With System Messages

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.6",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful coding assistant specialized in Python"
      },
      {
        "role": "user",
        "content": "Write a function to calculate fibonacci"
      }
    ],
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

### With Temperature and Top-P

```python
response = client.chat.completions.create(
    model="glm-4.6",
    messages=[{"role": "user", "content": "Be creative: write a sci-fi story opening"}],
    temperature=0.9,  # More creative
    top_p=0.95,
    max_tokens=300
)
```

## Production Deployment Examples

### nginx Reverse Proxy

```nginx
upstream api_router {
    server localhost:8000;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://api_router;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_cache_bypass $http_upgrade;
        
        # For streaming
        proxy_buffering off;
        proxy_read_timeout 300s;
    }
}
```

### Traefik (Docker Compose)

```yaml
version: '3.8'

services:
  api-router:
    image: api-router-transformer:latest
    environment:
      - BACKEND_URL=https://api.z.ai/api/anthropic
      - BACKEND_API_KEY=${BACKEND_API_KEY}
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api-router.rule=Host(`api.yourdomain.com`)"
      - "traefik.http.routers.api-router.entrypoints=websecure"
      - "traefik.http.routers.api-router.tls.certresolver=letsencrypt"
      - "traefik.http.services.api-router.loadbalancer.server.port=8000"

  traefik:
    image: traefik:v2.10
    command:
      - "--api.insecure=true"
      - "--providers.docker=true"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.letsencrypt.acme.tlschallenge=true"
      - "--certificatesresolvers.letsencrypt.acme.email=your@email.com"
      - "--certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json"
    ports:
      - "443:443"
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./letsencrypt:/letsencrypt
```

### Multi-Instance Load Balancing

```yaml
version: '3.8'

services:
  api-router-1:
    image: api-router-transformer:latest
    environment:
      - BACKEND_URL=${BACKEND_URL}
      - BACKEND_API_KEY=${BACKEND_API_KEY}
  
  api-router-2:
    image: api-router-transformer:latest
    environment:
      - BACKEND_URL=${BACKEND_URL}
      - BACKEND_API_KEY=${BACKEND_API_KEY}
  
  api-router-3:
    image: api-router-transformer:latest
    environment:
      - BACKEND_URL=${BACKEND_URL}
      - BACKEND_API_KEY=${BACKEND_API_KEY}
  
  nginx:
    image: nginx:alpine
    ports:
      - "8000:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api-router-1
      - api-router-2
      - api-router-3
```

`nginx.conf`:
```nginx
events {}

http {
    upstream api_backends {
        least_conn;
        server api-router-1:8000;
        server api-router-2:8000;
        server api-router-3:8000;
    }

    server {
        listen 80;
        
        location / {
            proxy_pass http://api_backends;
            proxy_http_version 1.1;
            proxy_buffering off;
        }
    }
}
```

## Monitoring Examples

### Prometheus Metrics (Future Enhancement)

```yaml
# Add to docker-compose.yml
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
```

### Health Check Script

```bash
#!/bin/bash

ENDPOINT="http://localhost:8000/health"
RESPONSE=$(curl -s $ENDPOINT)
STATUS=$(echo $RESPONSE | jq -r '.status')

if [ "$STATUS" == "healthy" ]; then
    echo "✅ Service is healthy"
    exit 0
else
    echo "❌ Service is unhealthy"
    exit 1
fi
```

## Troubleshooting Examples

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=debug
npm start

# Or with Docker
docker-compose up  # Remove -d flag to see logs
```

### Test Backend Connection

```bash
# Test Z.AI directly
curl -X POST https://api.z.ai/api/anthropic/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "glm-4.6",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "test"}]
  }'
```

### Request Tracing

```bash
# Check request ID in response headers
curl -v http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-4.6","messages":[{"role":"user","content":"test"}]}' \
  2>&1 | grep -i "x-request-id"
```

---

Need more examples? Check the [README.md](./README.md) or [create an issue](https://github.com/Zeeeepa/agent/issues)!

