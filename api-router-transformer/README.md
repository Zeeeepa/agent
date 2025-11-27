# API Router/Transformer

🔀 **Production-ready API router/transformer for converting between OpenAI and Anthropic API formats with 100% accuracy.**

Enable any tool expecting OpenAI format to work with Anthropic/Claude APIs (like Z.AI) seamlessly!

## ✨ Features

- ✅ **Accurate Transformations**: 100% compatible request/response transformation
- ✅ **Streaming Support**: Full SSE streaming for both formats
- ✅ **Production-Ready**: TypeScript, error handling, logging, health checks
- ✅ **Multiple Deployment Options**: Docker, Docker Compose, npm, programmatic
- ✅ **Zero Configuration**: Works out of the box with sensible defaults
- ✅ **Comprehensive Logging**: Structured logging with request IDs
- ✅ **CORS Support**: Enable cross-origin requests if needed
- ✅ **Health Checks**: Built-in health endpoint for monitoring

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Create .env file
cat > .env << EOF
BACKEND_URL=https://api.z.ai/api/anthropic
BACKEND_API_KEY=your-api-key-here
EOF

# 2. Run with Docker
docker build -t api-router-transformer .
docker run -p 8000:8000 --env-file .env api-router-transformer
```

### Option 2: Docker Compose

```bash
# 1. Create .env file (same as above)

# 2. Start service
docker-compose up -d

# 3. Check logs
docker-compose logs -f

# 4. Stop service
docker-compose down
```

### Option 3: NPM

```bash
# 1. Install dependencies
npm install

# 2. Set environment variables
export BACKEND_URL=https://api.z.ai/api/anthropic
export BACKEND_API_KEY=your-api-key-here

# 3. Build and run
npm run build
npm start

# Or for development with hot-reload:
npm run dev
```

## 📋 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BACKEND_URL` | ✅ Yes | - | Backend API URL (e.g., `https://api.z.ai/api/anthropic`) |
| `BACKEND_API_KEY` | ✅ Yes | - | API key for backend service |
| `PORT` | No | `8000` | Port to listen on |
| `LOG_LEVEL` | No | `info` | Log level (`error`, `warn`, `info`, `debug`) |
| `CORS_ENABLED` | No | `true` | Enable CORS headers |
| `REQUEST_TIMEOUT` | No | `60000` | Request timeout in milliseconds |

### Example Configurations

**For Z.AI:**
```bash
BACKEND_URL=https://api.z.ai/api/anthropic
BACKEND_API_KEY=665b963943b647dc9501dff942afb877.A47LrMc7sgGjyfBJ
```

**For Anthropic:**
```bash
BACKEND_URL=https://api.anthropic.com
BACKEND_API_KEY=sk-ant-api03-xxx
```

## 🔧 Usage Examples

### Using with Jinx Agent

```bash
# 1. Start the transformer
docker-compose up -d

# 2. Configure Jinx to use OpenAI format
export OPENAI_API_BASE=http://localhost:8000
export OPENAI_API_KEY=dummy  # Not used, but required by some tools
export OPENAI_MODEL=glm-4.6   # Or your preferred model

# 3. Run Jinx
python jinx.py "Write a Python function to sort a list"
```

### Using with Aider

```bash
# 1. Start the transformer
export BACKEND_URL=https://api.z.ai/api/anthropic
export BACKEND_API_KEY=your-key
npm start

# 2. Configure Aider
export OPENAI_API_BASE=http://localhost:8000
export OPENAI_API_KEY=dummy

# 3. Run Aider
aider --model glm-4.6
```

### Using with cURL

```bash
# Non-streaming request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.6",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'

# Streaming request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.6",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5"}
    ],
    "max_tokens": 100,
    "stream": true
  }'
```

### Using with Python OpenAI SDK

```python
from openai import OpenAI

# Configure client to use transformer
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not used, but required
)

# Make request
response = client.chat.completions.create(
    model="glm-4.6",
    messages=[
        {"role": "user", "content": "Write a haiku about AI"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Using with JavaScript/TypeScript

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'dummy',
});

const response = await client.chat.completions.create({
  model: 'glm-4.6',
  messages: [
    { role: 'user', content: 'Explain quantum computing in simple terms' }
  ],
  max_tokens: 200,
});

console.log(response.choices[0].message.content);
```

## 📊 API Endpoints

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint. Accepts OpenAI format, transforms to Anthropic, forwards to backend.

**Request Body (OpenAI format):**
```json
{
  "model": "glm-4.6",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

**Response (OpenAI format):**
```json
{
  "id": "msg_123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "glm-4.6",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "backend": "https://api.z.ai/api/anthropic"
}
```

## 🔀 Parameter Mapping

### Request Parameters

| OpenAI | Anthropic | Notes |
|--------|-----------|-------|
| `model` | `model` | Direct mapping |
| `messages` | `messages` | System messages extracted to `system` field |
| `max_tokens` | `max_tokens` | Direct mapping (required by Anthropic) |
| `temperature` | `temperature` | Direct mapping |
| `top_p` | `top_p` | Direct mapping |
| `stop` | `stop_sequences` | Converted to array |
| `stream` | `stream` | Direct mapping |
| `user` | `metadata.user_id` | Mapped to metadata |
| `presence_penalty` | ❌ | Not supported (ignored) |
| `frequency_penalty` | ❌ | Not supported (ignored) |
| `n` | ❌ | Not supported (must be 1) |

### Response Parameters

| Anthropic | OpenAI | Notes |
|-----------|--------|-------|
| `id` | `id` | Direct mapping |
| `content[].text` | `choices[0].message.content` | Text blocks joined |
| `stop_reason` | `finish_reason` | Mapped (`end_turn`→`stop`, `max_tokens`→`length`) |
| `usage.input_tokens` | `usage.prompt_tokens` | Direct mapping |
| `usage.output_tokens` | `usage.completion_tokens` | Direct mapping |

## 🧪 Testing

```bash
# Run tests
npm test

# Watch mode
npm run test:watch

# Coverage
npm run test:coverage
```

## 📝 Development

```bash
# Install dependencies
npm install

# Run in development mode with hot-reload
npm run dev

# Build
npm run build

# Lint
npm run lint

# Format
npm run format
```

## 🐳 Deployment

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-router-transformer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-router-transformer
  template:
    metadata:
      labels:
        app: api-router-transformer
    spec:
      containers:
      - name: api-router-transformer
        image: api-router-transformer:latest
        ports:
        - containerPort: 8000
        env:
        - name: BACKEND_URL
          value: "https://api.z.ai/api/anthropic"
        - name: BACKEND_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: backend-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

### Docker Swarm

```bash
docker service create \
  --name api-router-transformer \
  --replicas 3 \
  --publish 8000:8000 \
  --env BACKEND_URL=https://api.z.ai/api/anthropic \
  --env BACKEND_API_KEY=your-key \
  api-router-transformer:latest
```

## 🔒 Security Considerations

1. **API Key Protection**: Never commit API keys to version control
2. **HTTPS**: Use HTTPS in production (put behind nginx/traefik)
3. **Rate Limiting**: Consider adding rate limiting for public deployments
4. **Input Validation**: All requests are validated before forwarding
5. **Error Handling**: Backend errors are sanitized before returning

## 🛠️ Troubleshooting

### Issue: "Environment variable BACKEND_URL is required"

**Solution**: Create a `.env` file with required variables or export them:
```bash
export BACKEND_URL=https://api.z.ai/api/anthropic
export BACKEND_API_KEY=your-key
```

### Issue: "Request to backend failed"

**Solution**: Check:
1. Backend URL is correct and accessible
2. API key is valid
3. Network connectivity
4. Check logs: `docker-compose logs -f`

### Issue: Empty responses from API

**Solution**:
1. Check if API has rate limits (especially free tiers)
2. Verify API key is still valid
3. Enable debug logging: `LOG_LEVEL=debug npm start`

## 📚 Learn More

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference/messages_post)
- [Z.AI Documentation](https://z.ai/docs)

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📞 Support

- GitHub Issues: [Create an issue](https://github.com/Zeeeepa/agent/issues)
- Documentation: See [EXAMPLES.md](./EXAMPLES.md) for more examples

---

**Built with ❤️ for the AI community**

