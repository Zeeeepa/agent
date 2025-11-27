# Jinx Agent - Requirements & Setup Guide

## System Requirements

### Minimum Requirements
- **Python**: 3.9 or higher
- **RAM**: 2GB minimum, 4GB recommended
- **Disk**: 500MB for embeddings cache
- **Network**: Internet access for API calls

### Optional Requirements
- **Node.js**: 16+ (for droid2api integration)
- **Git**: For cloning repositories and plugins

---

## Python Dependencies

### Core Dependencies (Auto-installed)
```txt
openai>=1.0.0           # OpenAI SDK
aiofiles>=23.0.0        # Async file operations
prompt_toolkit>=3.0.0   # Interactive CLI
networkx>=3.0           # Dependency graphs (for plugins)
python-dotenv>=1.0.0    # Environment variable loading
watchdog>=3.0.0         # File system monitoring
```

### Installation Methods

#### Method 1: Automatic (Recommended)
```bash
# Jinx auto-installs dependencies at runtime
python jinx.py
```

#### Method 2: Manual Pre-installation
```bash
# Install all dependencies upfront
pip install openai aiofiles prompt_toolkit networkx python-dotenv watchdog
```

#### Method 3: Using pip from requirements list
```bash
# Create requirements.txt
cat > requirements.txt << 'EOF'
openai>=1.0.0
aiofiles>=23.0.0
prompt_toolkit>=3.0.0
networkx>=3.0
python-dotenv>=1.0.0
watchdog>=3.0.0
EOF

# Install
pip install -r requirements.txt
```

---

## Configuration Requirements

### 1. API Keys

#### For OpenAI (Direct)
```bash
# In .env file:
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4.1
```

#### For Z.AI (via droid2api)
```bash
# Z.AI API key
OPENAI_API_KEY=665b963943b647dc9501dff942afb877.A47LrMc7sgGjyfBJ

# Point to droid2api
OPENAI_BASE_URL=http://localhost:8765/v1

# Z.AI model
OPENAI_MODEL=glm-4.6
```

### 2. Optional Configurations

#### Vector Store (for RAG)
```bash
# If using OpenAI Vector Stores
OPENAI_VECTOR_STORE_ID=vs_xxxxxxxxxxxxxxxxxxxxxxxx
```

#### Proxy Support
```bash
# SOCKS5/HTTP proxy
PROXY=socks5://127.0.0.1:12334
```

#### Embeddings
```bash
# Enable project code embeddings
EMBED_PROJECT_ENABLE=true
EMBED_PROJECT_ROOT=.
```

---

## droid2api Setup (for Z.AI)

### Requirements
```json
{
  "node": ">=16.0.0",
  "npm": ">=8.0.0",
  "dependencies": {
    "express": "^4.18.2",
    "node-fetch": "^3.3.2"
  }
}
```

### Installation Steps

#### 1. Clone droid2api
```bash
git clone https://github.com/Zeeeepa/droid2api
cd droid2api
```

#### 2. Install Dependencies
```bash
npm install
```

#### 3. Configure (Optional)
```bash
# Edit config.json if you want to change port (default: 3000)
# Our setup uses port 8765 to avoid conflicts
cat > config.json << 'EOF'
{
  "port": 8765,
  "model_redirects": {},
  "endpoint": [
    {
      "name": "anthropic",
      "base_url": "https://api.z.ai/api/anthropic/v1/messages"
    }
  ],
  "models": [
    {
      "name": "GLM-4.6",
      "id": "glm-4.6",
      "type": "anthropic"
    }
  ],
  "dev_mode": true,
  "user_agent": "jinx-agent/1.0",
  "system_prompt": ""
}
EOF
```

#### 4. Start Server
```bash
npm start
```

Expected output:
```
[INFO] Configuration loaded successfully
[INFO] Dev mode: true
[INFO] No auth configuration found, will use client authorization headers
[INFO] Auth system initialized for client authorization mode
[INFO] Auth system initialized successfully
[INFO] Starting server on port 8765...
[INFO] ====================================
[INFO]     Server URL: http://localhost:8765
[INFO]     Endpoint: /v1/chat/completions
[INFO]     Models: GLM-4.6
[INFO] ====================================
```

#### 5. Test Connection
```bash
curl -X POST http://localhost:8765/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 665b963943b647dc9501dff942afb877.A47LrMc7sgGjyfBJ" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "glm-4.6",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

Expected response:
```json
{
  "id": "...",
  "type": "message",
  "role": "assistant",
  "model": "glm-4.6",
  "content": [{"type": "text", "text": "Hello! How can I help you today?"}],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 12, "output_tokens": 9}
}
```

---

## Complete Setup Guide

### Quick Start (5 Minutes)

#### 1. Clone Jinx
```bash
git clone https://github.com/Zeeeepa/agent.git
cd agent
```

#### 2. Create Virtual Environment (Recommended)
```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
py -m venv .venv
.venv\Scripts\activate
```

#### 3. Configure Environment
```bash
# Copy example config
cp .env.example .env

# Edit with your API key
vi .env  # or use your preferred editor
```

#### 4. Run Jinx
```bash
python jinx.py
```

---

### Advanced Setup (with droid2api + Z.AI)

#### Terminal 1: Start droid2api
```bash
cd droid2api
npm install
npm start
```

#### Terminal 2: Configure & Run Jinx
```bash
cd agent

# Option 1: Environment variables
export OPENAI_API_KEY="665b963943b647dc9501dff942afb877.A47LrMc7sgGjyfBJ"
export OPENAI_BASE_URL="http://localhost:8765/v1"
export OPENAI_MODEL="glm-4.6"
python jinx.py

# Option 2: Update .env file
cat >> .env << 'EOF'
OPENAI_API_KEY=665b963943b647dc9501dff942afb877.A47LrMc7sgGjyfBJ
OPENAI_BASE_URL=http://localhost:8765/v1
OPENAI_MODEL=glm-4.6
EOF
python jinx.py
```

---

## Verification Checklist

### ✅ Python Environment
```bash
# Check Python version (should be 3.9+)
python --version

# Check pip
pip --version

# Check virtual environment (optional but recommended)
which python  # Should show .venv path if activated
```

### ✅ Dependencies
```bash
# Test imports
python -c "import openai; print('OpenAI SDK:', openai.__version__)"
python -c "import aiofiles; print('aiofiles OK')"
python -c "import prompt_toolkit; print('prompt_toolkit OK')"
python -c "import networkx; print('networkx OK')"
```

### ✅ Configuration
```bash
# Check .env exists
ls -la .env

# Verify API key is set
grep OPENAI_API_KEY .env
```

### ✅ droid2api (if using Z.AI)
```bash
# Check Node.js version
node --version  # Should be 16+

# Check if droid2api is running
curl http://localhost:8765/

# Test API
curl -X POST http://localhost:8765/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"glm-4.6","messages":[{"role":"user","content":"test"}],"max_tokens":10}'
```

### ✅ Jinx Agent
```bash
# Dry run (should initialize without errors)
python -c "from jinx import orchestrator; print('Jinx imports OK')"

# Full run
python jinx.py
```

---

## Troubleshooting

### Issue: "Module not found"
**Solution:** Install missing dependencies
```bash
pip install openai aiofiles prompt_toolkit networkx python-dotenv watchdog
```

### Issue: "API key not found"
**Solution:** Check .env configuration
```bash
# Verify .env file exists and has the key
cat .env | grep OPENAI_API_KEY

# Or set as environment variable
export OPENAI_API_KEY="your-key-here"
```

### Issue: droid2api port conflict
**Solution:** Change port in config.json
```bash
cd droid2api
# Edit config.json, change "port": 3000 to different port
# Update OPENAI_BASE_URL in Jinx accordingly
```

### Issue: Connection refused to droid2api
**Solution:** Ensure droid2api is running
```bash
# Check if process is running
ps aux | grep node

# Check if port is listening
lsof -i :8765

# Restart droid2api
cd droid2api
npm start
```

### Issue: Embeddings taking too much space
**Solution:** Adjust embedding settings
```bash
# In .env:
EMBED_PROJECT_ENABLE=false  # Disable if not needed
# Or clear cache:
rm -rf emb/
```

---

## Performance Tuning

### For Low-Resource Environments
```bash
# Reduce embedding budgets
JINX_CHAINED_PRE_DIALOG_MS=50
JINX_CHAINED_PRE_CODE_MS=100
JINX_CHAINED_DIALOG_CTX_MS=80
JINX_CHAINED_PROJECT_CTX_MS=250

# Reduce verification budget
JINX_VERIFY_TOPK=3
JINX_VERIFY_MS=200

# Disable auto-commit
JINX_PATCH_AUTOCOMMIT=false
```

### For High-Performance Environments
```bash
# Increase embedding budgets
JINX_CHAINED_DIALOG_CTX_MS=200
JINX_CHAINED_PROJECT_CTX_MS=800
JINX_VERIFY_MS=600

# Enable all features
JINX_CHAINED_REASONING=true
JINX_CHAINED_REFLECT=true
JINX_CONTINUITY_ENABLE=true
JINX_STATEFRAME_ENABLE=true
JINX_VERIFY_AUTORUN=true
```

---

## Production Deployment Checklist

### Security
- [ ] API keys stored in environment variables (not in code)
- [ ] Proxy configured if needed
- [ ] File permissions restricted on .env
- [ ] Logs directory secured

### Performance
- [ ] Embedding budgets tuned for workload
- [ ] Cache directories configured
- [ ] Memory limits set if needed

### Monitoring
- [ ] Log rotation configured
- [ ] Disk space monitoring enabled
- [ ] API rate limits understood

### Backup
- [ ] Brain directory backed up (.jinx/brain/)
- [ ] Memory directory backed up (.jinx/memory/)
- [ ] Embeddings directory backed up (emb/)

---

## Summary

### Minimum Setup (OpenAI)
```bash
1. Python 3.9+
2. pip install openai
3. Create .env with OPENAI_API_KEY
4. Run: python jinx.py
```

### Full Setup (Z.AI via droid2api)
```bash
1. Python 3.9+ and Node.js 16+
2. pip install openai aiofiles prompt_toolkit networkx python-dotenv watchdog
3. Setup droid2api (npm install && npm start)
4. Configure .env with Z.AI credentials and OPENAI_BASE_URL
5. Run: python jinx.py
```

### Plugin Development
```bash
1. Complete Full Setup
2. pip install networkx (for dependency graphs)
3. Create plugin following examples/example_plugin.py
4. Load via: quick_integrate("plugin_path")
```

---

*Last updated: 2025-01-13*
*Tested with: Python 3.11, Node.js 18, droid2api 1.3.4, Z.AI glm-4.6*

