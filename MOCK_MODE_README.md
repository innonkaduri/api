# API Mock Mode

The Business Development API supports a mock mode for testing and development without requiring a database connection or external services.

## Configuration

Set the environment variable `API_MOCK_MODE=true` to enable mock mode:

```bash
# Windows
set API_MOCK_MODE=true

# Linux/Mac
export API_MOCK_MODE=true
```

Or create a `.env` file in the `api/` directory:

```
API_MOCK_MODE=true
```

## Mock Data Structure

All mock data is organized in `api/mock_data.py` with the following structure:

### Data Files
- **`MOCK_BUSINESSES`** - Sample business data (3 businesses)
- **`MOCK_AGENTS`** - Agent configurations and statuses (10 agents)
- **`MOCK_AGENT_MESSAGES`** - Sample LLM analysis messages
- **`MOCK_NEGOTIATIONS`** - Negotiation data and expert contacts
- **`MOCK_LLM_DECISIONS`** - LLM decision history
- **`MOCK_REAL_EXPERTS`** - Expert profiles for negotiation agent
- **`MOCK_SESSIONS`** - Session data

### Helper Functions
- `get_mock_businesses()` - Get all businesses
- `get_mock_agents()` - Get all agents
- `get_mock_agent_messages(business_id)` - Get messages, optionally filtered
- `get_mock_negotiations(business_id)` - Get negotiations, optionally filtered
- `get_mock_llm_decisions(business_id)` - Get decisions, optionally filtered
- `get_mock_real_experts(business_id)` - Get experts, optionally filtered
- `get_mock_sessions()` - Get all sessions

## Mock Data Content

When in mock mode, the API returns predefined mock data for all endpoints:

### Businesses
- `mock_business_001`: Tech Startup - AI Platform (75% progress)
- `mock_business_002`: Restaurant - Pizza Bar (45% progress)  
- `mock_business_003`: E-commerce - Fashion Store (30% progress)

### Agents
- 10 predefined agents with different enabled/disabled states and modes
- Mix of automatic and manual modes

### Agent Messages
- Sample LLM analysis messages from experts and market research agents
- Realistic content for testing the chat interface

### Negotiations
- Sample negotiation data with expert contact information
- Different negotiation stages and statuses

## Endpoints with Mock Data

All major endpoints support mock mode:

- `POST /api/sessions` - Creates mock session
- `GET /api/businesses/{business_id}/agents` - Returns mock agent status
- `POST /api/businesses/{business_id}/agents/{agent_id}/toggle` - Mock toggle response
- `POST /api/businesses/{business_id}/agents/{agent_id}/mode` - Mock mode change
- `GET /api/businesses/{business_id}/conversations` - Returns mock conversations
- `GET /api/businesses/{business_id}/negotiations` - Returns mock negotiations
- `POST /api/businesses/{business_id}/negotiations/start` - Mock negotiation start
- `GET /health` - Shows current mode (MOCK/PRODUCTION)

## Benefits

1. **No Database Required**: Test the frontend without MongoDB
2. **No External Services**: No need for OpenAI, Twilio, or MQTT
3. **Consistent Data**: Same mock data every time for reliable testing
4. **Fast Response**: Instant responses without processing delays
5. **Development Friendly**: Perfect for frontend development and testing

## Usage

1. Set `API_MOCK_MODE=true`
2. Start the API server: `python api/consultation_api.py`
3. All API calls will return mock data
4. Check `/health` endpoint to confirm mock mode is active

## Switching Modes

- **Mock Mode**: `API_MOCK_MODE=true` - Returns mock data
- **Production Mode**: `API_MOCK_MODE=false` - Uses real database and services

The API will log which mode it's running in at startup:
```
🔧 API Mode: MOCK
```
or
```
🔧 API Mode: PRODUCTION
```
