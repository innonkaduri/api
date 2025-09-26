"""
Consultation API Backend
Receives consultation requests from React frontend and sends them via MQTT to the broker
"""

import asyncio
import json
import logging
import sys
import uuid
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
import aiomqtt
from twilio.request_validator import RequestValidator

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.database import DatabaseManager
from storage.session_manager import SessionManager
from broker.mqtt_broker import MQTTBroker
from models.business import ExpertSession, SessionMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration flags
MOCK_MODE = os.getenv("API_MOCK_MODE", "false").lower() == "true"
logger.info(f"🔧 API Mode: {'MOCK' if MOCK_MODE else 'PRODUCTION'}")

# Import mock data
from mock_data import MOCK_DATA, get_mock_businesses, get_mock_agents, get_mock_agent_messages, get_mock_negotiations, get_mock_llm_decisions

# Pydantic models for API requests/responses
class ConsultationRequest(BaseModel):
    businessType: str
    description: str
    requirements: str = ""
    budget: str = ""
    timeline: str = ""

class ConsultationResponse(BaseModel):
    consultation_id: str
    status: str
    message: str
    timestamp: str

class SessionRequest(BaseModel):
    business_type: str
    description: str
    requirements: str = ""
    budget: str = ""
    timeline: str = ""
    location: str = ""
    execution_mode: str = "manual"

class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str
    timestamp: str

class AgentToggleRequest(BaseModel):
    business_id: str
    agent_id: str
    is_enabled: bool
    reason: Optional[str] = None

class AgentModeRequest(BaseModel):
    business_id: str
    agent_id: str
    mode: str  # "manual" or "automatic"
    reason: Optional[str] = None

class AgentStatusResponse(BaseModel):
    agent_id: str
    business_id: str
    is_enabled: bool
    mode: str = "manual"  # manual or automatic
    enabled_at: Optional[datetime]
    disabled_at: Optional[datetime]
    reason: Optional[str]
    enabled_by: str
    priority: int

class LLMDecisionResponse(BaseModel):
    decision_id: str
    business_id: str
    agents_needed: List[Dict[str, Any]]
    total_agents: int
    completion_estimate: str
    next_steps: str
    reasoning: str
    created_at: datetime
    llm_model: str
    confidence_score: Optional[float]

def get_mock_response(endpoint: str, business_id: str = None) -> Dict[str, Any]:
    """Get mock response for API endpoints"""
    if endpoint == "businesses":
        if business_id:
            # Return specific business
            businesses = get_mock_businesses()
            for business in businesses:
                if business["business_id"] == business_id:
                    return business
            return {"error": "Business not found"}
        else:
            # Return all businesses
            businesses = get_mock_businesses()
            return {"businesses": businesses, "count": len(businesses)}
    
    elif endpoint == "agents":
        agents = get_mock_agents()
        if business_id:
            # Return agents for specific business
            return {"agents": agents, "business_id": business_id}
        else:
            return {"agents": agents}
    
    elif endpoint == "agent_messages":
        messages = get_mock_agent_messages(business_id)
        if business_id:
            return {"messages": messages, "business_id": business_id}
        else:
            return {"messages": messages}
    
    elif endpoint == "negotiations":
        negotiations = get_mock_negotiations(business_id)
        if business_id:
            return {"negotiations": negotiations, "business_id": business_id}
        else:
            return {"negotiations": negotiations}
    
    elif endpoint == "llm_decisions":
        decisions = get_mock_llm_decisions(business_id)
        if business_id:
            return {"decisions": decisions, "business_id": business_id}
        else:
            return {"decisions": decisions}
    
    elif endpoint == "session_created":
        return {
            "session_id": f"mock_session_{uuid.uuid4().hex[:8]}",
            "status": "created",
            "message": "Mock session created successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    elif endpoint == "agent_toggled":
        return {
            "message": "Agent toggled successfully (mock)",
            "business_id": business_id,
            "agent_id": "mock_agent",
            "is_enabled": True,
            "timestamp": datetime.now().isoformat()
        }
    
    elif endpoint == "agent_mode_set":
        return {
            "message": "Agent mode set successfully (mock)",
            "business_id": business_id,
            "agent_id": "mock_agent",
            "mode": "automatic",
            "timestamp": datetime.now().isoformat()
        }
    
    elif endpoint == "negotiation_started":
        return {
            "status": "success",
            "message": "Negotiations started successfully (mock)",
            "business_id": business_id,
            "negotiation_id": f"mock_negotiation_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat()
        }
    
    else:
        return {"error": "Unknown endpoint", "endpoint": endpoint}

class ConsultationAPI:
    """API for handling business consultation requests"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Business Consultation API",
            description="API for business development consultation requests and agent management",
            version="1.0.0"
        )
        
        # Configure CORS for React frontend
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",  # React dev server (default port)
                "http://localhost:3001",  # React dev server (alternative port)
                "http://localhost:3002",  # React dev server (your current port)
                "http://localhost:3003",  # React dev server (alternative port)
                "http://localhost:3004",  # React dev server (alternative port)
                "http://localhost:3005",  # React dev server (alternative port)
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize managers
        self.db_manager = DatabaseManager()
        self.session_manager = SessionManager(self.db_manager)
        self.mqtt_broker = None  # Will be initialized later if needed
        self.mega_agent = None  # Will be initialized later if needed
        
        self._setup_routes()
        
        # Store consultations locally (in memory)
        self.consultations = []
        
        # MQTT configuration (optional)
        self.mqtt_host = "localhost"
        self.mqtt_port = 1883
        self.mqtt_enabled = True  # Disable MQTT by default
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "Business Consultation API",
                "version": "1.0.0",
                "status": "running"
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "mode": "MOCK" if MOCK_MODE else "PRODUCTION",
                "database_connected": self.db_manager.is_connected() if self.db_manager else False,
                "mqtt_available": self.mqtt_broker is not None
            }
        
        @self.app.post("/api/consultation", response_model=ConsultationResponse)
        async def create_consultation(request: ConsultationRequest):
            """Create a new consultation request"""
            try:
                # Generate consultation ID
                consultation_id = str(uuid.uuid4())
                
                # Create consultation data
                consultation_data = {
                    "consultation_id": consultation_id,
                    "business_type": request.businessType,
                    "description": request.description,
                    "requirements": request.requirements,
                    "budget": request.budget,
                    "timeline": request.timeline,
                    "timestamp": datetime.now().isoformat(),
                    "status": "pending"
                }
                
                # Store locally
                self.consultations.append(consultation_data)
                logger.info(f"Consultation stored locally: {consultation_id}")
                
                # Try to send to MQTT (optional, won't fail if MQTT is down)
                if self.mqtt_enabled:
                    try:
                        await self._send_to_mqtt(consultation_data)
                        logger.info(f"Consultation also sent to MQTT: {consultation_id}")
                    except Exception as mqtt_error:
                        logger.warning(f"MQTT failed, but consultation saved locally: {mqtt_error}")
                
                return ConsultationResponse(
                    consultation_id=consultation_id,
                    status="stored",
                    message="Consultation request stored successfully",
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error creating consultation: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create consultation: {str(e)}"
                )
        
        @self.app.get("/api/consultations")
        async def get_consultations():
            """Get all stored consultations"""
            return {
                "consultations": self.consultations,
                "count": len(self.consultations)
            }
        
        # Expert Sessions endpoints
        @self.app.post("/api/sessions", response_model=SessionResponse)
        async def create_session(request: SessionRequest):
            """Create a new expert consultation session"""
            try:
                # Debug: Log the request data
                logger.info(f"📋 Received session request:")
                logger.info(f"   business_type: {request.business_type}")
                logger.info(f"   description: {request.description}")
                logger.info(f"   requirements: {request.requirements}")
                logger.info(f"   budget: {request.budget}")
                logger.info(f"   timeline: {request.timeline}")
                logger.info(f"   location: {request.location}")
                logger.info(f"   execution_mode: {request.execution_mode}")
                
                # Mock mode response
                if MOCK_MODE:
                    logger.info("🎭 MOCK MODE: Returning mock session response")
                    mock_response = get_mock_response("session_created")
                    return SessionResponse(**mock_response)
                
                # Create session data
                session_data = {
                    "business_type": request.business_type,
                    "description": request.description,
                    "requirements": request.requirements,
                    "budget": request.budget,
                    "timeline": request.timeline,
                    "location": request.location,
                    "execution_mode": request.execution_mode,
                    "created_at": datetime.now().isoformat(),
                    "status": "pending"
                }
                
                # Create session in session manager
                session = await self.session_manager.create_session(session_data)
                
                # Add session_id to session_data for MQTT
                session_data["session_id"] = session.session_id
                
                # Send to MQTT for processing (non-blocking) - this will trigger mega agent
                logger.info(f"🚀 Attempting to send session to MQTT: {session.session_id}")
                logger.info(f"📋 Session data: {session_data}")
                try:
                    await self._send_to_mqtt(session_data)
                    logger.info(f"✅ Session sent to MQTT for mega agent processing: {session.session_id}")
                except Exception as mqtt_error:
                    logger.warning(f"❌ MQTT failed, trying direct mega agent processing: {mqtt_error}")
                    # Fallback: trigger mega agent directly
                    asyncio.create_task(self._trigger_mega_agent_processing(session.session_id, session_data))
                
                return SessionResponse(
                    session_id=session.session_id,
                    status="created",
                    message="Expert consultation session created successfully",
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error creating session: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create session: {str(e)}"
                )
        
        @self.app.get("/api/sessions/{session_id}")
        async def get_session(session_id: str):
            """Get session details"""
            try:
                session = await self.session_manager.get_session(session_id)
                if not session:
                    raise HTTPException(status_code=404, detail="Session not found")
                return session
            except Exception as e:
                logger.error(f"Error getting session: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get session: {str(e)}"
                )
        
        @self.app.get("/api/sessions")
        async def list_sessions():
            """List all sessions"""
            try:
                sessions = await self.session_manager.list_sessions()
                return {"sessions": sessions, "count": len(sessions)}
            except Exception as e:
                logger.error(f"Error listing sessions: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to list sessions: {str(e)}"
                )
        
        @self.app.get("/api/sessions/{session_id}/events")
        async def stream_session_events(session_id: str):
            """Stream real-time session events"""
            async def event_generator():
                try:
                    # Send initial connection message
                    yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"
                    
                    # Keep connection alive and send updates
                    while True:
                        # Check for session updates
                        session = await self.session_manager.get_session(session_id)
                        if session:
                            yield f"data: {json.dumps({'type': 'update', 'session': session.to_dict()})}\n\n"
                        
                        await asyncio.sleep(5)  # Update every 5 seconds
                        
                except Exception as e:
                    logger.error(f"Error in event stream: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        
        # Agent Management endpoints
        @self.app.get("/api/businesses/{business_id}/agents", response_model=List[AgentStatusResponse])
        async def get_agent_status(business_id: str):
            """Get agent status for a business"""
            try:
                logger.info(f"🔍 API: Getting agent status for business_id: {business_id}")
                logger.info(f"🔍 API: business_id type: {type(business_id)}")
                
                # Mock mode response
                if MOCK_MODE:
                    logger.info("🎭 MOCK MODE: Returning mock agent status")
                    mock_agents = get_mock_agents()
                    return [AgentStatusResponse(
                        agent_id=agent["agent_id"],
                        business_id=business_id,
                        is_enabled=agent["enabled"],
                        mode=agent["mode"],
                        last_activity=datetime.now().isoformat()
                    ) for agent in mock_agents]
                
                if not self.db_manager.is_connected():
                    logger.error("❌ Database not connected, attempting to reconnect...")
                    try:
                        await self.db_manager.connect()
                        logger.info("✅ Database reconnected successfully")
                    except Exception as db_error:
                        logger.error(f"❌ Failed to reconnect to database: {db_error}")
                        raise HTTPException(status_code=503, detail="Database not connected")
                
                logger.info(f"🔍 Database connected, querying agent flags...")
                logger.info(f"🔍 Looking for business_id: '{business_id}' (type: {type(business_id)})")
                agent_flags = await self.db_manager.get_agent_flags(business_id)
                logger.info(f"🔍 Found {len(agent_flags)} agent flags for business_id: {business_id}")
                if agent_flags:
                    logger.info(f"📋 Agent flags: {[f.get('agent_id', 'unknown') for f in agent_flags]}")
                    # Debug: Show the mode of each flag
                    for flag in agent_flags:
                        logger.info(f"   📋 {flag.get('agent_id')}: mode={flag.get('mode')}, enabled={flag.get('is_enabled')}")
                else:
                    logger.warning(f"❌ No agent flags found for business_id: {business_id}")
                
                if not agent_flags:
                    # Return default agent status if no flags exist
                    all_agents = ['experts', 'marketresearch', 'ideallocation', 'legal', 'suppliers', 
                                 'hr', 'contractors', 'government', 'interiordesign', 'negotiation']
                    return [
                        AgentStatusResponse(
                            agent_id=agent_id,
                            business_id=business_id,
                            is_enabled=False,
                            mode="manual",
                            enabled_at=None,
                            disabled_at=datetime.utcnow(),
                            reason="No flags found",
                            enabled_by="system",
                            priority=1
                        )
                        for agent_id in all_agents
                    ]
                
                logger.info(f"🔍 Creating AgentStatusResponse objects...")
                try:
                    response_data = []
                    for flag in agent_flags:
                        logger.info(f"🔍 Processing flag: {flag}")
                        logger.info(f"🔍 Flag type: {type(flag)}")
                        logger.info(f"🔍 Flag keys: {list(flag.keys()) if isinstance(flag, dict) else 'Not a dict'}")
                        
                        # Ensure we're working with a dictionary
                        if isinstance(flag, dict):
                            response_obj = AgentStatusResponse(
                                agent_id=flag.get('agent_id', 'unknown'),
                                business_id=flag.get('business_id', business_id),
                                is_enabled=flag.get('is_enabled', False),
                                mode=flag.get('mode', 'manual'),
                                enabled_at=flag.get('enabled_at'),
                                disabled_at=flag.get('disabled_at'),
                                reason=flag.get('reason', 'No reason provided'),
                                enabled_by=flag.get('enabled_by', 'unknown'),
                                priority=flag.get('priority', 1)
                            )
                            response_data.append(response_obj)
                            logger.info(f"✅ Created response for agent: {flag.get('agent_id', 'unknown')}")
                        else:
                            logger.error(f"❌ Flag is not a dictionary: {type(flag)} - {flag}")
                    
                    logger.info(f"✅ Returning {len(response_data)} agent status responses")
                    return response_data
                except Exception as e:
                    logger.error(f"❌ Error creating AgentStatusResponse: {e}")
                    logger.error(f"❌ Flag data: {agent_flags}")
                    import traceback
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    raise HTTPException(status_code=500, detail=f"Error creating response: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error getting agent status: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")
        
        @self.app.post("/api/businesses/{business_id}/agents/{agent_id}/toggle")
        async def toggle_agent(business_id: str, agent_id: str, request: AgentToggleRequest):
            """Toggle agent enable/disable status"""
            try:
                # Mock mode response
                if MOCK_MODE:
                    logger.info("🎭 MOCK MODE: Returning mock toggle response")
                    mock_response = get_mock_response("agent_toggled", business_id)
                    return mock_response
                
                if not self.db_manager.is_connected():
                    raise HTTPException(status_code=503, detail="Database not connected")
                
                # Update agent flag
                modified_count = await self.db_manager.update_agent_flag(
                    business_id=business_id,
                    agent_id=agent_id,
                    is_enabled=request.is_enabled,
                    enabled_by="user"
                )
                
                if modified_count == 0:
                    # Create new flag if it doesn't exist
                    flag_data = {
                        'agent_id': agent_id,
                        'business_id': business_id,
                        'is_enabled': request.is_enabled,
                        'mode': 'manual',  # Default to manual mode
                        'enabled_at': datetime.utcnow() if request.is_enabled else None,
                        'disabled_at': None if request.is_enabled else datetime.utcnow(),
                        'reason': request.reason or f"Manually {'enabled' if request.is_enabled else 'disabled'} by user",
                        'enabled_by': 'user',
                        'priority': 1
                    }
                    await self.db_manager.save_agent_flags([flag_data])
                
                return {
                    "message": f"Agent {agent_id} {'enabled' if request.is_enabled else 'disabled'} successfully",
                    "business_id": business_id,
                    "agent_id": agent_id,
                    "is_enabled": request.is_enabled,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error toggling agent: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to toggle agent: {str(e)}")
        
        @self.app.post("/api/businesses/{business_id}/agents/{agent_id}/mode")
        async def set_agent_mode(business_id: str, agent_id: str, request: AgentModeRequest):
            """Set agent mode (manual/automatic)"""
            try:
                # Mock mode response
                if MOCK_MODE:
                    logger.info("🎭 MOCK MODE: Returning mock mode set response")
                    mock_response = get_mock_response("agent_mode_set", business_id)
                    return mock_response
                
                if not self.db_manager.is_connected():
                    raise HTTPException(status_code=503, detail="Database not connected")
                
                if request.mode not in ["manual", "automatic"]:
                    raise HTTPException(status_code=400, detail="Mode must be 'manual' or 'automatic'")
                
                # Update agent mode
                modified_count = await self.db_manager.update_agent_mode(
                    business_id=business_id,
                    agent_id=agent_id,
                    mode=request.mode,
                    enabled_by="user"
                )
                
                if modified_count == 0:
                    # Create new flag if it doesn't exist
                    flag_data = {
                        'agent_id': agent_id,
                        'business_id': business_id,
                        'is_enabled': True,
                        'mode': request.mode,
                        'enabled_at': datetime.utcnow(),
                        'disabled_at': None,
                        'reason': request.reason or f"Mode set to {request.mode} by user",
                        'enabled_by': 'user',
                        'priority': 1
                    }
                    await self.db_manager.save_agent_flags([flag_data])
                
                return {
                    "message": f"Agent {agent_id} mode set to {request.mode} successfully",
                    "business_id": business_id,
                    "agent_id": agent_id,
                    "mode": request.mode,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error setting agent mode: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to set agent mode: {str(e)}")
        
        @self.app.get("/api/businesses/{business_id}/llm-decisions", response_model=List[LLMDecisionResponse])
        async def get_llm_decisions(business_id: str):
            """Get LLM decisions for a business"""
            try:
                if not self.db_manager.is_connected():
                    raise HTTPException(status_code=503, detail="Database not connected")
                
                decisions = await self.db_manager.get_llm_decisions(business_id)
                
                return [
                    LLMDecisionResponse(
                        decision_id=str(decision['_id']),
                        business_id=decision['business_id'],
                        agents_needed=decision.get('agents_needed', []),
                        total_agents=decision.get('total_agents', 0),
                        completion_estimate=decision.get('completion_estimate', ''),
                        next_steps=decision.get('next_steps', ''),
                        reasoning=decision.get('reasoning', ''),
                        created_at=decision.get('created_at', datetime.utcnow()),
                        llm_model=decision.get('llm_model', 'unknown'),
                        confidence_score=decision.get('confidence_score')
                    )
                    for decision in decisions
                ]
                
            except Exception as e:
                logger.error(f"Error getting LLM decisions: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get LLM decisions: {str(e)}")
        
        @self.app.get("/api/businesses/{business_id}/enabled-agents")
        async def get_enabled_agents(business_id: str):
            """Get list of enabled agents for a business"""
            try:
                if not self.db_manager.is_connected():
                    raise HTTPException(status_code=503, detail="Database not connected")
                
                enabled_agents = await self.db_manager.get_enabled_agents(business_id)
                
                return {
                    "business_id": business_id,
                    "enabled_agents": enabled_agents,
                    "count": len(enabled_agents),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting enabled agents: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get enabled agents: {str(e)}")
        
        @self.app.get("/api/debug/business-ids")
        async def debug_business_ids():
            """Debug endpoint to see all business IDs in the database"""
            try:
                if not self.db_manager.is_connected():
                    raise HTTPException(status_code=503, detail="Database not connected")
                
                collection = await self.db_manager.get_collection('agent_flags')
                cursor = collection.find({}, {'business_id': 1, 'agent_id': 1, 'is_enabled': 1})
                all_flags = await cursor.to_list(length=None)
                
                # Group by business_id
                business_ids = {}
                for flag in all_flags:
                    biz_id = flag.get('business_id')
                    if biz_id not in business_ids:
                        business_ids[biz_id] = []
                    business_ids[biz_id].append({
                        'agent_id': flag.get('agent_id'),
                        'is_enabled': flag.get('is_enabled')
                    })
                
                return {
                    "total_flags": len(all_flags),
                    "unique_business_ids": len(business_ids),
                    "business_ids": business_ids
                }
                
            except Exception as e:
                logger.error(f"Error getting business IDs: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get business IDs: {str(e)}")
        
        # Agent Conversation endpoints
        @self.app.get("/api/businesses/{business_id}/agents/{agent_id}/messages")
        async def get_agent_messages(business_id: str, agent_id: str):
            """Get messages for a specific agent"""
            try:
                if not self.db_manager.is_connected():
                    raise HTTPException(status_code=503, detail="Database not connected")
                
                messages = await self.db_manager.get_agent_messages(business_id, agent_id)
                return {
                    "business_id": business_id,
                    "agent_id": agent_id,
                    "messages": messages,
                    "count": len(messages)
                }
                
            except Exception as e:
                logger.error(f"Error getting agent messages: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get agent messages: {str(e)}")
        
        @self.app.get("/api/debug/database")
        async def debug_database():
            """Debug database connection and collections"""
            try:
                if not self.db_manager.is_connected():
                    return {"status": "error", "message": "Database not connected"}
                
                # Test basic database operations
                collection = await self.db_manager.get_collection('agent_messages')
                count = await collection.count_documents({})
                
                return {
                    "status": "success",
                    "database_connected": True,
                    "agent_messages_count": count,
                    "collections": ["agent_messages", "businesses", "agent_flags", "llm_decisions"]
                }
            except Exception as e:
                return {"status": "error", "message": str(e), "database_connected": False}
        
        @self.app.get("/api/debug/business/{business_id}")
        async def debug_business_data(business_id: str):
            """Debug endpoint to check business data in database"""
            try:
                if self.db_manager and self.db_manager.is_connected():
                    # Get business data
                    business_data = await self.db_manager.get_business_data(business_id)
                    
                    if business_data:
                        return {
                            "status": "success",
                            "business_id": business_id,
                            "business_data": business_data,
                            "location": business_data.get('location', 'NOT_FOUND'),
                            "budget": business_data.get('budget', 'NOT_FOUND'),
                            "timeline": business_data.get('timeline', 'NOT_FOUND'),
                            "requirements": business_data.get('requirements', 'NOT_FOUND'),
                            "business_type": business_data.get('business_type', 'NOT_FOUND'),
                            "description": business_data.get('description', 'NOT_FOUND')
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"Business data not found for ID: {business_id}"
                        }
                else:
                    return {
                        "status": "error",
                        "message": "Database not connected"
                    }
            except Exception as e:
                logger.error(f"Business debug error: {e}")
                return {
                    "status": "error",
                    "message": f"Error: {str(e)}"
                }
        
        @self.app.get("/api/businesses/{business_id}/conversations")
        async def get_agent_conversations(business_id: str):
            """Get all agent conversations for a business"""
            try:
                logger.info(f"📋 API: Getting conversations for business_id: {business_id}")
                
                # Mock mode response
                if MOCK_MODE:
                    logger.info("🎭 MOCK MODE: Returning mock conversations")
                    mock_messages = get_mock_agent_messages(business_id)
                    # Group messages by agent_id
                    conversations = {}
                    for msg in mock_messages:
                        agent_id = msg["agent_id"]
                        if agent_id not in conversations:
                            conversations[agent_id] = {
                                "agent_id": agent_id,
                                "business_id": business_id,
                                "messages": [],
                                "last_activity": msg["timestamp"]
                            }
                        conversations[agent_id]["messages"].append(msg)
                    
                    return {
                        "business_id": business_id,
                        "conversations": list(conversations.values()),
                        "total_conversations": len(conversations)
                    }
                
                if not self.db_manager.is_connected():
                    logger.error("Database not connected")
                    raise HTTPException(status_code=503, detail="Database not connected")
                
                logger.info("Database is connected, calling get_agent_conversations")
                conversations = await self.db_manager.get_agent_conversations(business_id)
                logger.info(f"📋 Retrieved {len(conversations)} conversations for business {business_id}")
                
                # Debug: Log conversation details
                for conv in conversations:
                    logger.info(f"   Agent {conv.get('agent_id')}: {len(conv.get('messages', []))} messages")
                
                return {
                    "business_id": business_id,
                    "conversations": conversations,
                    "count": len(conversations)
                }
                
            except Exception as e:
                logger.error(f"❌ API Error getting agent conversations: {e}")
                logger.error(f"   business_id: {business_id}")
                logger.error(f"   Error type: {type(e).__name__}")
                import traceback
                logger.error(f"   Traceback: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Failed to get agent conversations: {str(e)}")
        
        @self.app.post("/api/businesses/{business_id}/agents/{agent_id}/messages")
        async def send_message_to_agent(business_id: str, agent_id: str, message_data: dict):
            """Send a message to a specific agent"""
            try:
                if not self.db_manager.is_connected():
                    raise HTTPException(status_code=503, detail="Database not connected")
                
                # Create message
                message = {
                    "agent_id": agent_id,
                    "business_id": business_id,
                    "message_type": message_data.get("message_type", "text"),
                    "content": message_data.get("content", ""),
                    "timestamp": datetime.utcnow(),
                    "metadata": message_data.get("metadata", {})
                }
                
                message_id = await self.db_manager.save_agent_message(message)
                
                return {
                    "message_id": message_id,
                    "status": "sent",
                    "message": "Message sent to agent successfully"
                }
                
            except Exception as e:
                logger.error(f"Error sending message to agent: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")
        
        @self.app.post("/api/businesses/{business_id}/agents/{agent_id}/start")
        async def start_agent_execution(business_id: str, agent_id: str):
            """Start automatic execution of an agent's tasks"""
            try:
                logger.info(f"🚀 Starting automatic execution for agent {agent_id} in business {business_id}")
                
                if not self.db_manager.is_connected():
                    raise HTTPException(status_code=503, detail="Database not connected")
                
                # Get agent status to verify it's enabled and in automatic mode
                agent_flags = await self.db_manager.get_agent_flags(business_id)
                agent_flag = next((flag for flag in agent_flags if flag.get('agent_id') == agent_id), None)
                
                if not agent_flag:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found for business {business_id}")
                
                if not agent_flag.get('is_enabled', False):
                    raise HTTPException(status_code=400, detail=f"Agent {agent_id} is not enabled")
                
                if agent_flag.get('mode') != 'automatic':
                    raise HTTPException(status_code=400, detail=f"Agent {agent_id} is not in automatic mode")
                
                # Send message to MQTT broker to trigger agent execution
                await self._send_agent_execution_request(business_id, agent_id)
                
                return {
                    "status": "started",
                    "agent_id": agent_id,
                    "business_id": business_id,
                    "message": f"Agent {agent_id} execution started successfully"
                }
                
            except Exception as e:
                logger.error(f"Error starting agent execution: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to start agent execution: {str(e)}")
        
        @self.app.post("/api/chat")
        async def send_chat_message(request: dict):
            """General chat API - send message to any agent"""
            try:
                agent = request.get("agent")
                message = request.get("message")
                business_id = request.get("business_id", "default")
                
                if not agent or not message:
                    raise HTTPException(status_code=400, detail="Agent and message are required")
                
                if not self.db_manager.is_connected():
                    raise HTTPException(status_code=503, detail="Database not connected")
                
                # Create message data for MQTT
                chat_data = {
                    "agent": agent,
                    "message": message,
                    "business_id": business_id,
                    "message_type": "chat",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {
                        "source": "user",
                        "session_id": business_id
                    }
                }
                
                # Send to MQTT broker for processing
                try:
                    await self._send_to_mqtt(chat_data)
                    logger.info(f"✅ Chat message sent to MQTT for agent {agent}: {message}")
                except Exception as mqtt_error:
                    logger.warning(f"❌ MQTT failed, trying direct processing: {mqtt_error}")
                    # Fallback: process directly
                    asyncio.create_task(self._process_chat_message_directly(chat_data))
                
                # Save user message to database
                user_message = {
                    "agent_id": agent,
                    "business_id": business_id,
                    "message_type": "user_message",
                    "content": message,
                    "timestamp": datetime.utcnow(),
                    "metadata": {"source": "user"}
                }
                
                message_id = await self.db_manager.save_agent_message(user_message)
                
                return {
                    "message_id": message_id,
                    "status": "sent",
                    "agent": agent,
                    "message": "Message sent to agent successfully"
                }
                
            except Exception as e:
                logger.error(f"Error sending chat message: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to send chat message: {str(e)}")
        
        # WhatsApp Webhook Endpoints
        @self.app.post("/webhook/whatsapp")
        async def handle_whatsapp_webhook(request: Request):
            """Handle incoming WhatsApp messages from Twilio"""
            try:
                # Get the raw body for signature validation
                body = await request.body()
                form_data = await request.form()
                
                # Extract message data
                message_sid = form_data.get("MessageSid")
                from_number = form_data.get("From", "").replace("whatsapp:", "")
                to_number = form_data.get("To", "").replace("whatsapp:", "")
                message_body = form_data.get("Body", "")
                message_type = form_data.get("MessageType", "text")
                
                logger.info(f"📱 Received WhatsApp message from {from_number}: {message_body[:50]}...")
                
                # Find the expert by phone number
                expert_data = await self.db_manager.find_document(
                    'real_experts', 
                    {"phone": from_number}
                )
                
                if not expert_data:
                    logger.warning(f"❌ No expert found with phone number: {from_number}")
                    return PlainTextResponse("Expert not found", status_code=404)
                
                expert_id = expert_data.get("expert_id")
                business_id = expert_data.get("business_id")
                
                if not business_id:
                    logger.warning(f"❌ No business_id found for expert: {expert_id}")
                    return PlainTextResponse("Business not found", status_code=404)
                
                # Process the message with negotiation agent
                try:
                    from models.business import AgentTask
                    from agents.negotiation.agent import NegotiationAgent
                    from agents.negotiation.tools.env_config import NegotiationAgentConfig
                    
                    # Get Twilio configuration
                    twilio_config = NegotiationAgentConfig.get_twilio_config()
                    
                    # Initialize negotiation agent
                    negotiation_agent = NegotiationAgent(
                        db_manager=self.db_manager,
                        openai_api_key=os.getenv("OPENAI_API_KEY"),
                        twilio_config=twilio_config
                    )
                    
                    # Create task for processing expert response
                    task = AgentTask(
                        task_id=f"whatsapp_response_{message_sid}",
                        agent_name="negotiation",
                        business_id=business_id,
                        task_type="process_expert_response",
                        description=f"Process WhatsApp response from expert {expert_id}",
                        result={
                            "expert_id": expert_id,
                            "response_message": message_body,
                            "from_number": from_number,
                            "message_sid": message_sid,
                            "message_type": message_type
                        }
                    )
                    
                    # Process the response
                    result = await negotiation_agent.process_task(task)
                    
                    if result.get("status") == "success":
                        logger.info(f"✅ Successfully processed WhatsApp response from {expert_id}")
                        return PlainTextResponse("Message processed successfully")
                    else:
                        logger.error(f"❌ Failed to process WhatsApp response: {result.get('message')}")
                        return PlainTextResponse("Failed to process message", status_code=500)
                        
                except Exception as e:
                    logger.error(f"❌ Error processing WhatsApp message: {e}")
                    return PlainTextResponse("Internal server error", status_code=500)
                
            except Exception as e:
                logger.error(f"❌ Error in WhatsApp webhook: {e}")
                return PlainTextResponse("Webhook error", status_code=500)

        @self.app.get("/webhook/whatsapp")
        async def verify_webhook(request: Request):
            """Handle webhook verification (GET request)"""
            return PlainTextResponse("WhatsApp webhook is active")

        @self.app.post("/webhook/whatsapp/status")
        async def handle_message_status(request: Request):
            """Handle message status updates from Twilio"""
            try:
                form_data = await request.form()
                message_sid = form_data.get("MessageSid")
                status = form_data.get("MessageStatus")
                
                logger.info(f"📊 Message {message_sid} status: {status}")
                return PlainTextResponse("Status received")
                
            except Exception as e:
                logger.error(f"❌ Error handling message status: {e}")
                return PlainTextResponse("Status error", status_code=500)
        
        @self.app.get("/api/businesses/{business_id}/negotiations")
        async def get_negotiation_status(business_id: str):
            """Get negotiation status for a business"""
            try:
                # Mock mode response
                if MOCK_MODE:
                    logger.info("🎭 MOCK MODE: Returning mock negotiation status")
                    mock_negotiations = get_mock_negotiations(business_id)
                    return {
                        "business_id": business_id,
                        "negotiation_states": mock_negotiations,
                        "negotiation_results": mock_negotiations,
                        "total_negotiations": len(mock_negotiations)
                    }
                
                if not self.db_manager.is_connected():
                    raise HTTPException(status_code=503, detail="Database not connected")
                
                # Get negotiation states
                negotiation_states = await self.db_manager.find_documents(
                    'negotiation_states', 
                    {"business_id": business_id}
                )
                
                # Get negotiation results
                negotiation_results = await self.db_manager.find_documents(
                    'negotiation_results', 
                    {"business_id": business_id}
                )
                
                return {
                    "business_id": business_id,
                    "negotiation_states": negotiation_states,
                    "negotiation_results": negotiation_results,
                    "total_negotiations": len(negotiation_states)
                }
                
            except Exception as e:
                logger.error(f"Error getting negotiation status: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get negotiation status: {str(e)}")
        
        @self.app.post("/api/businesses/{business_id}/negotiations/start")
        async def start_negotiations(business_id: str):
            """Start negotiations with experts for a business"""
            try:
                # Mock mode response
                if MOCK_MODE:
                    logger.info("🎭 MOCK MODE: Returning mock negotiation start response")
                    mock_response = get_mock_response("negotiation_started", business_id)
                    return mock_response
                
                if not self.db_manager.is_connected():
                    raise HTTPException(status_code=503, detail="Database not connected")
                
                from models.business import AgentTask
                from agents.negotiation.agent import NegotiationAgent
                from agents.negotiation.tools.env_config import NegotiationAgentConfig
                
                # Get Twilio configuration
                twilio_config = NegotiationAgentConfig.get_twilio_config()
                
                # Initialize negotiation agent
                negotiation_agent = NegotiationAgent(
                    db_manager=self.db_manager,
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    twilio_config=twilio_config
                )
                
                # Create task for starting negotiations
                task = AgentTask(
                    task_id=f"start_negotiations_{business_id}",
                    agent_name="negotiation",
                    business_id=business_id,
                    task_type="negotiation_support",
                    description=f"Start negotiations with experts for business {business_id}",
                    result={}
                )
                
                # Start negotiations
                result = await negotiation_agent.process_task(task)
                
                return result
                
            except Exception as e:
                logger.error(f"Error starting negotiations: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to start negotiations: {str(e)}")
    
    async def _send_to_mqtt(self, data: Dict[str, Any]):
        """Send data to MQTT broker for mega agent processing"""
        try:
            # Use aiomqtt context manager for proper connection handling
            async with aiomqtt.Client(
                hostname=self.mqtt_host,
                port=self.mqtt_port,
                timeout=5,
                keepalive=30
            ) as client:
                
                # Determine topic based on message type
                if data.get("message_type") == "chat":
                    topic = "agent/chat"
                elif data.get("message_type") == "agent_execution":
                    topic = "agent/execution"
                else:
                    topic = "business/requests"
                
                # Format the data as expected by the mega agent
                if data.get("message_type") == "chat":
                    mqtt_payload = {
                        "agent": data.get("agent"),
                        "message": data.get("message"),
                        "business_id": data.get("business_id"),
                        "message_type": "chat",
                        "timestamp": data.get("timestamp"),
                        "metadata": data.get("metadata", {})
                    }
                elif data.get("message_type") == "agent_execution":
                    mqtt_payload = {
                        "agent": data.get("agent"),
                        "business_id": data.get("business_id"),
                        "message": data.get("message"),
                        "message_type": "agent_execution",
                        "timestamp": data.get("timestamp"),
                        "metadata": data.get("metadata", {})
                    }
                else:
                    mqtt_payload = {
                        "business_type": data.get("business_type", "unknown"),
                        "description": data.get("description", ""),
                        "requirements": data.get("requirements", ""),
                        "budget": data.get("budget", ""),
                        "timeline": data.get("timeline", ""),
                        "location": data.get("location", ""),
                        "execution_mode": data.get("execution_mode", "manual"),  # Include execution_mode
                        "session_id": data.get("session_id", ""),
                        "created_at": data.get("created_at", ""),
                        "status": data.get("status", "pending")
                    }
                
                payload = json.dumps(mqtt_payload)
                await client.publish(topic, payload, qos=0)  # QoS 0 for simplicity
                
                logger.info(f"✅ Published to MQTT topic: {topic}")
                logger.info(f"📤 MQTT payload: {mqtt_payload}")
                logger.info(f"🔗 MQTT broker: {self.mqtt_host}:{self.mqtt_port}")
                
                return True  # Return success
            
        except Exception as e:
            logger.warning(f"MQTT not available: {e}")
            return False  # Return failure
    
    async def _trigger_mega_agent_processing(self, session_id: str, session_data: Dict[str, Any]):
        """Trigger mega agent processing for the session"""
        try:
            logger.info(f"🔄 Starting direct mega agent processing for session: {session_id}")
            if not self.db_manager.is_connected():
                logger.warning("Database not connected, skipping mega agent processing")
                return
            
            # Import mega agent here to avoid circular imports
            from mega_agent.megaAgent import MegaAgent
            
            # Create mega agent instance
            logger.info(f"🤖 Creating mega agent instance...")
            mega_agent = MegaAgent(self.db_manager)
            logger.info(f"✅ Mega agent created successfully")
            
            # Create business plan data
            business_data = {
                "business_id": session_id,  # Use session_id as business_id for consistency
                "name": f"Business Consultation - {session_data.get('business_type', 'Unknown')}",
                "business_type": session_data.get('business_type', 'other'),
                "description": session_data.get('description', ''),
                "location": session_data.get('location', ''),
                "requirements": session_data.get('requirements', ''),
                "budget": session_data.get('budget', ''),
                "timeline": session_data.get('timeline', ''),
                "execution_mode": session_data.get('execution_mode', 'manual'),  # Pass execution_mode to BusinessPlan
                "mission_statement": f"Mission: {session_data.get('description', '')}",
                "vision_statement": f"Vision: {session_data.get('description', '')}",
                "business_stage": "idea",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "status": "active",
                "progress_percentage": 0.0,
                "assigned_agents": [],
                "completed_tasks": [],
                "pending_tasks": [],
                "llm_decisions": [],
                "agent_flags": []
            }
            
            logger.info(f"🔑 Sending business_id to mega agent: {session_id}")
            logger.info(f"📋 Business data: {business_data}")
            
            # Create business plan
            from models.business import BusinessPlan
            business_plan = BusinessPlan(**business_data)
            
            # Process with mega agent
            result = await mega_agent.create_business(business_data)
            logger.info(f"Mega agent processing completed for session {session_id}: {result}")
            
        except Exception as e:
            logger.error(f"Error triggering mega agent processing for session {session_id}: {e}")
    
    async def _process_chat_message_directly(self, chat_data: Dict[str, Any]):
        """Process chat message directly when MQTT is not available"""
        try:
            logger.info(f"🔄 Processing chat message directly for agent: {chat_data.get('agent')}")
            
            # Import mega agent here to avoid circular imports
            from mega_agent.megaAgent import MegaAgent
            
            # Create mega agent instance
            mega_agent = MegaAgent(self.db_manager)
            
            # Process chat message
            result = await mega_agent.process_chat_message(chat_data)
            logger.info(f"Chat message processed: {result}")
            
        except Exception as e:
            logger.error(f"Error processing chat message directly: {e}")
    
    async def _send_agent_execution_request(self, business_id: str, agent_id: str):
        """Send agent execution request through MQTT broker"""
        try:
            logger.info(f"🚀 Sending agent execution request for {agent_id} in business {business_id}")
            
            # Send initial progress message
            await self._send_agent_progress_message(business_id, agent_id, "Starting automatic execution...", "system")
            
            # Create MQTT message for agent execution
            message_data = {
                "agent": agent_id,
                "business_id": business_id,
                "message": f"Execute {agent_id} agent tasks automatically",
                "message_type": "agent_execution",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "execution_type": "automatic",
                    "agent_id": agent_id,
                    "business_id": business_id
                }
            }
            
            # Send to MQTT broker
            success = await self._send_to_mqtt(message_data)
            
            if success:
                logger.info(f"✅ Agent execution request sent to MQTT for {agent_id}")
                await self._send_agent_progress_message(business_id, agent_id, f"🤖 {agent_id.title()} agent execution started via MQTT...", "system")
            else:
                logger.warning(f"⚠️ MQTT failed, trying direct execution for {agent_id}")
                # Fallback: Execute directly if MQTT fails
                await self._execute_agent_directly(business_id, agent_id)
                
        except Exception as e:
            logger.error(f"Error sending agent execution request: {e}")
            await self._send_agent_progress_message(business_id, agent_id, f"❌ Error starting execution: {str(e)}", "error")
    
    async def _execute_agent_directly(self, business_id: str, agent_id: str):
        """Execute agent directly as fallback when MQTT fails"""
        try:
            logger.info(f"🔄 Executing {agent_id} agent directly (MQTT fallback)")
            
            # Import mega agent
            from mega_agent.megaAgent import MegaAgent
            
            # Create mega agent instance
            mega_agent = MegaAgent(self.db_manager)
            
            # Create task for the specific agent
            from models.business import AgentTask
            import uuid
            
            # Map agent_id to task_type
            task_type_mapping = {
                'experts': 'expert_analysis',
                'marketresearch': 'market_analysis', 
                'ideallocation': 'location_analysis',
                'legal': 'legal_setup',
                'suppliers': 'supply_chain',
                'hr': 'recruitment',
                'contractors': 'contractor_management',
                'government': 'government_compliance',
                'interiordesign': 'design_planning',
                'negotiation': 'negotiation_support'
            }
            
            task_type = task_type_mapping.get(agent_id, 'general_analysis')
            
            # Create task
            task = AgentTask(
                task_id=str(uuid.uuid4()),
                agent_name=agent_id,
                business_id=business_id,
                task_type=task_type,
                description=f"Direct execution of {agent_id} agent tasks",
                status='pending',
                priority=1
            )
            
            # Send progress message
            await self._send_agent_progress_message(business_id, agent_id, f"🔄 Executing {task_type} task directly...", "system")
            
            # Execute the task using mega agent
            result = await mega_agent._execute_task_with_agent(task.model_dump())
            
            if result.get('status') == 'success':
                result_content = result.get('result', 'Task completed successfully')
                # Truncate very long results for better readability
                if len(str(result_content)) > 500:
                    result_content = str(result_content)[:500] + "... (truncated)"
                await self._send_agent_progress_message(business_id, agent_id, f"✅ Task completed: {result_content}", "success")
                await self._send_agent_progress_message(business_id, agent_id, f"🎉 {agent_id.title()} agent execution completed successfully!", "completion")
            else:
                error_msg = result.get('error', 'Unknown error occurred')
                logger.error(f"❌ Direct agent execution failed: {error_msg}")
                await self._send_agent_progress_message(business_id, agent_id, f"❌ Task failed: {error_msg}", "error")
            
            logger.info(f"✅ Direct execution completed for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Error in direct agent execution: {e}")
            await self._send_agent_progress_message(business_id, agent_id, f"❌ Direct execution error: {str(e)}", "error")
    
    async def _send_agent_progress_message(self, business_id: str, agent_id: str, content: str, message_type: str):
        """Send progress message to agent chat"""
        try:
            message_data = {
                'agent_id': agent_id,
                'business_id': business_id,
                'message_type': 'agent_message',
                'content': content,
                'timestamp': datetime.utcnow(),
                'metadata': {
                    'auto_execution': True,
                    'message_type': message_type
                }
            }
            
            await self.db_manager.save_agent_message(message_data)
            logger.info(f"📝 Sent progress message to {agent_id}: {content}")
            
        except Exception as e:
            logger.error(f"Error sending progress message: {e}")
    
    async def _publish_to_mqtt_async(self, session_data: Dict[str, Any], session_id: str):
        """Publish session data to MQTT asynchronously (non-blocking)"""
        if not self.mqtt_broker:
            logger.info(f"MQTT broker not available, skipping publish for session: {session_id}")
            return
            
        try:
            mqtt_success = await self.mqtt_broker.publish_message(
                "business/requests",
                session_data
            )
            if mqtt_success:
                logger.info(f"Session successfully published to MQTT: {session_id}")
            else:
                logger.warning(f"Failed to publish session to MQTT: {session_id}")
        except Exception as mqtt_error:
            logger.warning(f"MQTT publishing failed for session {session_id}: {mqtt_error}")
    
    async def start(self):
        """Start the API server"""
        # Initialize database connection
        try:
            await self.db_manager.connect()
            logger.info("Database connected successfully")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            logger.warning("API will run without database functionality")
        
        # MQTT broker is running in main.py (port 8000)
        # We'll just publish directly to MQTT without starting our own broker
        logger.info("API will publish to MQTT broker running in main.py")
        
        import uvicorn
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

# Create API instance
api = ConsultationAPI()

# For direct execution
if __name__ == "__main__":
    # Use Windows-compatible event loop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(api.start())
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
    except Exception as e:
        logger.error(f"API server failed: {e}")
