"""
Expert Sessions API for the Business Development System
Handles session creation, management, and real-time updates
"""

import asyncio
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.business import ExpertSession, SessionMessage

logger = logging.getLogger(__name__)

class SessionRequest(BaseModel):
    business_type: str
    description: str
    requirements: str = ""
    budget: str = ""
    timeline: str = ""

class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str
    timestamp: str

class ExpertSessionsAPI:
    """API for managing expert consultation sessions"""
    
    def __init__(self, session_manager, mqtt_broker):
        self.session_manager = session_manager
        self.mqtt_broker = mqtt_broker
        
        self.app = FastAPI(
            title="Expert Sessions API",
            description="API for business development expert sessions",
            version="1.0.0"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "Expert Sessions API",
                "version": "1.0.0",
                "status": "running"
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.post("/api/sessions", response_model=SessionResponse)
        async def create_session(request: SessionRequest):
            """Create a new expert consultation session"""
            try:
                # Create session data
                session_data = {
                    "business_type": request.business_type,
                    "description": request.description,
                    "requirements": request.requirements,
                    "budget": request.budget,
                    "timeline": request.timeline,
                    "created_at": datetime.now().isoformat(),
                    "status": "pending"
                }
                
                # Create session in session manager
                session = await self.session_manager.create_session(session_data)
                
                # Send to MQTT for processing (non-blocking)
                if self.mqtt_broker:
                    try:
                        # Use asyncio.create_task to avoid blocking the response
                        asyncio.create_task(self._publish_to_mqtt_async(session_data, session.session_id))
                        logger.info(f"Session queued for MQTT publishing: {session.session_id}")
                    except Exception as mqtt_error:
                        logger.warning(f"Failed to queue MQTT publishing: {mqtt_error}")
                else:
                    logger.warning("MQTT broker not available, session created without MQTT")
                
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
    
    async def _publish_to_mqtt_async(self, session_data: Dict[str, Any], session_id: str):
        """Publish session data to MQTT asynchronously (non-blocking)"""
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
    
    def get_app(self):
        """Get the FastAPI app instance"""
        return self.app
