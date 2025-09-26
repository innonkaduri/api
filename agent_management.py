"""
Agent Management API for Business Development System
Handles agent enable/disable functionality and LLM decision retrieval
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.database import DatabaseManager

logger = logging.getLogger(__name__)

class AgentToggleRequest(BaseModel):
    business_id: str
    agent_id: str
    is_enabled: bool
    reason: Optional[str] = None

class AgentStatusResponse(BaseModel):
    agent_id: str
    business_id: str
    is_enabled: bool
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

class AgentManagementAPI:
    """API for managing agent flags and LLM decisions"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
        self.app = FastAPI(
            title="Agent Management API",
            description="API for managing agent enable/disable flags and LLM decisions",
            version="1.0.0"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",
                "http://localhost:3001", 
                "http://localhost:3002",
                "http://localhost:3003",
                "http://localhost:3004",
                "http://localhost:3005"
            ],
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
                "message": "Agent Management API",
                "version": "1.0.0",
                "status": "running"
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/api/businesses/{business_id}/agents", response_model=List[AgentStatusResponse])
        async def get_agent_status(business_id: str):
            """Get agent status for a business"""
            try:
                if not self.db_manager.is_connected():
                    raise HTTPException(status_code=503, detail="Database not connected")
                
                agent_flags = await self.db_manager.get_agent_flags(business_id)
                
                if not agent_flags:
                    # Return default agent status if no flags exist
                    all_agents = ['experts', 'marketresearch', 'ideallocation', 'legal', 'suppliers', 
                                 'hr', 'contractors', 'government', 'interiordesign', 'negotiation']
                    return [
                        AgentStatusResponse(
                            agent_id=agent_id,
                            business_id=business_id,
                            is_enabled=False,
                            enabled_at=None,
                            disabled_at=datetime.utcnow(),
                            reason="No flags found",
                            enabled_by="system",
                            priority=1
                        )
                        for agent_id in all_agents
                    ]
                
                return [
                    AgentStatusResponse(
                        agent_id=flag['agent_id'],
                        business_id=flag['business_id'],
                        is_enabled=flag['is_enabled'],
                        enabled_at=flag.get('enabled_at'),
                        disabled_at=flag.get('disabled_at'),
                        reason=flag.get('reason'),
                        enabled_by=flag.get('enabled_by', 'unknown'),
                        priority=flag.get('priority', 1)
                    )
                    for flag in agent_flags
                ]
                
            except Exception as e:
                logger.error(f"Error getting agent status: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")
        
        @self.app.post("/api/businesses/{business_id}/agents/{agent_id}/toggle")
        async def toggle_agent(business_id: str, agent_id: str, request: AgentToggleRequest):
            """Toggle agent enable/disable status"""
            try:
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
    
    def get_app(self):
        """Get the FastAPI app instance"""
        return self.app

# For direct execution
if __name__ == "__main__":
    import uvicorn
    
    # Create database manager
    db_manager = DatabaseManager()
    
    # Create API instance
    api = AgentManagementAPI(db_manager)
    
    # Use Windows-compatible event loop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        uvicorn.run(api.app, host="0.0.0.0", port=8002, log_level="info")
    except KeyboardInterrupt:
        logger.info("Agent Management API stopped by user")
    except Exception as e:
        logger.error(f"Agent Management API failed: {e}")
