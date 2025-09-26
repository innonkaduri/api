"""
Mock Data for Business Development API
Contains all mock data used when API_MOCK_MODE=true
"""

from datetime import datetime
from typing import Dict, List, Any

# Mock businesses data
MOCK_BUSINESSES = [
    {
        "business_id": "mock_business_001",
        "name": "Tech Startup - AI Platform",
        "business_type": "technology",
        "description": "AI-powered business automation platform",
        "location": "San Francisco, CA",
        "budget": "UNDER_50K",
        "timeline": "6_MONTHS",
        "status": "active",
        "progress_percentage": 75.0,
        "created_at": "2025-01-15T10:30:00Z"
    },
    {
        "business_id": "mock_business_002", 
        "name": "Restaurant - Pizza Bar",
        "business_type": "food_service",
        "description": "Artisanal pizza restaurant with delivery",
        "location": "New York, NY",
        "budget": "UNDER_100K",
        "timeline": "3_MONTHS",
        "status": "active",
        "progress_percentage": 45.0,
        "created_at": "2025-01-10T14:20:00Z"
    },
    {
        "business_id": "mock_business_003",
        "name": "E-commerce - Fashion Store",
        "business_type": "retail",
        "description": "Online fashion store with sustainable clothing",
        "location": "Los Angeles, CA",
        "budget": "UNDER_25K",
        "timeline": "2_MONTHS",
        "status": "active",
        "progress_percentage": 30.0,
        "created_at": "2025-01-12T09:15:00Z"
    }
]

# Mock agents data
MOCK_AGENTS = [
    {"agent_id": "experts", "name": "Experts Agent", "enabled": True, "mode": "automatic"},
    {"agent_id": "marketresearch", "name": "Market Research Agent", "enabled": True, "mode": "automatic"},
    {"agent_id": "ideallocation", "name": "Location Analysis Agent", "enabled": True, "mode": "manual"},
    {"agent_id": "legal", "name": "Legal Agent", "enabled": False, "mode": "manual"},
    {"agent_id": "suppliers", "name": "Suppliers Agent", "enabled": True, "mode": "automatic"},
    {"agent_id": "hr", "name": "HR Agent", "enabled": True, "mode": "manual"},
    {"agent_id": "contractors", "name": "Contractors Agent", "enabled": False, "mode": "manual"},
    {"agent_id": "government", "name": "Government Agent", "enabled": False, "mode": "manual"},
    {"agent_id": "interiordesign", "name": "Interior Design Agent", "enabled": True, "mode": "automatic"},
    {"agent_id": "negotiation", "name": "Negotiation Agent", "enabled": True, "mode": "automatic"}
]

# Mock agent messages data
MOCK_AGENT_MESSAGES = [
    {
        "agent_id": "experts",
        "business_id": "mock_business_001",
        "message_type": "llm_analysis",
        "content": "**Expert Analysis Complete**\n\nBased on your AI platform requirements, I recommend focusing on:\n1. Machine Learning Engineers\n2. Data Scientists\n3. Cloud Infrastructure Specialists\n4. Product Managers with AI experience\n\nThese experts will be crucial for building a robust AI platform that can scale effectively.",
        "timestamp": "2025-01-15T10:35:00Z",
        "metadata": {"analysis_type": "expert_categories", "llm_model": "gpt-4o-mini"}
    },
    {
        "agent_id": "marketresearch",
        "business_id": "mock_business_001", 
        "message_type": "llm_analysis",
        "content": "**Market Research Complete**\n\nMarket analysis shows:\n- AI automation market: $12.5B by 2025\n- Target customers: SMBs and enterprises\n- Competition: High but growing market\n- Pricing strategy: Freemium model recommended\n- Key competitors: Zapier, UiPath, Automation Anywhere",
        "timestamp": "2025-01-15T10:40:00Z",
        "metadata": {"analysis_type": "market_analysis", "llm_model": "gpt-4o-mini"}
    },
    {
        "agent_id": "ideallocation",
        "business_id": "mock_business_001",
        "message_type": "llm_analysis",
        "content": "**Location Analysis Complete**\n\nSan Francisco is ideal for your AI startup:\n- Access to top tech talent\n- Strong investor network\n- Proximity to major tech companies\n- High cost but high potential returns\n- Consider co-working spaces for initial setup",
        "timestamp": "2025-01-15T10:45:00Z",
        "metadata": {"analysis_type": "location_analysis", "llm_model": "gpt-4o-mini"}
    },
    {
        "agent_id": "experts",
        "business_id": "mock_business_002",
        "message_type": "llm_analysis",
        "content": "**Expert Analysis Complete**\n\nFor your pizza restaurant, I recommend:\n1. Head Chef with Italian cuisine expertise\n2. Restaurant Manager with NYC experience\n3. Marketing Specialist for delivery platforms\n4. Financial Advisor for restaurant operations\n\nThese roles are essential for a successful pizza business in New York.",
        "timestamp": "2025-01-10T14:25:00Z",
        "metadata": {"analysis_type": "expert_categories", "llm_model": "gpt-4o-mini"}
    },
    {
        "agent_id": "marketresearch",
        "business_id": "mock_business_002",
        "message_type": "llm_analysis",
        "content": "**Market Research Complete**\n\nPizza market in NYC analysis:\n- Market size: $2.8B annually in NYC\n- High competition but strong demand\n- Delivery platforms: DoorDash, Uber Eats, Grubhub\n- Target: Young professionals and families\n- Pricing: $15-25 per pizza average",
        "timestamp": "2025-01-10T14:30:00Z",
        "metadata": {"analysis_type": "market_analysis", "llm_model": "gpt-4o-mini"}
    }
]

# Mock negotiations data
MOCK_NEGOTIATIONS = [
    {
        "business_id": "mock_business_001",
        "expert_id": "expert_001",
        "expert_name": "Dr. Sarah Chen",
        "expertise": "Machine Learning",
        "status": "in_progress",
        "stage": "initial_contact",
        "messages_sent": 2,
        "last_message": "Hi Dr. Chen, we're interested in discussing a Machine Learning Engineer position for our AI platform. Are you available for a call?",
        "created_at": "2025-01-15T11:00:00Z",
        "phone": "+1-555-0123",
        "email": "sarah.chen@email.com"
    },
    {
        "business_id": "mock_business_001",
        "expert_id": "expert_002",
        "expert_name": "Michael Rodriguez",
        "expertise": "Cloud Infrastructure",
        "status": "completed",
        "stage": "offer_accepted",
        "messages_sent": 5,
        "last_message": "Thank you for the offer! I'm excited to join the team as Cloud Infrastructure Lead.",
        "created_at": "2025-01-15T11:30:00Z",
        "phone": "+1-555-0124",
        "email": "michael.rodriguez@email.com"
    },
    {
        "business_id": "mock_business_002",
        "expert_id": "expert_003",
        "expert_name": "Chef Marco Bianchi",
        "expertise": "Italian Cuisine",
        "status": "in_progress",
        "stage": "salary_negotiation",
        "messages_sent": 3,
        "last_message": "I'm interested in the Head Chef position. What's the salary range and benefits package?",
        "created_at": "2025-01-10T15:00:00Z",
        "phone": "+1-555-0125",
        "email": "chef.marco@email.com"
    }
]

# Mock LLM decisions data
MOCK_LLM_DECISIONS = [
    {
        "business_id": "mock_business_001",
        "decision_type": "agent_selection",
        "content": "Selected agents: experts, marketresearch, suppliers, hr, legal, interiordesign, negotiation",
        "reasoning": "AI platform requires technical expertise, market validation, supply chain for cloud services, HR for team building, legal for IP protection, design for user experience, and negotiation for partnerships",
        "timestamp": "2025-01-15T10:30:00Z",
        "llm_model": "gpt-4o-mini",
        "confidence_score": 0.92
    },
    {
        "business_id": "mock_business_002",
        "decision_type": "agent_selection",
        "content": "Selected agents: experts, marketresearch, ideallocation, suppliers, hr, legal, interiordesign, contractors",
        "reasoning": "Restaurant business needs culinary experts, market research for location and competition, location analysis for optimal placement, suppliers for ingredients, HR for staff, legal for permits, interior design for ambiance, and contractors for setup",
        "timestamp": "2025-01-10T14:20:00Z",
        "llm_model": "gpt-4o-mini",
        "confidence_score": 0.88
    },
    {
        "business_id": "mock_business_003",
        "decision_type": "business_analysis",
        "content": "E-commerce fashion store analysis complete",
        "reasoning": "Sustainable fashion is trending with 40% growth in eco-friendly clothing. Target demographic: 25-40 year olds, environmentally conscious consumers. Recommended focus on digital marketing and social media presence.",
        "timestamp": "2025-01-12T09:20:00Z",
        "llm_model": "gpt-4o-mini",
        "confidence_score": 0.85
    }
]

# Mock real experts data (for negotiation agent)
MOCK_REAL_EXPERTS = [
    {
        "_id": "expert_001",
        "expert_id": "expert_001",
        "business_id": "mock_business_001",
        "name": "Dr. Sarah Chen",
        "contact_person": "Dr. Sarah Chen",
        "email": "sarah.chen@email.com",
        "phone": "+1-555-0123",
        "website": "https://sarahchen-ml.com",
        "location": "San Francisco, CA",
        "specialization": "Machine Learning",
        "experience_years": 8,
        "rating": 4.8,
        "services_offered": ["ML Model Development", "AI Consulting", "Data Science Training"],
        "pricing": "150-200",
        "availability": "Available",
        "languages": ["English", "Mandarin"],
        "certifications": ["AWS ML Specialty", "Google ML Certificate"],
        "description": "Senior ML Engineer with 8 years experience in AI/ML, specializing in deep learning and computer vision",
        "expert_type": "machine_learning_engineer",
        "business_type_target": "technology",
        "contact_details_verified": True,
        "created_at": "2025-01-15T10:00:00Z",
        "updated_at": "2025-01-15T10:00:00Z",
        "status": "active"
    },
    {
        "_id": "expert_002",
        "expert_id": "expert_002",
        "business_id": "mock_business_001",
        "name": "Michael Rodriguez",
        "contact_person": "Michael Rodriguez",
        "email": "michael.rodriguez@email.com",
        "phone": "+1-555-0124",
        "website": "https://michael-rodriguez-cloud.com",
        "location": "San Francisco, CA",
        "specialization": "Cloud Infrastructure",
        "experience_years": 6,
        "rating": 4.7,
        "services_offered": ["AWS Architecture", "DevOps Consulting", "Cloud Migration"],
        "pricing": "120-180",
        "availability": "Available",
        "languages": ["English", "Spanish"],
        "certifications": ["AWS Solutions Architect", "Kubernetes Certified"],
        "description": "Cloud Infrastructure Specialist with expertise in AWS, Azure, and Kubernetes",
        "expert_type": "cloud_engineer",
        "business_type_target": "technology",
        "contact_details_verified": True,
        "created_at": "2025-01-15T10:05:00Z",
        "updated_at": "2025-01-15T10:05:00Z",
        "status": "active"
    },
    {
        "_id": "expert_003",
        "expert_id": "expert_003",
        "business_id": "mock_business_002",
        "name": "Chef Marco Bianchi",
        "contact_person": "Chef Marco Bianchi",
        "email": "chef.marco@email.com",
        "phone": "+1-555-0125",
        "website": "https://chefmarco-italian.com",
        "location": "New York, NY",
        "specialization": "Italian Cuisine",
        "experience_years": 15,
        "rating": 4.9,
        "services_offered": ["Menu Development", "Kitchen Management", "Staff Training"],
        "pricing": "80-120",
        "availability": "Available",
        "languages": ["English", "Italian"],
        "certifications": ["Culinary Institute of America", "Italian Cuisine Master"],
        "description": "Master Italian Chef with 15 years experience in authentic Italian cuisine and restaurant management",
        "expert_type": "head_chef",
        "business_type_target": "food_service",
        "contact_details_verified": True,
        "created_at": "2025-01-10T14:00:00Z",
        "updated_at": "2025-01-10T14:00:00Z",
        "status": "active"
    }
]

# Mock session data
MOCK_SESSIONS = [
    {
        "session_id": "mock_session_001",
        "business_type": "TECHNOLOGY",
        "description": "AI-powered business automation platform",
        "requirements": "Machine learning expertise, cloud infrastructure, user experience design",
        "budget": "UNDER_50K",
        "timeline": "6_MONTHS",
        "location": "San Francisco, CA",
        "execution_mode": "semi_automatic",
        "created_at": "2025-01-15T10:30:00Z",
        "status": "active"
    },
    {
        "session_id": "mock_session_002",
        "business_type": "FOOD_SERVICE",
        "description": "Artisanal pizza restaurant with delivery",
        "requirements": "Head chef, restaurant manager, marketing specialist",
        "budget": "UNDER_100K",
        "timeline": "3_MONTHS",
        "location": "New York, NY",
        "execution_mode": "manual",
        "created_at": "2025-01-10T14:20:00Z",
        "status": "active"
    }
]

# Consolidated mock data dictionary
MOCK_DATA = {
    "businesses": MOCK_BUSINESSES,
    "agents": MOCK_AGENTS,
    "agent_messages": MOCK_AGENT_MESSAGES,
    "negotiations": MOCK_NEGOTIATIONS,
    "llm_decisions": MOCK_LLM_DECISIONS,
    "real_experts": MOCK_REAL_EXPERTS,
    "sessions": MOCK_SESSIONS
}

def get_mock_businesses() -> List[Dict[str, Any]]:
    """Get mock businesses data"""
    return MOCK_BUSINESSES

def get_mock_agents() -> List[Dict[str, Any]]:
    """Get mock agents data"""
    return MOCK_AGENTS

def get_mock_agent_messages(business_id: str = None) -> List[Dict[str, Any]]:
    """Get mock agent messages, optionally filtered by business_id"""
    if business_id:
        return [msg for msg in MOCK_AGENT_MESSAGES if msg["business_id"] == business_id]
    return MOCK_AGENT_MESSAGES

def get_mock_negotiations(business_id: str = None) -> List[Dict[str, Any]]:
    """Get mock negotiations, optionally filtered by business_id"""
    if business_id:
        return [neg for neg in MOCK_NEGOTIATIONS if neg["business_id"] == business_id]
    return MOCK_NEGOTIATIONS

def get_mock_llm_decisions(business_id: str = None) -> List[Dict[str, Any]]:
    """Get mock LLM decisions, optionally filtered by business_id"""
    if business_id:
        return [dec for dec in MOCK_LLM_DECISIONS if dec["business_id"] == business_id]
    return MOCK_LLM_DECISIONS

def get_mock_real_experts(business_id: str = None) -> List[Dict[str, Any]]:
    """Get mock real experts, optionally filtered by business_id"""
    if business_id:
        return [exp for exp in MOCK_REAL_EXPERTS if exp["business_id"] == business_id]
    return MOCK_REAL_EXPERTS

def get_mock_sessions() -> List[Dict[str, Any]]:
    """Get mock sessions data"""
    return MOCK_SESSIONS
