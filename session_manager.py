class SessionManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.sessions = {}

    async def create_session(self, session_data):
        session_id = "mock_session"
        self.sessions[session_id] = session_data
        return type("Session", (), {"session_id": session_id})

    async def get_session(self, session_id):
        return self.sessions.get(session_id)

    async def list_sessions(self):
        return list(self.sessions.values())
