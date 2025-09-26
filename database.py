class DatabaseManager:
    def __init__(self):
        self.connected = False

    async def connect(self):
        self.connected = True
        return self.connected

    def is_connected(self):
        return self.connected

    async def get_agent_flags(self, business_id):
        return []
    
    async def save_agent_message(self, message):
        return "mock_message_id"
