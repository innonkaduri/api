class MegaAgent:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    async def create_business(self, business_data):
        return {"status": "success", "business": business_data}

    async def process_chat_message(self, chat_data):
        return {"status": "success", "message": chat_data}
