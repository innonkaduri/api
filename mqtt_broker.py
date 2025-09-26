class MQTTBroker:
    async def publish_message(self, topic, message):
        print(f"Mock publish to {topic}: {message}")
        return True
