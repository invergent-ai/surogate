# surogate/eval/targets/application.py
import httpx
from typing import Dict, Any, Optional
from surogate.utils.logger import get_logger

from .base import BaseTarget, TargetRequest, TargetResponse, TargetType

logger = get_logger()


class ApplicationTarget(BaseTarget):
    """Base class for LLM application targets."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get('endpoint')
        self.timeout = config.get('timeout', 120)

        self.client = httpx.Client(
            timeout=self.timeout,
            headers=self._build_headers()
        )

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers."""
        headers = {"Content-Type": "application/json"}

        # API key
        if 'api_key' in self.config:
            headers["Authorization"] = f"Bearer {self.config['api_key']}"

        # Custom headers
        headers.update(self.config.get('headers', {}))

        return headers

    def cleanup(self):
        """Close HTTP client."""
        self.client.close()


class RAGTarget(ApplicationTarget):
    """Target for RAG (Retrieval-Augmented Generation) systems."""

    def _validate_config(self):
        """Validate RAG configuration."""
        required = ['endpoint']
        for field in required:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

    def send_request(self, request: TargetRequest) -> TargetResponse:
        """Send request to RAG system."""
        try:
            # Build RAG payload
            payload = {
                "query": request.prompt,
                "top_k": request.parameters.get('top_k', 5) if request.parameters else 5
            }

            if request.parameters:
                payload.update(request.parameters)

            # Send request
            response = self.client.post(self.endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

            # Parse response
            return TargetResponse(
                content=data.get('answer', ''),
                raw_response=data,
                metadata={
                    'retrieved_docs': data.get('documents', []),
                    'num_retrieved': len(data.get('documents', []))
                }
            )

        except Exception as e:
            logger.error(f"RAG request error: {e}")
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                error=str(e)
            )

    def health_check(self) -> bool:
        """Check RAG system health."""
        try:
            health_endpoint = self.config.get('health_endpoint', f"{self.endpoint}/health")
            response = self.client.get(health_endpoint)
            return response.status_code == 200
        except:
            return False


class AgentTarget(ApplicationTarget):
    """Target for AI agent systems."""

    def _validate_config(self):
        """Validate agent configuration."""
        required = ['endpoint']
        for field in required:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

    def send_request(self, request: TargetRequest) -> TargetResponse:
        """Send request to agent system."""
        try:
            # Build agent payload
            payload = {
                "task": request.prompt or request.inputs.get('task'),
                "max_iterations": request.parameters.get('max_iterations', 10) if request.parameters else 10
            }

            if request.parameters:
                payload.update(request.parameters)

            # Send request
            response = self.client.post(self.endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

            # Parse response
            return TargetResponse(
                content=data.get('result', ''),
                raw_response=data,
                metadata={
                    'steps': data.get('steps', []),
                    'tools_used': data.get('tools_used', []),
                    'iterations': data.get('iterations', 0)
                }
            )

        except Exception as e:
            logger.error(f"Agent request error: {e}")
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                error=str(e)
            )

    def health_check(self) -> bool:
        """Check agent system health."""
        try:
            health_endpoint = self.config.get('health_endpoint', f"{self.endpoint}/health")
            response = self.client.get(health_endpoint)
            return response.status_code == 200
        except:
            return False


class ChatbotTarget(ApplicationTarget):
    """Target for chatbot systems."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.session_id = None
        self.conversation_history = []

    def _validate_config(self):
        """Validate chatbot configuration."""
        required = ['endpoint']
        for field in required:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

    def send_request(self, request: TargetRequest) -> TargetResponse:
        """Send request to chatbot."""
        try:
            # Build chatbot payload
            payload = {
                "message": request.prompt or request.messages[-1]['content'] if request.messages else "",
                "session_id": self.session_id,
                "history": self.conversation_history if self.config.get('include_history') else []
            }

            if request.parameters:
                payload.update(request.parameters)

            # Send request
            response = self.client.post(self.endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

            # Update conversation state
            bot_response = data.get('response', '')
            if self.config.get('track_conversation'):
                self.conversation_history.append({
                    "role": "user",
                    "content": payload["message"]
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": bot_response
                })

            if not self.session_id and 'session_id' in data:
                self.session_id = data['session_id']

            # Parse response
            return TargetResponse(
                content=bot_response,
                raw_response=data,
                metadata={
                    'session_id': self.session_id,
                    'turn_count': len(self.conversation_history) // 2
                }
            )

        except Exception as e:
            logger.error(f"Chatbot request error: {e}")
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                error=str(e)
            )

    def reset_conversation(self):
        """Reset conversation state."""
        self.session_id = None
        self.conversation_history = []

    def health_check(self) -> bool:
        """Check chatbot health."""
        try:
            health_endpoint = self.config.get('health_endpoint', f"{self.endpoint}/health")
            response = self.client.get(health_endpoint)
            return response.status_code == 200
        except:
            return False


class MCPTarget(ApplicationTarget):
    """Target for MCP (Model Context Protocol) systems."""

    def _validate_config(self):
        """Validate MCP configuration."""
        required = ['endpoint']
        for field in required:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

    def send_request(self, request: TargetRequest) -> TargetResponse:
        """Send request to MCP system."""
        try:
            # Build MCP payload (following MCP spec)
            payload = {
                "method": request.inputs.get('method', 'completion') if request.inputs else 'completion',
                "params": {
                    "prompt": request.prompt,
                }
            }

            if request.parameters:
                payload['params'].update(request.parameters)

            # Send request
            response = self.client.post(self.endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

            # Parse response
            return TargetResponse(
                content=data.get('result', {}).get('content', ''),
                raw_response=data,
                metadata={
                    'method': payload['method'],
                    'context_used': data.get('result', {}).get('context_used', False)
                }
            )

        except Exception as e:
            logger.error(f"MCP request error: {e}")
            return TargetResponse(
                content="",
                raw_response={},
                metadata={},
                error=str(e)
            )

    def health_check(self) -> bool:
        """Check MCP system health."""
        try:
            health_endpoint = self.config.get('health_endpoint', f"{self.endpoint}/health")
            response = self.client.get(health_endpoint)
            return response.status_code == 200
        except:
            return False