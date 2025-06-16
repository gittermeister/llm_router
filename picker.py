import json
import boto3
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import time
import os
import traceback

# Configure logging with more detailed format for CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Remove file handler since Lambda has read-only filesystem
# CloudWatch Logs will automatically capture all logging output

class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    TEXT_ANALYSIS = "text_analysis"
    CREATIVE_WRITING = "creative_writing"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"

class Priority(Enum):
    COST = "cost"
    SPEED = "speed"
    QUALITY = "quality"

@dataclass
class Request:
    prompt: str
    task_type: TaskType
    priority: Priority
    max_tokens: int = 1000
    temperature: float = 0.7
    user_id: str = ""
    complexity_score: int = 5  # 1-10 scale
    
class ProviderResponse:
    def __init__(self, content: str, provider: str, cost: float, latency: float):
        self.content = content
        self.provider = provider
        self.cost = cost
        self.latency = latency
        self.timestamp = time.time()

class BaseProvider(ABC):
    def __init__(self, name: str, cost_per_token: float, avg_latency: float):
        self.name = name
        self.cost_per_token = cost_per_token
        self.avg_latency = avg_latency
        self.is_available = True
        logger.info(f"Initialized provider {name} with cost_per_token={cost_per_token}, avg_latency={avg_latency}")
        
    @abstractmethod
    def format_request(self, request: Request) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def parse_response(self, response: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    def get_model_id(self, request: Request) -> str:
        pass
    
    def process(self, request: Request, bedrock_client) -> ProviderResponse:
        start_time = time.time()
        logger.info(f"Processing request with {self.name} provider")
        logger.debug(f"Request details: task_type={request.task_type}, priority={request.priority}, complexity={request.complexity_score}")
        
        try:
            formatted_request = self.format_request(request)
            model_id = self.get_model_id(request)
            logger.info(f"Using model {model_id} for provider {self.name}")
            
            logger.debug(f"Sending request to Bedrock: {json.dumps(formatted_request)}")
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(formatted_request)
            )
            
            response_body = json.loads(response['body'].read())
            content = self.parse_response(response_body)
            
            latency = time.time() - start_time
            estimated_cost = self.estimate_cost(request.max_tokens)
            
            logger.info(f"Successfully processed request with {self.name}. Latency: {latency:.2f}s, Estimated cost: ${estimated_cost:.4f}")
            return ProviderResponse(content, self.name, estimated_cost, latency)
            
        except Exception as e:
            logger.error(f"Error with {self.name}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            self.is_available = False
            raise
    
    def estimate_cost(self, tokens: int) -> float:
        return tokens * self.cost_per_token

class AnthropicProvider(BaseProvider):
    def __init__(self):
        # Updated pricing for Claude models
        super().__init__("anthropic", 0.00001102, 2.5)
        
    def get_model_id(self, request: Request) -> str:
        # Using currently available Claude models on Bedrock
        if request.complexity_score >= 8:
            return "anthropic.claude-3-7-sonnet-20250219-v1:0"
        elif request.complexity_score >= 6:
            return "anthropic.claude-3-sonnet-20240229-v1:0"
        else:
            return "anthropic.claude-3-haiku-20240307-v1:0"
    
    def format_request(self, request: Request) -> Dict[str, Any]:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]
        }
    
    def parse_response(self, response: Dict[str, Any]) -> str:
        return response.get("content", [{}])[0].get("text", "")

class MetaProvider(BaseProvider):
    def __init__(self):
        super().__init__("meta", 0.000006, 1.8)
        
    def get_model_id(self, request: Request) -> str:
        # Using Llama models available on Bedrock
        if request.task_type == TaskType.CODE_GENERATION or request.complexity_score >= 7:
            return "meta.llama3-70b-instruct-v1:0"
        return "meta.llama3-8b-instruct-v1:0"
    
    def format_request(self, request: Request) -> Dict[str, Any]:
        return {
            "prompt": request.prompt,
            "max_gen_len": request.max_tokens,
            "temperature": request.temperature,
            "top_p": 0.9
        }
    
    def parse_response(self, response: Dict[str, Any]) -> str:
        return response.get("generation", "")

class CohereProvider(BaseProvider):
    def __init__(self):
        super().__init__("cohere", 0.000004, 1.2)
        
    def get_model_id(self, request: Request) -> str:
        # Using Cohere Command model available on Bedrock
        if request.complexity_score >= 7:
            return "cohere.command-text-v14"
        return "cohere.command-light-text-v14"
    
    def format_request(self, request: Request) -> Dict[str, Any]:
        return {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "p": 0.75,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        }
    
    def parse_response(self, response: Dict[str, Any]) -> str:
        generations = response.get("generations", [])
        return generations[0].get("text", "") if generations else ""

class AI21Provider(BaseProvider):
    def __init__(self):
        super().__init__("ai21", 0.000008, 2.1)
        
    def get_model_id(self, request: Request) -> str:
        # Using AI21 Jurassic models available on Bedrock
        if request.complexity_score >= 7:
            return "ai21.j2-ultra-v1"
        return "ai21.j2-mid-v1"
    
    def format_request(self, request: Request) -> Dict[str, Any]:
        return {
            "prompt": request.prompt,
            "maxTokens": request.max_tokens,
            "temperature": request.temperature,
            "topP": 1,
            "stopSequences": [],
            "countPenalty": {"scale": 0},
            "presencePenalty": {"scale": 0},
            "frequencyPenalty": {"scale": 0}
        }
    
    def parse_response(self, response: Dict[str, Any]) -> str:
        completions = response.get("completions", [])
        return completions[0].get("data", {}).get("text", "") if completions else ""

class TitanProvider(BaseProvider):
    def __init__(self):
        super().__init__("titan", 0.000003, 1.0)
        
    def get_model_id(self, request: Request) -> str:
        # Using Amazon Titan models
        if request.task_type == TaskType.TEXT_ANALYSIS:
            return "amazon.titan-text-express-v1"
        return "amazon.titan-text-lite-v1"
    
    def format_request(self, request: Request) -> Dict[str, Any]:
        return {
            "inputText": request.prompt,
            "textGenerationConfig": {
                "maxTokenCount": request.max_tokens,
                "temperature": request.temperature,
                "topP": 0.9,
                "stopSequences": []
            }
        }
    
    def parse_response(self, response: Dict[str, Any]) -> str:
        results = response.get("results", [])
        return results[0].get("outputText", "") if results else ""

class ProviderMetrics:
    def __init__(self):
        self.dynamodb = None
        self.metrics_table = None
        self.table_name = os.environ.get('METRICS_TABLE_NAME', 'provider-metrics')
        
        try:
            self.dynamodb = boto3.resource('dynamodb')
            self.metrics_table = self.dynamodb.Table(self.table_name)
        except Exception as e:
            logger.warning(f"Could not initialize DynamoDB metrics: {str(e)}")
    
    def record_usage(self, provider: str, latency: float, cost: float, success: bool):
        if not self.metrics_table:
            logger.debug("Metrics table not available, skipping metric recording")
            return
            
        try:
            self.metrics_table.put_item(
                Item={
                    'provider': provider,
                    'timestamp': int(time.time()),
                    'latency': latency,
                    'cost': cost,
                    'success': success,
                    'date': time.strftime('%Y-%m-%d')
                }
            )
        except Exception as e:
            logger.error(f"Failed to record metrics: {str(e)}")
    
    def get_provider_performance(self, provider: str, days: int = 7) -> Dict[str, float]:
        # Default values if metrics not available
        default_metrics = {
            'avg_latency': 2.0,
            'success_rate': 0.95,
            'avg_cost': 0.01
        }
        
        if not self.metrics_table:
            return default_metrics
            
        try:
            # Implementation would query DynamoDB for recent performance data
            # For now, return default values
            return default_metrics
        except Exception as e:
            logger.error(f"Failed to get provider metrics: {str(e)}")
            return default_metrics

class RouterEngine:
    def __init__(self):
        logger.info("Initializing RouterEngine")
        self.providers = {
            'anthropic': AnthropicProvider(),
            'meta': MetaProvider(),
            'cohere': CohereProvider(),
            'ai21': AI21Provider(),
            'titan': TitanProvider()
        }
        self.metrics = ProviderMetrics()
        
        region = os.environ.get('AWS_REGION', 'us-east-1')
        logger.info(f"Initializing Bedrock client in region: {region}")
        self.bedrock = boto3.client('bedrock-runtime', region_name=region)
        
    def select_provider(self, request: Request) -> str:
        logger.info(f"Selecting provider for task_type={request.task_type}, priority={request.priority}")
        
        selected_provider = None
        
        # Task-specific routing logic
        if request.task_type == TaskType.CODE_GENERATION:
            if request.priority == Priority.QUALITY:
                selected_provider = 'anthropic'
            elif request.priority == Priority.COST:
                selected_provider = 'meta'
            else:
                selected_provider = 'anthropic'
            logger.info(f"Selected {selected_provider} for code generation task")
                
        elif request.task_type == TaskType.CREATIVE_WRITING:
            # Claude and Llama are good for creative tasks
            if request.priority == Priority.QUALITY:
                selected_provider = 'anthropic'
            else:
                selected_provider = 'meta'
                
        elif request.task_type == TaskType.TEXT_ANALYSIS:
            if request.complexity_score >= 7:
                selected_provider = 'anthropic'
            elif request.priority == Priority.COST:
                selected_provider = 'titan'  # Amazon Titan for simple analysis
            else:
                selected_provider = 'cohere'
                
        elif request.task_type == TaskType.SUMMARIZATION:
            if request.priority == Priority.COST:
                selected_provider = 'titan'
            else:
                selected_provider = 'cohere'
                
        elif request.task_type == TaskType.TRANSLATION:
            selected_provider = 'anthropic'  # Claude handles multiple languages well
            
        if not selected_provider:
            selected_provider = self._get_best_available(request)
            logger.info(f"Selected {selected_provider} based on best available criteria")
            
        return selected_provider
    
    def _get_best_available(self, request: Request) -> str:
        """Score providers based on multiple factors"""
        scores = {}
        
        for name, provider in self.providers.items():
            if not provider.is_available:
                continue
                
            # Calculate composite score (lower is better)
            cost_score = provider.cost_per_token * request.max_tokens
            latency_score = provider.avg_latency
            
            # Get performance metrics
            metrics = self.metrics.get_provider_performance(name)
            reliability_score = 1.0 - metrics['success_rate']
            
            # Weight factors based on request priority
            if request.priority == Priority.COST:
                scores[name] = (cost_score * 0.6 + 
                              latency_score * 0.2 + 
                              reliability_score * 0.2)
            elif request.priority == Priority.SPEED:
                scores[name] = (cost_score * 0.2 + 
                              latency_score * 0.6 + 
                              reliability_score * 0.2)
            else:  # Quality priority
                scores[name] = (cost_score * 0.3 + 
                              latency_score * 0.2 + 
                              reliability_score * 0.2 +
                              (10 - request.complexity_score) * 0.3)
        
        return min(scores.keys(), key=lambda x: scores[x]) if scores else 'anthropic'
    
    def process_request(self, request: Request) -> ProviderResponse:
        logger.info(f"Processing new request: task_type={request.task_type}, priority={request.priority}")
        logger.debug(f"Full request details: {request.__dict__}")
        
        primary_provider = self.select_provider(request)
        logger.info(f"Selected primary provider: {primary_provider}")
        
        provider_order = [primary_provider]
        fallback_providers = [name for name in self.providers.keys() 
                            if name != primary_provider and self.providers[name].is_available]
        
        fallback_providers.sort(key=lambda x: self.providers[x].cost_per_token)
        provider_order.extend(fallback_providers)
        logger.info(f"Provider fallback order: {provider_order}")
        
        last_error = None
        for provider_name in provider_order:
            try:
                logger.info(f"Attempting provider: {provider_name}")
                provider = self.providers[provider_name]
                
                response = provider.process(request, self.bedrock)
                self.metrics.record_usage(provider_name, response.latency, response.cost, True)
                
                logger.info(f"Successfully processed with {provider_name}")
                logger.debug(f"Response details: latency={response.latency:.2f}s, cost=${response.cost:.4f}")
                return response
                
            except Exception as e:
                last_error = e
                logger.error(f"Provider {provider_name} failed: {str(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                self.metrics.record_usage(provider_name, 0, 0, False)
                
                self.providers[provider_name].is_available = False
                continue
        
        logger.error(f"All providers failed. Last error: {str(last_error)}")
        raise Exception(f"All providers failed. Last error: {str(last_error)}")

def lambda_handler(event, context):
    logger.info("Received new Lambda invocation")
    logger.debug(f"Event: {json.dumps(event)}")
    
    try:
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event.get('body', {})
        logger.info(f"Processing request for user: {body.get('user_id', 'unknown')}")
        
        if not body.get('prompt'):
            logger.error("Missing required field: prompt")
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Missing required field: prompt',
                    'message': 'Request must include a prompt'
                })
            }
        
        request = Request(
            prompt=body.get('prompt', ''),
            task_type=TaskType(body.get('task_type', 'question_answering')),
            priority=Priority(body.get('priority', 'quality')),
            max_tokens=min(body.get('max_tokens', 1000), 4000),
            temperature=max(0, min(body.get('temperature', 0.7), 1.0)),
            user_id=body.get('user_id', ''),
            complexity_score=max(1, min(body.get('complexity_score', 5), 10))
        )
        logger.info(f"Created request object: task_type={request.task_type}, priority={request.priority}")
        
        router = RouterEngine()
        response = router.process_request(request)
        
        logger.info(f"Successfully processed request with provider: {response.provider}")
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'content': response.content,
                'provider': response.provider,
                'cost': response.cost,
                'latency': response.latency,
                'timestamp': response.timestamp
            })
        }
        
    except ValueError as e:
        logger.error(f"Invalid request format: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'message': 'Invalid request format'
            })
        }
        
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'message': 'Request processing failed'
            })
        }

# Example usage for testing
if __name__ == "__main__":
    # Test the router locally
    test_request = Request(
        prompt="Explain quantum computing in simple terms",
        task_type=TaskType.QUESTION_ANSWERING,
        priority=Priority.QUALITY,
        complexity_score=7
    )
    
    router = RouterEngine()
    try:
        response = router.process_request(test_request)
        print(f"Response from {response.provider}:")
        print(f"Content: {response.content[:200]}...")
        print(f"Cost: ${response.cost:.4f}")
        print(f"Latency: {response.latency:.2f}s")
    except Exception as e:
        print(f"Error: {str(e)}")