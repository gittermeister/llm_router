# LLM Router for AWS Bedrock

An intelligent routing system that automatically selects the best LLM provider (Claude, Llama, Titan, etc.) based on task type, complexity, and priority. Deploy as an AWS Lambda function to optimize cost, speed, and quality for your AI workloads.

## 🎯 Features

- **Smart Routing**: Automatically selects the optimal model based on:
  - Task type (code generation, summarization, Q&A, etc.)
  - Complexity score (1-10)
  - Priority (cost, speed, or quality)
- **Multi-Provider Support**: Claude 3 (Opus/Sonnet/Haiku), Llama 3, Amazon Titan, Cohere, AI21
- **Automatic Failover**: Falls back to alternative providers if primary fails
- **Cost Tracking**: Records usage metrics in DynamoDB
- **Docker Deployment**: Easy containerized deployment to AWS Lambda

## 🚀 Quick Start

### Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Test locally
python picker.py
```

### Docker Deployment
```bash
# Build and test locally
docker build -t llm-router .
docker run -p 9000:8080 -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID llm-router


## 📋 Prerequisites

- AWS Account with Bedrock model access
- Docker (for containerized deployment)
- IAM role with Bedrock and DynamoDB permissions
- Python 3.11+

## 🔧 Configuration

### Request Format
```json
{
  "body": {
    "prompt": "Your question or task here",
    "task_type": "question_answering",  // or code_generation, summarization, etc.
    "priority": "quality",               // or cost, speed
    "max_tokens": 1000,
    "temperature": 0.7,
    "complexity_score": 5               // 1-10, helps select appropriate model
  }
}
```

### Supported Task Types
- `code_generation` - Writing code (prefers Claude)
- `text_analysis` - Analyzing text content
- `creative_writing` - Stories, articles, creative content
- `question_answering` - General Q&A
- `summarization` - Condensing text
- `translation` - Language translation

### Priority Options
- `cost` - Minimizes cost (uses Titan, Cohere)
- `speed` - Minimizes latency
- `quality` - Maximizes output quality (uses Claude Opus)

## 📁 Project Structure
```
├── picker.py               # Main Lambda handler and routing logic
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
```

## 🏗️ Architecture

```
Request → Lambda Router → Provider Selection → Bedrock API → Response
                ↓                                    ↓
           DynamoDB Metrics                  Automatic Failover
```

The router:
1. Analyzes incoming requests
2. Selects optimal provider based on task and priorities
3. Handles failures with automatic fallback
4. Tracks performance and costs

## 💰 Cost Optimization

- **Automatic model selection**: Uses cheaper models for simple tasks
- **Complexity-based routing**: Routes only complex queries to expensive models
- **Provider comparison**: Tracks costs across all providers
- **Usage metrics**: Monitor spending via DynamoDB

## 🛠️ Deployment Options

Deploy Using Docker Image + ECR + Lambda


## 📊 Monitoring





## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request



## 🔗 Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [Available Bedrock Models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html)
