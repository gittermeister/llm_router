# Use AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies if needed
# RUN yum install -y <package-name>

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY picker.py ${LAMBDA_TASK_ROOT}/



# Copy any additional modules or utilities
# COPY utils/ ${LAMBDA_TASK_ROOT}/utils/

# Set environment variables (can be overridden at runtime)
ENV AWS_REGION=us-east-1
ENV METRICS_TABLE_NAME=provider-metrics
ENV LOG_LEVEL=INFO


# Command to run the Lambda function
CMD ["picker.lambda_handler"]