#!/bin/bash

# Pull the latest Docker image from Docker Hub
docker pull `${{ secrets.DOCKER_USERNAME }}/california-housing-api`

# Run the Docker container locally
docker run -d -p 8000:8000 `${{ secrets.DOCKER_USERNAME }}/california-housing-api`

echo "API service is running locally on http://localhost:8000"