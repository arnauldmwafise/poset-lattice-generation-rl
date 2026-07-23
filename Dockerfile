FROM public.ecr.aws/docker/library/python:3.11

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy the code layout directly into the container
COPY . .

# Run pytest inside the container
CMD ["python", "-m", "pytest"]
