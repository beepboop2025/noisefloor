# Minimal image: the package has zero runtime dependencies, so this is
# essentially just Python plus six source files.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY noisefloor ./noisefloor
RUN pip install --no-cache-dir .
ENTRYPOINT ["noisefloor-mcp"]
