FROM mldb-clearml-runner:torch2.5.1-cu124-v1

WORKDIR /app
COPY mldb_v2/services/runtime_registry /app/runtime_registry

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

ENTRYPOINT []
CMD ["python", "-m", "runtime_registry.api"]
