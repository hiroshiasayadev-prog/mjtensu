FROM mldb-clearml-runner:torch2.5.1-cu124-v1

WORKDIR /repo
COPY mldb_v2 /repo/mldb_v2

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

ENTRYPOINT []
CMD ["python", "-m", "mldb_v2.services.runtime_registry.api"]
