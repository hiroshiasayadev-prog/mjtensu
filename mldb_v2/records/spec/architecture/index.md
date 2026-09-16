# Overview: MLDB v2 architecture

- **id**: `spec:mldb.v2.architecture`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2`

## What this is

Defines the stable responsibility boundaries of MLDB v2.

## Current contract

MLDB core is a protocol/control boundary around canonical records. It can resolve and validate
definitions, compile immutable plans, request execution through a backend port, collect backend
outcomes, validate formal results, and persist canonical history.

It does not execute a second operational scheduler beside the backend.

## Topics

| ref | responsibility |
|---|---|
| `spec:mldb.v2.architecture.responsibility_model` | Ownership split among Git, MLDB, backend, and object storage. |
| `spec:mldb.v2.architecture.component_model` | Internal component flow and dependency direction. |
