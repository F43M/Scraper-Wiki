# Scaling Guidelines

This document provides high level recommendations for deploying Scraper Wiki in a distributed environment.

## Worker Sizing

- **API**: Typically 1-2 replicas are sufficient. Each instance requires about **0.5 CPU** and **512Mi** of memory.
- **Scraper Workers**: Start with the number of replicas defined in `cluster.yaml` (`workers: 4`). Each worker consumes roughly **1 CPU** and **1Gi** of memory.
- **Dask/Ray Workers**: Match the value in `cluster.yaml`. A baseline of **1 CPU** and **2Gi** per worker is recommended.

## Scheduler and Queue

- The Dask scheduler is lightweight and can run with **0.5 CPU** and **512Mi** memory.
- RabbitMQ should run with persistent storage (5–10Gi) and at least **0.5 CPU**.

## Storage

- Prometheus requires persistent volume of 10–20Gi for metrics retention.
- Application logs and datasets can be stored on network volumes or object storage such as S3 or GCS.

## Autoscaling

Horizontal Pod Autoscaling can be enabled on the worker deployments using CPU utilization targets. Increase the node count in your Terraform modules to ensure sufficient capacity.
