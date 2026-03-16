# Monitoring

This guide covers the key metrics, limits, and alerting recommendations for
operating mlflow-dynamodbstore in production.

## CloudWatch Metrics

DynamoDB publishes metrics to CloudWatch automatically. These are the most
important ones to monitor:

### Capacity Metrics

| Metric                      | Dimension  | What to Watch                              |
|-----------------------------|------------|--------------------------------------------|
| `ConsumedReadCapacityUnits` | Table, GSI | Sustained spikes indicate hot partitions    |
| `ConsumedWriteCapacityUnits`| Table, GSI | Track write patterns during training runs   |
| `ThrottledRequests`         | Table, GSI | **Must be zero** in steady state            |
| `ReadThrottleEvents`        | Table, GSI | Read-side throttling                        |
| `WriteThrottleEvents`       | Table, GSI | Write-side throttling                       |

!!! warning "Throttling"
    Any non-zero `ThrottledRequests` means your application is hitting capacity
    limits. For on-demand tables, this indicates a partition-level throughput
    limit (3,000 RCU / 1,000 WCU per partition). For provisioned tables,
    consider increasing capacity or switching to on-demand.

### Latency Metrics

| Metric                    | What to Watch                                        |
|---------------------------|------------------------------------------------------|
| `SuccessfulRequestLatency`| p50 and p99 latency; spikes indicate large scans     |
| `SystemErrors`            | DynamoDB internal errors (should be near zero)        |
| `UserErrors`              | Client-side errors (e.g., validation, conditions)     |

### Example CloudWatch Dashboard

```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/DynamoDB", "ConsumedReadCapacityUnits", "TableName", "mlflow"],
          ["AWS/DynamoDB", "ConsumedWriteCapacityUnits", "TableName", "mlflow"]
        ],
        "period": 60,
        "stat": "Sum",
        "title": "Table Capacity"
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/DynamoDB", "ThrottledRequests", "TableName", "mlflow"]
        ],
        "period": 60,
        "stat": "Sum",
        "title": "Throttled Requests"
      }
    }
  ]
}
```

## Partition Size Monitoring

### 10 GB LSI Partition Limit

DynamoDB enforces a **10 GB limit per partition key** when Local Secondary
Indexes (LSIs) are present. mlflow-dynamodbstore uses LSIs, so this limit
applies.

Each experiment's data lives under a single partition key (`EXP#<id>`). If an
experiment accumulates more than 10 GB of items (runs, metrics, tags, params,
traces), writes to that partition will fail with an
`ItemCollectionSizeLimitExceededException`.

!!! danger "Hard Limit"
    The 10 GB LSI partition limit cannot be increased. It is a fundamental
    DynamoDB constraint. Plan your data layout accordingly.

### What Consumes Space

| Item Type       | Typical Size | Volume Driver                    |
|-----------------|-------------|----------------------------------|
| Run META        | 1-2 KB      | Number of runs                   |
| Tags            | 100-500 B   | Tags per run                     |
| Params          | 100-500 B   | Params per run                   |
| Latest Metrics  | 200-500 B   | Metric keys per run              |
| Metric History  | 100-200 B   | Steps per metric key             |
| Trace META      | 1-2 KB      | Number of traces                 |
| Cached Spans    | 5-50 KB     | Span count per trace             |

### Estimating Partition Size

Rough formula per experiment:

```
partition_size ≈ runs × (2KB + tags × 300B + params × 300B + metrics × 300B)
                + runs × metrics × steps × 150B  (metric history)
                + traces × (2KB + cached_spans_size)
```

**Example:** 1,000 runs, 10 tags, 20 params, 15 metrics, 100 steps each, no traces:

```
≈ 1000 × (2KB + 3KB + 6KB + 4.5KB) + 1000 × 15 × 100 × 150B
≈ 15.5 MB + 225 MB ≈ 240 MB
```

This experiment is well within the 10 GB limit. The danger zone is experiments
with many runs, many metric keys, and long step histories.

### Mitigation Strategies

1. **Enable metric history TTL** -- Set `metric_history_retention_days` to
   prune old step data automatically.
2. **Limit steps per metric** -- Log metrics at intervals rather than every step.
3. **Split large experiments** -- Create new experiments periodically instead of
   logging thousands of runs to one experiment.
4. **Monitor item counts** -- Track item counts per experiment with the
   CloudWatch `ItemCount` metric or periodic scans.

## Item Count Monitoring

Track the total item count in your table to understand growth trends:

```bash
aws dynamodb describe-table \
    --table-name mlflow \
    --query 'Table.ItemCount'
```

!!! note
    `ItemCount` is updated approximately every 6 hours. For real-time counts,
    use a `Scan` with `Select=COUNT`, but be aware this consumes read capacity.

## X-Ray Trace Monitoring

If X-Ray integration is enabled, monitor these additional dimensions:

| Metric / Signal                  | Where to Check       | Action                        |
|----------------------------------|----------------------|-------------------------------|
| X-Ray `TracesProcessed`          | CloudWatch X-Ray     | Track trace ingestion rate    |
| Cached span item count           | DynamoDB table scan  | Verify cache-spans runs       |
| X-Ray 30-day retention           | Calendar / alarm     | Run cache-spans before expiry |

## Alerting Recommendations

### Critical Alerts

| Alert                                          | Threshold           | Action                                     |
|------------------------------------------------|---------------------|--------------------------------------------|
| `ThrottledRequests > 0` for 5 minutes          | Any throttling      | Increase capacity or investigate hot keys   |
| `SystemErrors > 0` for 5 minutes               | Any system error    | Check AWS Health Dashboard                  |
| `ItemCollectionSizeLimitExceededException`      | Any occurrence      | Split experiment or enable metric TTL       |

### Warning Alerts

| Alert                                          | Threshold                    | Action                              |
|------------------------------------------------|------------------------------|-------------------------------------|
| `ConsumedReadCapacityUnits` > 80% of provisioned | Sustained over 15 minutes | Consider scaling up or auto-scaling |
| `SuccessfulRequestLatency` p99 > 100ms         | Sustained over 10 minutes    | Check for large scans               |
| Item count growth > 20% week-over-week         | Weekly check                 | Review TTL settings and cleanup     |

### Example CloudWatch Alarm (Terraform)

```hcl
resource "aws_cloudwatch_metric_alarm" "throttled_requests" {
  alarm_name          = "mlflow-dynamodb-throttled-requests"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ThrottledRequests"
  namespace           = "AWS/DynamoDB"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    TableName = "mlflow"
  }
}
```
