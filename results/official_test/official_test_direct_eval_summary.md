# Official AIIMS-test direct evaluation

| Arm | n | bal acc | mean AUC | combined | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 10 | 0.2804±0.0046 | 0.7177±0.0015 | 0.4991±0.0026 | 0.1758±0.0098 |
| random10596_s0 | 10 | 0.2786±0.0038 | 0.7352±0.0036 | 0.5069±0.0028 | 0.3142±0.0119 |
| le6 | 10 | 0.2688±0.0034 | 0.7533±0.0013 | 0.5110±0.0019 | 0.0988±0.0063 |

## Paired Contrasts

| Contrast | metric | n | mean diff | SE |
| --- | --- | ---: | ---: | ---: |
| le6_minus_baseline | bal_acc | 10 | -0.0117 | 0.0063 |
| le6_minus_baseline | mean_auc | 10 | +0.0356 | 0.0021 |
| le6_minus_baseline | combined | 10 | +0.0120 | 0.0039 |
| le6_minus_baseline | acc | 10 | -0.0770 | 0.0092 |
| le6_minus_random10596_s0 | bal_acc | 10 | -0.0098 | 0.0058 |
| le6_minus_random10596_s0 | mean_auc | 10 | +0.0180 | 0.0030 |
| le6_minus_random10596_s0 | combined | 10 | +0.0041 | 0.0033 |
| le6_minus_random10596_s0 | acc | 10 | -0.2154 | 0.0122 |
| random10596_s0_minus_baseline | bal_acc | 10 | -0.0018 | 0.0060 |
| random10596_s0_minus_baseline | mean_auc | 10 | +0.0176 | 0.0039 |
| random10596_s0_minus_baseline | combined | 10 | +0.0079 | 0.0042 |
| random10596_s0_minus_baseline | acc | 10 | +0.1384 | 0.0155 |
