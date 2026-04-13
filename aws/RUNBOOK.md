# EC2 Training Runbook — Nova Brain

## Instance Details
- **AMI**: AWS Deep Learning AMI (Amazon Linux 2023) — NVIDIA driver 580, CUDA 13.0
- **Type**: g4dn.xlarge (Tesla T4, 16GB VRAM)
- **SSH user**: `ec2-user` (not ubuntu)
- **Python**: `/opt/pytorch/bin/python3`
- **IAM profile**: `NovaTraderS3Access` (provides S3 credentials automatically)
- **Security group**: `sg-016246bb106769540` (port 22 open)
- **Key pair**: `nova-trainer`

## Launch Checklist

### 1. Clone repo
```bash
cd ~ && git clone https://github.com/Airyxison/crypto-brain.git
```

### 2. Download tick data from S3
```bash
mkdir -p ~/crypto-engine
/opt/pytorch/bin/python3 -c "
import boto3
boto3.client('s3').download_file(
    'nova-trader-data-249899228939-us-east-1-an',
    'ticks/ticks.db',
    '/home/ec2-user/crypto-engine/ticks.db'
)
print('ticks.db ready')
"
```

### 3. Launch all 4 assets in parallel
```bash
cd ~/crypto-brain
nohup /opt/pytorch/bin/python3 train.py --symbol BTCUSDT --steps 200000 --alpha 0.05 > ~/btc_train.log 2>&1 &
nohup /opt/pytorch/bin/python3 train.py --symbol ETHUSDT --steps 200000 --alpha 0.10 > ~/eth_train.log 2>&1 &
nohup /opt/pytorch/bin/python3 train.py --symbol SOLUSDT --steps 200000 --alpha 0.10 > ~/sol_train.log 2>&1 &
nohup /opt/pytorch/bin/python3 train.py --symbol ADAUSDT --steps 200000 --alpha 0.05 > ~/ada_train.log 2>&1 &
```

### 4. Verify
```bash
ps aux | grep 'train.py' | grep -v grep | wc -l  # should be 4
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader  # should climb to 90%+
```

## Monitoring
- Logs: `~/btc_train.log`, `~/eth_train.log`, `~/sol_train.log`, `~/ada_train.log`
- Checkpoints: `~/crypto-brain/checkpoints/{symbol}/`
- S3 backup: every checkpoint uploads automatically to `s3://nova-trader-data-249899228939-us-east-1-an/checkpoints/{symbol}/`

## Notes
- GPU utilization drops to 5-15% during validation backtests (every 10k steps) — normal
- Multiple assets hitting checkpoints simultaneously causes longer low-GPU stretches
- **Terminate instance immediately after all 4 complete** to stop billing
- Spot instances: safe to use now — S3 sync means reclamation loses at most one 10k-step interval
