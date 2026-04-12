# AWS Training Setup — Nova Trader

GPU-accelerated training on AWS `g4dn.xlarge` spot instances (NVIDIA T4).
Expected speedup: ~50-100x over 4-core CPU. A 30-minute training run becomes ~20-40 seconds.

---

## One-Time AWS Setup (do this once)

### Step 1 — Create an S3 bucket

S3 is where tick data and trained checkpoints live. Both this machine and the EC2
training instance read/write from it — it's the shared file store between them.

1. Go to **AWS Console → S3 → Create bucket**
2. Bucket name: `nova-trader-data` (must be globally unique — add your initials if taken, e.g. `nova-trader-data-ej`)
3. Region: **us-east-1**
4. Block all public access: **ON** (default, leave it)
5. Everything else: defaults. Click **Create bucket**.

Note the exact bucket name — you'll set it in the environment below.

---

### Step 2 — Create an IAM user for programmatic access

This gives scripts on both machines permission to read/write S3.

1. Go to **AWS Console → IAM → Users → Create user**
2. User name: `nova-trader-s3`
3. Select **Attach policies directly**
4. Click **Create policy** → JSON tab → paste the contents of `aws/iam_policy.json`
   (replace `YOUR_BUCKET_NAME` with your actual bucket name)
5. Name the policy `nova-trader-s3-policy` → Create
6. Attach it to the user → Create user
7. Go to the user → **Security credentials → Create access key**
8. Use case: **Other** → Next → Create
9. **Save the Access Key ID and Secret Access Key** — you only see the secret once

---

### Step 3 — Configure AWS credentials on this machine

Run in your terminal (not inside Claude):

```bash
! aws configure
```

Enter:
- AWS Access Key ID: (from Step 2)
- AWS Secret Access Key: (from Step 2)
- Default region: `us-east-1`
- Default output format: `json`

---

### Step 4 — Set your bucket name

Edit `aws/config.env` and set `S3_BUCKET` to your actual bucket name:

```bash
nano /home/agent/nova-trader/crypto-brain/aws/config.env
```

---

### Step 5 — Upload tick data to S3

```bash
cd /home/agent/nova-trader/crypto-brain
bash aws/sync_to_s3.sh
```

This uploads `ticks.db` to S3. Takes a few minutes depending on file size.
Only needs to be done again if you collect significantly more tick data.

---

### Step 6 — Create a key pair for SSH

1. Go to **AWS Console → EC2 → Key Pairs → Create key pair**
2. Name: `nova-trainer`
3. Type: RSA, format: `.pem`
4. Download the `.pem` file — move it somewhere safe:

```bash
mv ~/Downloads/nova-trainer.pem ~/.ssh/nova-trainer.pem
chmod 400 ~/.ssh/nova-trainer.pem
```

---

## Launching a Training Run

### Step 7 — Launch a spot instance

1. Go to **AWS Console → EC2 → Launch instance**
2. Name: `nova-trainer`
3. AMI: search for **"Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"**
   — choose the most recent one (by AWS, not a third party)
4. Instance type: `g4dn.xlarge`
5. Key pair: `nova-trainer` (from Step 6)
6. Network: default VPC, default subnet — **enable Auto-assign public IP**
7. Security group: create new:
   - Name: `nova-trainer-sg`
   - Inbound rule: SSH (port 22), source: **My IP**
8. Storage: 50GB gp3 (increase from default 8GB)
9. **Advanced details → Purchasing option → Request Spot Instances ✓**
10. **Advanced details → User data** → paste the contents of `aws/userdata.sh`
    (after editing `S3_BUCKET` inside it to match your bucket name)
11. Launch instance

---

### Step 8 — Wait for setup, then start training

The instance takes ~5 minutes to boot and run the setup script. Check progress:

```bash
# Find the public IP in EC2 console, then:
ssh -i ~/.ssh/nova-trainer.pem ubuntu@<INSTANCE_IP>

# Check bootstrap log:
tail -f /var/log/nova-bootstrap.log

# When you see "Bootstrap complete — ready to train", start training:
cd /home/ubuntu/crypto-brain
bash aws/run_training.sh
```

---

## After Training

Checkpoints are synced to S3 automatically during training.
Download them to this machine:

```bash
bash aws/sync_from_s3.sh
```

Then stop the instance to avoid charges:

1. EC2 Console → select instance → Instance state → **Stop**
   (Stop, not Terminate — stopped instances don't charge for compute, only storage)

Or terminate entirely if you're done:
- EC2 Console → Instance state → **Terminate**

---

## Cost Reference

| Scenario | Cost |
|---|---|
| g4dn.xlarge spot, us-east-1 | ~$0.16/hr |
| Full 4-asset training run (~2h on GPU) | ~$0.32 |
| Storage (50GB EBS while stopped) | ~$4/month |
| S3 storage (~1GB data + checkpoints) | ~$0.02/month |

**Tip:** Stop (not terminate) the instance between runs to keep your environment intact.
The EBS volume persists and you won't need to re-run the bootstrap.
