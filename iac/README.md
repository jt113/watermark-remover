# AWS CDK Infrastructure

This folder contains an AWS CDK (TypeScript) stack that provisions a GPU-capable EC2 instance (default `g4dn.xlarge`) preconfigured for the temporal video overlay removal project.

## Prerequisites

- Node.js 18+
- AWS CLI configured with credentials and default region
- AWS CDK v2 installed globally (`npm install -g aws-cdk`) or use `npx cdk`
- An existing EC2 key pair in the target region (for SSH access)
- Appropriate service quotas for the selected GPU (request 1× NVIDIA T4 for `g4dn.xlarge`)

## Project Structure

```
iac/
├── bin/temporal-vid-edit.ts       # CDK app entrypoint
├── lib/temporal-vid-edit-stack.ts # EC2 instance definition
├── package.json                   # Dependencies
├── tsconfig.json
├── cdk.json
├── README.md
├── deploy.sh                      # Helper to deploy the stack
└── destroy.sh                     # Helper to tear it down
```

## Configuration Options

| Context key  | Description | Default |
|--------------|-------------|---------|
| `keyName`    | EC2 key pair name used for SSH access | **required** |
| `instanceType` | EC2 instance type | `g4dn.xlarge` |
| `amiId`      | AMI ID for the instance. Defaults include Deep Learning Base Ubuntu 22.04 images for `us-east-1`, `us-west-2`, `eu-west-1`. Provide your own for other regions. | varies |
| `vpcId`      | Optional VPC ID. If omitted the default VPC is used. | default VPC |

Example: `npx cdk deploy -c keyName=my-key -c instanceType=g6.xlarge`.

## Quick Start

```bash
cd iac
npm install

# Deploy (requires KEY_NAME env or -c keyName=...)
INSTANCE_KEY_NAME=my-key ./deploy.sh

# Destroy when finished
./destroy.sh
```

The stack outputs the instance ID, public DNS, and region once deployment succeeds.

## Post-Deployment Checklist

1. SSH into the instance using the key pair:
   ```bash
   ssh -i /path/to/my-key.pem ubuntu@<public-dns>
   ```
2. Clone the temporal-vid-edit repository (if not already baked into an AMI).
3. Follow the README GPU setup instructions (Conda environment, E2FGVI download, etc.).
4. Shut down resources when finished using `./destroy.sh` or `aws ec2 terminate-instances`.

## Notes

- The default AMI IDs map to the AWS Deep Learning Base GPU Ubuntu 22.04 images as of early 2025. Newer images can be supplied via `-c amiId=...`.
- Security group opens SSH (port 22) to the world; tighten to your IP range if desired by editing the stack.
- The root EBS volume is 200 GB gp3 and encrypted by default.
- Instance includes a user-data script that installs Git, build essentials, FFmpeg, and Miniforge for convenience.
