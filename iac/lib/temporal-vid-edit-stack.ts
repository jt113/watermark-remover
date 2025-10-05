import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

export class TemporalVidEditStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const keyName = this.node.tryGetContext('keyName') ?? process.env.EC2_KEY_NAME;
    if (!keyName) {
      throw new Error('Provide an EC2 key pair name via context (cdk deploy -c keyName=YOUR_KEY) or EC2_KEY_NAME env.');
    }

    const vpcId = this.node.tryGetContext('vpcId');
    const vpc = vpcId
      ? ec2.Vpc.fromLookup(this, 'Vpc', { vpcId })
      : ec2.Vpc.fromLookup(this, 'DefaultVpc', { isDefault: true });

    const stackRegion = cdk.Stack.of(this).region;
    const amiId: string | undefined = this.node.tryGetContext('amiId');
    const amiNamePattern: string = this.node.tryGetContext('amiName') ?? 'Deep Learning Base OSS*Ubuntu 22.04*';
    const amiOwner: string = this.node.tryGetContext('amiOwner') ?? '898082745236'; // Amazon Deep Learning AMIs

    const resolvedRegion = process.env.CDK_DEPLOY_REGION ?? process.env.CDK_DEFAULT_REGION ?? stackRegion;

    const machineImage = amiId && resolvedRegion
      ? ec2.MachineImage.genericLinux({ [resolvedRegion]: amiId })
      : ec2.MachineImage.lookup({ name: amiNamePattern, owners: [amiOwner] });

    const instanceType = this.node.tryGetContext('instanceType') ?? 'g4dn.xlarge';

    const securityGroup = new ec2.SecurityGroup(this, 'TemporalVidSg', {
      vpc,
      allowAllOutbound: true,
      description: 'Access for temporal video editing GPU instance',
    });
    securityGroup.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(22), 'SSH access');

    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      'sudo apt-get update -y',
      'sudo apt-get install -y git build-essential ffmpeg',
      '# NVIDIA driver & CUDA already present on the DLAMI. Install Miniforge for convenience:',
      'curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o ~/Miniforge3.sh',
      'bash ~/Miniforge3.sh -b -p $HOME/miniforge3',
      'echo "source \$HOME/miniforge3/bin/activate" >> ~/.bashrc',
    );

    const instance = new ec2.Instance(this, 'TemporalVidInstance', {
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      instanceType: new ec2.InstanceType(instanceType),
      machineImage,
      keyName,
      securityGroup,
      userData,
      blockDevices: [
        {
          deviceName: '/dev/sda1',
          volume: ec2.BlockDeviceVolume.ebs(200, {
            encrypted: true,
            volumeType: ec2.EbsDeviceVolumeType.GP3,
          }),
        },
      ],
    });

    new cdk.CfnOutput(this, 'InstanceId', { value: instance.instanceId });
    new cdk.CfnOutput(this, 'PublicDns', { value: instance.instancePublicDnsName });
    new cdk.CfnOutput(this, 'Region', { value: region });
  }
}
