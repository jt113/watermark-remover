#!/usr/bin/env ts-node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { TemporalVidEditStack } from '../lib/temporal-vid-edit-stack';

const app = new cdk.App();
new TemporalVidEditStack(app, 'TemporalVidEditStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});
