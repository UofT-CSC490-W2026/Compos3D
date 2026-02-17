## Delete

### Empty S3 buckets (Terrform doesn't handle this)
```bash
for env in dev staging prod; do
  for layer in bronze silver gold; do
    bucket="compos3d-${env}-${layer}"
    echo "Emptying bucket: $bucket"
    AWS_PROFILE=myisb_IsbUsersPS-136268833180 aws s3api delete-objects --bucket $bucket --delete "$(AWS_PROFILE=myisb_IsbUsersPS-136268833180 aws s3api list-object-versions --bucket $bucket --query='{Objects: Versions[].{Key:Key,VersionId:VersionId}}' --output json)" > /dev/null 2>&1
    AWS_PROFILE=myisb_IsbUsersPS-136268833180 aws s3api delete-objects --bucket $bucket --delete "$(AWS_PROFILE=myisb_IsbUsersPS-136268833180 aws s3api list-object-versions --bucket $bucket --query='{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' --output json)" > /dev/null 2>&1
  done
done
```

### Force Delete Secrets (Terrform doesn't handle this)
```bash
for env in dev staging prod; do
  for secret in compos3d-${env}-openai-key compos3d-${env}-anthropic-key compos3d-${env}-anyscale-key compos3d-${env}-wandb-key; do
    echo "Force deleting secret: $secret"
    AWS_PROFILE=myisb_IsbUsersPS-136268833180 aws secretsmanager delete-secret --secret-id $secret --force-delete-without-recovery > /dev/null 2>&1
  done
done
```

### Delete IAM Instance Profile Dependencies (Terrform doesn't handle this)

```bash
for env in dev staging prod; do
  for profile in compos3d-${env}-bronze-profile compos3d-${env}-silver-profile compos3d-${env}-gold-profile; do
    echo "Deleting IAM instance profile: $profile"
    AWS_PROFILE=myisb_IsbUsersPS-136268833180 aws iam delete-instance-profile --instance-profile-name $profile > /dev/null 2>&1
  done
done
for env in dev staging prod; do
  # List of all roles created by the modules
  roles=(
    "compos3d-${env}-anyscale-bronze"
    "compos3d-${env}-anyscale-silver"
    "compos3d-${env}-anyscale-gold"
    "compos3d-${env}-anyscale-access"
    "compos3d-${env}-batch-job"
    "compos3d-${env}-batch-service"
    "compos3d-${env}-athena-user"
    "compos3d_${env}-crawler-role"
  )
  
  for role in "${roles[@]}"; do
    echo "Cleaning up role: $role"
    
    # 1. Remove from any instance profiles
    profiles=$(AWS_PROFILE=myisb_IsbUsersPS-136268833180 aws iam list-instance-profiles-for-role --role-name $role --query 'InstanceProfiles[*].InstanceProfileName' --output text 2>/dev/null)
    for profile in $profiles; do
      AWS_PROFILE=myisb_IsbUsersPS-136268833180 aws iam remove-role-from-instance-profile --instance-profile-name $profile --role-name $role > /dev/null 2>&1
    done

    # 2. Detach all policies
    policies=$(AWS_PROFILE=myisb_IsbUsersPS-136268833180 aws iam list-attached-role-policies --role-name $role --query 'AttachedPolicies[*].PolicyArn' --output text 2>/dev/null)
    for policy in $policies; do
      AWS_PROFILE=myisb_IsbUsersPS-136268833180 aws iam detach-role-policy --role-name $role --policy-arn $policy > /dev/null 2>&1
    done
    
    # 3. Delete inline policies
    inline_policies=$(AWS_PROFILE=myisb_IsbUsersPS-136268833180 aws iam list-role-policies --role-name $role --query 'PolicyNames[*]' --output text 2>/dev/null)
    for policy in $inline_policies; do
      AWS_PROFILE=myisb_IsbUsersPS-136268833180 aws iam delete-role-policy --role-name $role --policy-name $policy > /dev/null 2>&1
    done

    # 4. Delete the role
    AWS_PROFILE=myisb_IsbUsersPS-136268833180 aws iam delete-role --role-name $role > /dev/null 2>&1
  done
done
```

### Terraform Destroy (All Envs)

```bash
# Destroy Dev
AWS_PROFILE=myisb_IsbUsersPS-136268833180 terraform workspace select dev
AWS_PROFILE=myisb_IsbUsersPS-136268833180 terraform destroy -auto-approve -var-file=terraform.tfvars
# Destroy Staging
AWS_PROFILE=myisb_IsbUsersPS-136268833180 terraform workspace select staging
AWS_PROFILE=myisb_IsbUsersPS-136268833180 terraform destroy -auto-approve -var-file=environments/staging.tfvars
# Destroy Prod
AWS_PROFILE=myisb_IsbUsersPS-136268833180 terraform workspace select prod
AWS_PROFILE=myisb_IsbUsersPS-136268833180 terraform destroy -auto-approve -var-file=environments/prod.tfvars
```

## Deploy

```bash
AWS_PROFILE=myisb_IsbUsersPS-136268833180 terraform init

# Deploy Dev
terraform workspace select dev || terraform workspace new dev
AWS_PROFILE=myisb_IsbUsersPS-136268833180 terraform apply -auto-approve -var-file=terraform.tfvars

# Deploy Staging
terraform workspace select staging || terraform workspace new staging
AWS_PROFILE=myisb_IsbUsersPS-136268833180 terraform apply -auto-approve -var-file=environments/staging.tfvars

# Deploy Prod
terraform workspace select prod || terraform workspace new prod
AWS_PROFILE=myisb_IsbUsersPS-136268833180 terraform apply -auto-approve -var-file=environments/prod.tfvars
```

## Run all data pipelines

### Submit to Anyscale
```bash
source .venv/bin/activate
python submit_job.py
```
