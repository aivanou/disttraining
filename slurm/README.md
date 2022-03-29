## Setting up SLURM cluster

#### Create aws key-pair

```bash
export AWS_KEY_NAME=trainer-key

aws ec2 create-key-pair \
--key-name $AWS_KEY_NAME \
--query KeyMaterial \
--output text > ~/.ssh/$AWS_KEY_NAME.pem

chmod 600 ~/.ssh/$AWS_KEY_NAME.pem

```

#### Create S3 Bucket

```bash
export S3_BUCKET_NAME=pytorch-disttrain-${USER}-${RANDOM}
aws s3 mb s3://${S3_BUCKET_NAME}
```

#### Upload train data to the s3 bucket

aws s3 cp ../charnn/data/input.txt s3://$S3_BUCKET_NAME/charnn/data/input.txt

The data will be accessible via `s3://$S3_BUCKET_NAME/charnn/data/input.txt`.

#### Install pcluster

```bash
python3 -m pip install --upgrade "aws-parallelcluster"

pcluster version
{
  "version": "3.1.2"
}
```

#### Create SLURM cluster

Before creating SLURM cluster you need to modify `cluster-config.yaml` file. Open the file in IDE and fill in all the
missing values.

After that run the following command:

```bash

export CLUSTER_NAME=train-cluster

pcluster create-cluster \
--cluster-name $CLUSTER_NAME \
--cluster-configuration cluster-config.yaml

```

This command should use cloudformation to create the cluster.

#### Accessing the cluster

Run the following cmd to access the cluster HeadNode:

```bash
pcluster ssh --cluster-name $CLUSTER_NAME -i ~/.ssh/$AWS_KEY_NAME.pem
```

#### Running job

Our SLURM cluster uses shared filesystem to make sure all files are synchronized between compute nodes. The shared fs is
mounted at `/shared` destination. We will be using this folder for storage

Lets install venv there:

```bash
sudo apt-get update
sudo apt-get install python3-venv
python3 -m venv /shared/venv/
source /shared/venv/bin/activate
pip install wheel
echo 'source /shared/venv/bin/activate' >> ~/.bashrc
```

Now lets download the source code there:

```bash
mkdir /shared/code
cd /shared/code/
git clone https://github.com/aivanou/disttraining.git 
cd disttraining
# NOTE: Important to do, due to https://github.com/pytorch/pytorch/pull/69904 bug
python3 -m pip install setuptools==59.5.0
pip install -r requirements.txt
```

Copy the data from S3 bucket to the shared fs:

```bash
mkdir /shared/data && cd /shared/data
aws s3 cp s3://#YOUR_BUCKET_NAME/charnn/data/input.txt ./
```

Now we can run our training job via the following cmd:

```bash
cd /shared/code/disttrain
sbatch ./slurm/slurm_sbatch_run.sh
```

Check that the job is in the queue:

```bash
squeue
```
