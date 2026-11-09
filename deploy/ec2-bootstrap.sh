#!/bin/bash
# Runs once as root on first boot (EC2 "user data"), Amazon Linux 2023.
set -euxo pipefail

# 2GB swap: a free-tier t3.micro has only 1GB RAM and PyTorch is hungry
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user

# docker compose plugin
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

touch /home/ec2-user/bootstrap-done
