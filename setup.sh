#!/usr/bin/env bash
# setup.sh

# Install system dependencies
apt-get update && apt-get install -y \
    build-essential \
    python3-dev

# Upgrade pip
pip install --upgrade pip
pip install wheel setuptools

# Install packages with pre-built wheels first
pip install --only-binary :all: numpy pandas

# Install remaining packages
pip install -r requirements.txt