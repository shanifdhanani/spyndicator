# Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy to AWS.

## Prerequisites
* Mac
* PyCharm (optional but highly recommended)
* This repository

## Setting up your Macbook configuration
The latest Macs use zsh instead of bash as the default terminal interpreter, if you need to update that, run the following command, and then restart your terminal:

```
chsh -s /bin/bash
```

You can find details instructions here: https://www.howtogeek.com/444596/how-to-change-the-default-shell-to-bash-in-macos-catalina/

# Getting the code

TBD

# Installing Dependencies

If this is a first-time setup

1. Make sure xcode is setup

```
xcode-select --install
```

2. Set up and use Homebrew to setup Python version and requirements

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install zlib 
brew install bzip2 
brew install libpq 
brew install libpq-dev 
brew install zeromq 
brew install pyenv
brew upgrade openssl
pyenv install --patch 3.8.3 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch)
pyenv local 3.8.3
pyenv global 3.8.3
eval "$(pyenv init -)"
```

3. Set up a virtual environment

```
sudo pip3 install -U virtualenv
python3 -m virtualenv venv
source venv/bin/activate
```

4. Install the remaining dependencies:

```
pip3 install -r requirements.txt
```

5. Install Docker:

```
Download the stable version of Docker from: https://docs.docker.com/v17.12/docker-for-mac/install/
Install docker and make sure it is running
Go to preferences > advanced > give it 16Gb memory
```

Note that if you use PyCharm, you need to configure it to use the virtual environment that you set up in the above step.

6. If you don't have gcloud installed:

```
$ curl -sSL https://sdk.cloud.google.com > /tmp/gcl && bash /tmp/gcl --install-dir=~/gcloud --disable-prompts
$ alias gcloud="~/gcloud/google-cloud-sdk/bin/gcloud"
```

Then:

```
$ gcloud auth login
$ gcloud auth configure-docker
$ cd ~/Documents/projects/apteo
```