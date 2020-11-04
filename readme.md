# F20 CS858 Repo
Repository for Fall 2020 CS858. Code to support the lexical analysis.


## Quickstart
NOTE: The quickstart assumes the user has docker & docker-compose installed.  
```
git clone git@github.com:JRWu/f20_cs858.git

cd f20_cs858

make
```

### Slow Start (Dependencies)
1. Install stable docker-ce. 
```
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - 
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce
```

(Optional) Perform post-install docker instructions in order to run docker as root.
```
sudo groupadd docker
sudo usermod -aG docker $USER
```

Install docker-compose.
```
sudo curl -L "https://github.com/docker/compose/releases/download/1.27.4/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose
```

## Current Packages Utilized:
[pytextrank](https://github.com/DerwenAI/pytextrank) is used to currently score phrases of text given a corpus (privacy policy).






