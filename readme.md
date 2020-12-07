# F20 CS858 Repo
Repository for Fall 2020 CS858. Code to support the lexical analysis.


## Quickstart
NOTE: The quickstart assumes the user has docker & docker-compose installed.  
If they are not installed, check the Slow Start (Dependencies) section.
```
git clone git@github.com:JRWu/f20_cs858.git

cd f20_cs858

make
```

## Lexical Analysis Workflow
This analysis is based on Google's Word2Vec model. It is rather large and unzips to approximately 4 Gigabytes.   

```
# Shell into the Docker (Virtual) environment
# Your working directory will be /f20_cs858
make shell

# Acquire the pre-trained GoogleNews-vectors-negative300.bin.gz word2vec model
bash scripts/acquire_w2v_model.sh

# Launch the Flask interface ... Will take a second to load due to the Word2Vec model being so large.

# Will be located at localhost:5000
python -m flask run --host=0.0.0.0


# After data has been collected in GenerateTables.py, generate the figures for rq1 and rq2
python /f20_cs858/src/GenerateTables.py
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






