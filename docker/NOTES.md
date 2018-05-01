### Clear all exited containers

`docker rm $(docker ps -a -f "status=exited" -q)`

... or add the following to `.bash_profile` or `.bashrc`:

- ```source ~/myDockerShortcuts```

where `myDockerShorcuts` is a file with the following contents:
```
function bye_docker() {
    docker rm $(docker ps -a -f "status=exited" -q)
}
```

then `bye_docker` will clear all the stopped containers

### Free Disk Space - remove dangling images

https://bobcares.com/blog/how-to-clear-docker-cache-and-save-disk-space/2/

`docker rmi $(docker images -f "dangling=true" -q)`
