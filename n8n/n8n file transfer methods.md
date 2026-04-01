# Docker File Transfer Guide

## 1) Where files are saved in a Docker container

A Docker container has its own isolated filesystem.

* Built from an image
* Uses layered architecture:

  * Read-only base image
  * Writable container layer

Example structure:

```
/ (root)
 ├── app/
 ├── usr/
 ├── var/
 └── tmp/
```

Note:

* Files inside container are lost when container is deleted (unless using volumes)

---

## 2) Why disk files are not visible to Docker container

Docker containers are isolated environments.

Reasons:

* Security
* Portability
* Consistency

Your local machine and container do not share filesystem by default.

Example:

```
Host: /home/user/file.txt
Container: cannot access it ❌
```

---

## 3) How to transfer files from disk to Docker container

### Method 1: docker cp

```
docker cp <local-path> <container-id>:<container-path>
```

Example:

```
docker cp file.txt my_container:/app/file.txt
```

---

### Method 2: Bind Mounts

```
docker run -v /local/folder:/container/folder image_name
```

The only accecible path on the container is `/home/node/.n8n-files/`, please use this container path to trasfer your files from the disk

Example:

```
docker run -v $(pwd):/app my_image
```

---

### Method 3: Docker Volumes

```
docker volume create my_volume
docker run -v my_volume:/app my_image
```

---

## Summary

| Method     | Persistence | Use Case        |
| ---------- | ----------- | --------------- |
| docker cp  | No          | Quick transfer  |
| Bind Mount | Yes         | Development     |
| Volume     | Yes         | Production data |

---
