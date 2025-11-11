In order to give your finetuned model a basic Chatgpt like UI folllow the below steps. This file will guide you how to Download and Use Docker and Ollama with Open WebUI.
### Installation

1. Download Ollaman (for Linux)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Now convert the existing Modelfile to an ollama model:
```bash
ollama create finetuned-1b -f Modelfile
```
If a 1B variant exists in the Ollama library, pull it (example name only):

```bash
ollama pull llama3.2:1b
ollama run llama3.2:1b
```
2. Download Docker Engine and Docker Desktop


https://docs.docker.com/engine/install/ubuntu/
https://docs.docker.com/desktop/setup/install/linux/

3. Type in the following command in the same terminal
   
```bash
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```
4. Open  Docker Desktop, click on containers and Click on the port number under Port(s) and it should take you to a link where you will be prompted to sign in, or make an account to sign in with Open Web UI.
5. After Signing in you should be able to use the UI
6. After hitting Select a model in the top left corner you should be able to see all the models you downloaded!



