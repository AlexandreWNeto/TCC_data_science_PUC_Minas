Trabalho de conclusão do curso de Ciência de Dados e Big Data da PUC Minas.

### Visualização dos resultados

Para visualizar os resultados das rotinas já executadas, abra os cadernos _Jupyter_ na pasta **Cadernos**.

### Execução do código (_Docker_)

Para executar as rotinas de treinamento dos modelos preditivos:
- Faça o _download_ da pasta **_Docker_**
- Acesse a pasta _Docker_ na janela de comando ou no _PowerShell_
- Monte a imagem _Docker_ com o seguinte comando:
```
docker build -t imagem .
```
- Após a montagem da imagem _Docker_, execute os modelos no contêiner com os seguintes comandos:
```
docker run -e FILE_NAME=modelos_1-4.py imagem
```
ou
```
docker run -e FILE_NAME=modelos_5-8.py imagem
```
Observação: o _Docker Desktop_ precisa estar instalado no seu computador para a execução dos comandos acima.

Para instruções sobre como instalar o _Docker Desktop_, visite: https://docs.docker.com/get-docker .
