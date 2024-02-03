Trabalho de conclusão do curso de Ciência de Dados e Big Data da PUC Minas.

### Visualização dos resultados

Para visualizar os resultados das rotinas já executadas, abra os cadernos _Jupyter_ na pasta **Cadernos**.

### Execução dos modelos preditivos (_Docker_)

Para executar as rotinas de treinamento dos modelos preditivos em uma ambiente isolado usando o _Docker_:
- Faça o _download_ da pasta **_Docker_**
- Acesse a pasta _Docker_ na janela de comando ou no _PowerShell_
- Monte a imagem _Docker_ com o seguinte comando:
```
docker build -t imagem .
```
- Após a montagem da imagem _Docker_, execute o ambiente isolado do contêiner _docker_ com o seguinte comando:
```
docker run -p 8888:8888 -v "$(pwd)/Cadernos:/code/Cadernos" -v "$(pwd)/Dados tratados:/code/Dados tratados" imagem
```
- Copie a URL mostrada no _prompt_ de comando no seu navegador para acessar o ambiente _Jupyter_. A URL terá o seguinte formato:
```
 To access the server, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/jpserver-1-open.html
    Or copy and paste one of these URLs:
        **http://localhost:8888/tree?token=<sequencia_de_caracteres>**
        **http://127.0.0.1:8888/tree?token=<sequencia_de_caracteres>**
```
- No ambiente _Jupyter_, acesse a pasta Cadernos.
- Clique nos arquivos _models_1-4_ ou _models_5-8_ para acessar as rotinas preditivas.
- Para executar uma seção do código, clique no botão _Run_.

Observações:
- O_Docker Desktop_ precisa estar instalado no seu computador para a execução dos comandos acima.
  - Para instruções sobre como instalar o _Docker Desktop_, visite: https://docs.docker.com/get-docker
- Para instruções sobre como executar um programa no ambiente _Jupyter_, visite; https://docs.jupyter.org/en/latest/start/index.html
- Somente as rotinas de previsão da geração de energia foram incluídas no contêiner _Docker_.
  - As rotinas de preparação e visualização de dados não foram incluídas porque os dados de entrada não foram carregados neste repositório.

### Organização dos arquivos
- **Cadernos**: contém os cadernos _Jupyter_ com as rotinas escritas neste trabalho.
- **Dados brutos**: contém os conjuntos de dados utilizados neste trabalho.
  - **IMPORTANTE**: conjuntos de dados maiores que 25 Mb não foram carregados neste repositório.
- **Dados tratado**s: contém os conjuntos de dados tratados, resultantes da etapa de tratamento e agregação de dados.
- **Docker**: contém os arquivos a serem armazenados em um contêiner _Docker_ caso se queira executar os modelos preditivos em um contêiner isolado. Também contém as instruções para a construção da imagem _docker_ (_Dockerfile_) e as bibliotecas _Python_ necessárias (_requirements.txt_).



