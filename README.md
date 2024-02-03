Trabalho de conclusão do curso de Ciência de Dados e Big Data da PUC Minas.

### Visualização dos resultados

Para visualizar os resultados das rotinas já executadas, abra os cadernos _Jupyter_ na pasta **Cadernos**.

### Execução dos modelos preditivos (_Docker_)

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

Nota: somente as rotinas de previsão da geração de energia foram incluídas no contêiner _Docker_.
As rotinas de preparação e visualização de dados não foram incluídas porque os dados de entrada não foram carregados neste repositório.


### Organização dos arquivos
- Cadernos: contém os cadernos _Jupyter_ com as rotinas escritas neste trabalho.
- Dados brutos: contém os conjuntos de dados utilizados neste trabalho. IMPORTANTE: conjuntos de dados maiores que 25 Mb não foram carregados neste repositório.
- Dados tratados: contém os conjuntos de dados tratados, resultantes da etapa de tratamento e agregação de dados.
- Docker: contém os arquivos a serem armazenados em um contêiner _Docker_ caso se queira executar os modelos preditivos em um contêiner isolado. Também contém as instruções para a construção da imagem _docker_ (_Dockerfile_) e as bibliotecas _Python_ necessárias (_requirements.txt_).



