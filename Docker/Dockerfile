FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./Cadernos /code/Cadernos
COPY ["./Dados tratados/geracao e tempo - umburanas-ba.csv", "/code/Dados tratados/geracao e tempo - umburanas-ba.csv"]

ARG FILE_NAME=modelos_1-4.py

CMD ["sh", "-c", "python Cadernos/$FILE_NAME"]

# docker build -t imagem .
# docker run -e FILE_NAME=modelos_1-4.py imagem
# docker run -e FILE_NAME=modelos_5-8.py imagem