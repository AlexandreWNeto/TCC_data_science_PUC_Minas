FROM python:3.9

WORKDIR /code

COPY . /code

RUN pip install --no-cache-dir -r /code/requirements.txt

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]

# docker build -t imagem .
# docker run -p 8888:8888 -v "$(pwd)/Cadernos:/code/Cadernos" -v "$(pwd)/Dados tratados:/code/Dados tratados" imagem