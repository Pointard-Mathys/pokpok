FROM python:3.10-slim 


WORKDIR /app 

COPY api.py . 
COPY requirements.txt .

RUN pip3 install --upgrade pip 
RUN pip3 install -r requirements.txt

CMD streamlit run api.py --server.port 8501