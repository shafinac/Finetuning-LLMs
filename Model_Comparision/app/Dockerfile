
FROM python:3.11-slim


WORKDIR /app


RUN pip install --no-cache-dir streamlit requests numpy 


# Optional: if you want BLEU/ROUGE without heavy deps, we bundle a tiny metrics module
COPY eval.py /app/eval.py
COPY app.py /app/app.py


EXPOSE 8501


ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]