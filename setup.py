from setuptools import setup, find_packages

setup(
    name="rag-assistant",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.8,<3.13",
    install_requires=[
        "langchain==0.0.350",
        "chromadb==0.4.18",
        "sentence-transformers==2.2.2",
        "gradio==4.8.0",
        "torch==2.2.1",
        "transformers==4.35.2",
        "python-dotenv==1.0.0",
        "huggingface-hub==0.19.4"
    ]
)