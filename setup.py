# setup.py
from setuptools import setup, find_packages

setup(
    name="document-translation-agent",
    version="1.0.0",
    description="PDF 문서를 한국어로 번역하는 서비스",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.1",
        "httpx>=0.25.2",
        "requests>=2.31.0",
        "pydantic>=2.5.0",
        "python-dateutil>=2.8.2",
        "python-dotenv>=1.0.0",
        "paddlepaddle>=2.5.2",
        "paddleocr>=2.7.3",
        "opencv-python>=4.8.1.78",
        "pillow>=10.1.0",
        "ollama>=0.1.7",
        "pathlib2>=2.3.7",
        "typing-extensions>=4.8.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "mypy>=1.7.1",
            "flake8>=6.1.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "doc-translator=run_server:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)