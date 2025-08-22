"""
Package Setup

1. pip install -e . 로 설치
2. spaCy 언어 모델 (별도 설치 필요):
   python -m spacy download en_core_web_sm
"""

from setuptools import setup, find_packages
import os

# requirements.txt에서 의존성 읽기
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
            return requirements
    return []

setup(
    name="restaurant-review-analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=read_requirements(),

)
