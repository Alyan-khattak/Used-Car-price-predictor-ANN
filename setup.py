# ═══════════════════════════════════════════════════════════════════
# setup.py
# ═══════════════════════════════════════════════════════════════════
# Python project ko installable package banata hai.
# pip install -e . chalane pe yeh file run hoti hai.
# find_packages() → carprice/ folder dhundhta hai aur register karta hai
###==============================================================

from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
    requirements.txt se sab packages read karta hai.
    '-e .' ko ignore karta hai — editable install marker hai
    """
    requirement_lst: List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt not found")
    return requirement_lst

setup(
    name="CarPricePredictor",
    version="0.0.1",
    author="M.Alyan",
    author_email="alyankhattake@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)