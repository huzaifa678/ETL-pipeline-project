import os
from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the libraries inside the requirements.txt file as a list
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name=os.getenv("PROJECT_NAME"),
version=os.getenv("VERSION"),
author=os.getenv("AUTHOR"),
author_email=os.getenv("AUTHOR_EMAIL"),
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)