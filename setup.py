from setuptools import find_packages,setup
from typing import List

hypen_e = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''this function return the list outpout'''
    with open(file_path) as file:
        requirement = file.readlines()  #read is use to read entire file and readline is use to read line by line
        requirement = [req.replace('\n','') for req in requirement] #req.strip() is also remove \n,\r
        
        if hypen_e in requirement:
            requirement.remove(hypen_e)
        return requirement



setup(
    name = 'dieases',
    version = '0.0.1',
    author = 'vardhman',
    author_email = 'vardhmanajmera76@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)