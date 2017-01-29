from setuptools import setup


PACKAGES = [
    'nlpdata'
]


def setup_package():
    setup(
        name="nlp-data",
        version='0.1.0',
        description="Datasets for NLP Tasks",
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/nlpdata',
        license='MIT',
        install_requires=['numpy', 'pandas', 'scikit-learn'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()
