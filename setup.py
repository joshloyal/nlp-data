from setuptools import setup


PACKAGES = [
    'nlp_data',
    'nlp_data.sentiment'
]


def setup_package():
    setup(
        name="nlp-data",
        version='0.1.0',
        description="Datasets for NLP Tasks",
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/nlp-data',
        license='MIT',
        install_requires=['numpy', 'pandas', 'scikit-learn', 'h5py', 'marisa-trie'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()
