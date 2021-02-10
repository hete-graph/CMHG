### Datasets

- We provide four processed datasets:
  - IMDB
  - ACM
  - Amazon
  - DBLP : DBLP-Citation-network V8 dataset from https://www.aminer.cn/citation
  - You can download all the preprocessed datasets used in the paper from [here](https://www.dropbox.com/s/48oe7shjq0ih151/data.tar.gz?dl=0)

- Data format:
  - Meta-paths
    - IMDB: ``MDM``, ``MAM`` 
    - ACM: ``PAP``,``PLP``
    - Amazon: ``IVI``,``IBI``,``ITI``,``IOI``
    - DBLP: ``PAP``, ``PPP``, ``PATAP``
  - ``train_idx``: training index, ``val_idx``: validation index, ``test_idx``: test index, ``feature``: feature matrix, ``label``: labels
