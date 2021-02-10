# CML-HG 

- The implementation of CML-HG.

### Requirements

- Python == 3.6.11
- Pytorch == 1.6.0
- Numpy == 1.19.1
- Scipy == 1.5.2
- networkx == 2.5
- torch-geometric == 1.6.3
- geoopt == 0.3.1

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
  
### How to Run

```
cd CML-HG
mkdir saved_model
```
- For running on IMDB:
```
python train.py --embedder CMVHG --dataset imdb --lr 0.001 --l2_coef 0.0005 --reg_coef 0.001 --w_rel 0.1 --dropadj_1 0 --dropadj_2 0 --dropfeat_1 0 --dropfeat_2 0 --sample_size 1000 --gpu 0
```
- For running on ACM:
```
python train.py --embedder CMVHG --dataset acm --lr 0.001 --l2_coef 0.0001 --reg_coef 1.0 --w_rel 0.01 --w_node 0.01 --dropadj_1 0.1 --dropadj_2 0.2 --dropfeat_1 0.1 --dropfeat_2 0.1 --isAttn --gpu 0
```
- For running on Amazon:
```
python train.py --embedder CMVHG --dataset amazon --lr 0.001 --l2_coef 0.0001 --reg_coef 0.01 --w_node 0.001 --dropadj_1 0.4 --dropadj_2 0.4 --dropfeat_1 0.1 --dropfeat_2 0.1 --sample_size 1000 --isAttn --gpu 0
```
- For running on DBLP:
```
python train.py --embedder CMVHG --dataset dblp --lr 0.001 --l2_coef 0.0005 --reg_coef 0.001 --w_rel 0.01 --w_node 0.001 --dropadj_1 0 --dropadj_2 0 --dropfeat_1 0.1 --dropfeat_2 0.1 --isAttn --gpu 0
```