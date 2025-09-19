## Usage

Install kerpy (a package adapted from https://github.com/oxcsml/kerpy)

```(bash)
cd kerpy
python setup.py develop
cd ..
```

Reproduce the results in Fig. 6.

```(bash)
cd algorithm
python discrimination.py
cd ..
```

Reproduce the results of our algorithm in Fig. 7.

```(bash)
cd algorithm
python estimation.py
cd ..
```

Reproduce the results of baselines (CM and GRICA) in Fig. 7.

```(bash)
cd baseline
python cm.py
python grica.py
cd ..
```

## Requirements

pytorch 2.1.2

numpy 1.24.3

scipy 1.10.1

networkx 3.1

matplotlib 3.7.1
