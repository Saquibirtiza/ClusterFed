# ClusterFed: Self-supervised Federated Network Intrusion Detection using Clustering

This repository contains the code for the paper:  **_ClusterFed: Self-supervised Federated Network Intrusion Detection using Clustering_**

---

## 📌 Environment and Installation

Python 3.12.3  was used to run all the scripts. To install the required libraries, run:

```bash
pip install -r requirements.txt
```


## 🚀 Getting Started
### Step 1: Partition the Dataset for Each Client
To prepare the dataset for federated learning, you need to partition it among clients.

#### Instructions:
1. Set the input_dir variable in `sampler_UNSW.py` to the path containing your .csv dataset.

2. Choose the distribution strategy:

    - IID / non-IID: Controlled using the `alpha` parameter.

      - Lower alpha → more non-IID

      - Higher alpha → closer to IID

3. Run the script:

  ```bash
  python sampler_UNSW.py
  ```

### Step 2: Train the Model
1. Open `config.py` and set:

    - `alpha` value (same as in Step 1)

    - Number of clients (same as in Step 1)

    - Any other relevant parameters

2. Run the training script:

  ```bash
  python main.py
  ```