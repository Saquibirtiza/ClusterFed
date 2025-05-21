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
To prepare the dataset for federated learning, you need to partition it among clients. The instructions to recreate the results for the UNSW dataset are given below. A separate script is also available for ACI_IoT dataset. We have included the csv files for the UNSW dataset in the `02_Data` folder. The [ACI_IoT](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023) dataset can be downloaded and added in the `02_Data` folder to recreate the results. 

To prepare the dataset for federated learning, it must be partitioned across multiple clients. Instructions for partitioning the data using the UNSW dataset are provided below. A separate script is available for the ACI\_IoT dataset. The CSV files for the UNSW dataset are located in the `02_Data` folder. You can download the [ACI\_IoT dataset](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023) and place it in the same `02_Data` folder to reproduce the corresponding results.


#### Instructions:
1. Navigate to `03_Scripts` folder.

2. Set the input_dir variable in `sampler_UNSW.py` to the path containing your .csv dataset.

3. Choose the distribution strategy:

    - IID / non-IID: Controlled using the `alpha` parameter.

      - Lower alpha → more non-IID

      - Higher alpha → closer to IID

4. Run the script:

  ```bash
  python sampler_UNSW.py
  ```

### Step 2: Train the Model
1. Open `config.py` and set:

    - `alpha` value (same as in Step 1)

    - Number of clients (same as in Step 1)

    - Any other relevant parameters

2. Run the training script using:

  ```bash
  python main.py
  ```