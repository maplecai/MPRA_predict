import os
import pickle
import numpy as np
import pandas as pd
from alphagenome.models import dna_client
from tqdm import tqdm
from utils import *


if __name__ == "__main__":
    # df = pd.read_csv('./Gosai_MPRA_my_processed_data_len200_norm.csv')
    # df = pd.read_csv('../data/Gosai_MPRA/Gosai_MPRA_my_processed_data_len200_norm.csv')
    df = pd.read_csv('./Zhang_MPRA_final.csv')

    seqs = df['seq'].tolist()

    os.environ["GRPC_PROXY_EXP"] = "http://127.0.0.1:7897"
    os.environ["http_proxy"] = "http://127.0.0.1:7897"
    os.environ["https_proxy"] = "http://127.0.0.1:7897"

    api_key = 'AIzaSyCH8xp8n5siM8N7G7-qgR7Xy4q0ektB03s'
    dna_model = dna_client.create(api_key)

    writer = HDF5MultiWriter(
        file_path="Zhang_AlphaGenome_pred.h5", 
        chunk_size=64,
        dtype="float32",
    )
    writer.create_dataset('RNA-seq', data_shape=(2048, 667))
    writer.create_dataset('CAGE', data_shape=(2048, 546))
    writer.create_dataset('DNase', data_shape=(2048, 305))
    writer.create_dataset('ATAC', data_shape=(2048, 167))
    writer.create_dataset('TF', data_shape=(16, 1617))
    writer.create_dataset('Histone', data_shape=(16, 1116))

    batch_size = 64
    start = len(writer.datasets['RNA-seq']['dset'])
    end = len(seqs)

    for i in tqdm(range(start, end, batch_size)):
        batch_seqs = seqs[i:i+batch_size]
        batch_len = len(batch_seqs)
        batch_seqs = [pad_seq(seq, 2048, 'N') for seq in batch_seqs]

        outputs = dna_model.predict_sequences(
            batch_seqs,
            requested_outputs=dna_client.OutputType,
            ontology_terms=None,
            progress_bar=True,
        )

        pred = np.array([outputs[i].rna_seq.values for i in range(batch_len)])
        writer.append('RNA-seq', pred)
        pred = np.array([outputs[i].dnase.values for i in range(batch_len)])
        writer.append('DNase', pred)
        pred = np.array([outputs[i].atac.values for i in range(batch_len)])
        writer.append('ATAC', pred)
        pred = np.array([outputs[i].chip_tf.values for i in range(batch_len)])
        writer.append('TF', pred)
        pred = np.array([outputs[i].chip_histone.values for i in range(batch_len)])
        writer.append('Histone', pred)

        print(f"写入数据 {i}-{i+batch_size}, 数据集大小: {writer.datasets['RNA-seq']['dset'].shape}")

    writer.close()
    