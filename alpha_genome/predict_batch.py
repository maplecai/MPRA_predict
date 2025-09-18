import os
import pickle
import numpy as np
import pandas as pd
from alphagenome.models import dna_client
from tqdm import tqdm
from utils import pad_seq, HDF5Writer


if __name__ == "__main__":
    # df = pd.read_csv('./Gosai_MPRA_my_processed_data_len200_norm.csv')
    df = pd.read_csv('../data/Gosai_MPRA/Gosai_MPRA_my_processed_data_len200_norm.csv')
    seqs = df['seq'].tolist()

    os.environ["GRPC_PROXY_EXP"] = "http://127.0.0.1:7897"
    os.environ["http_proxy"] = "http://127.0.0.1:7897"
    os.environ["https_proxy"] = "http://127.0.0.1:7897"

    api_key = 'AIzaSyCH8xp8n5siM8N7G7-qgR7Xy4q0ektB03s'
    dna_model = dna_client.create(api_key)

    writer_1 = HDF5Writer(
        file_path="Gosai_AlphaGenome_DNase.h5", 
        dataset_name="data",
        data_shape=(305,),
        chunk_size=1024,
        dtype="float32",
    )
    writer_2 = HDF5Writer(
        file_path="Gosai_AlphaGenome_histone.h5", 
        dataset_name="data",
        data_shape=(1116,),
        chunk_size=1024,
        dtype="float32",
    )

    batch_size = 1024
    start = len(writer_1.dset)
    end = len(seqs)

    for i in tqdm(range(start, end, batch_size)):
        batch_seqs = seqs[i:i+batch_size]
        batch_seqs = [pad_seq(seq, 2048, 'N') for seq in batch_seqs]

        outputs = dna_model.predict_sequences(
            batch_seqs,
            requested_outputs=[dna_client.OutputType.DNASE, dna_client.OutputType.CHIP_HISTONE],
            ontology_terms=None,
            progress_bar=True,
        )

        dnase = np.array([outputs[i].dnase.values for i in range(len(batch_seqs))])
        # shape = (batch_size, 2048, 305)
        dnase = dnase[:, 924:1124, :].mean(axis=1)

        histone = np.array([outputs[i].chip_histone.values for i in range(len(batch_seqs))])
        # shape = (batch_size, 16, 1116) 128bp resolution
        histone = histone[:, 7:9, :].mean(axis=1)

        writer_1.append(dnase)
        writer_2.append(histone)

        print(f"写入数据 {i}-{i+batch_size}, 数据集大小: {writer_1.dset.shape}, {writer_2.dset.shape}")

    writer_1.close()
    writer_2.close()
    