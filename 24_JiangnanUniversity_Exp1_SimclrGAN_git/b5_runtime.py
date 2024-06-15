from b4_ImageEmbeddingTraining import Trainer
import argparse

#datasetFile,batch_size,embedding_size,initial_Lr,checkpoints_path,epochs
#声明一个parser（解析器）
parser = argparse.ArgumentParser()
parser.add_argument("--datasetFile", default='./train_snowed.hdf5')
parser.add_argument("--batch_size", default=6, type=int)
parser.add_argument("--embedding_size", default=1024, type=int)
parser.add_argument("--initial_Lr", default=1e-3)
parser.add_argument('--checkpoints_path', default='./checkpoints/save_embedding_snowed')
parser.add_argument('--epochs', default=101, type=int)
args = parser.parse_args()

trainer = Trainer(datasetFile=args.datasetFile,
                  batch_size=args.batch_size,
                  embedding_size=args.embedding_size,
                  initial_Lr=args.initial_Lr,
                  checkpoints_path=args.checkpoints_path,
                  epochs=args.epochs
                  )

trainer._train_simclr()