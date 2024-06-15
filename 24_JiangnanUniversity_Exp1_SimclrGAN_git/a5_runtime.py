from a4_trainer import Trainer
import argparse

#声明一个parser（解析器）
parser = argparse.ArgumentParser()
parser.add_argument("--gparam", default=900, type=float)
parser.add_argument("--dparam", default=800, type=float)
parser.add_argument("--lr", default=0.0002, type=float)
parser.add_argument("--l1_coef", default=50, type=float)
parser.add_argument("--l2_coef", default=100, type=float)
# parser.add_argument("--vis_screen", default='gan')
parser.add_argument("--save_path", default='./save__snowed')
parser.add_argument('--data_path', default='./train_snowed.hdf5')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_workers', default=0, type=int)
#原先为8，但是跑不动，所以改为0
parser.add_argument('--epochs', default=101, type=int)
args = parser.parse_args()

trainer = Trainer(gparam=args.gparam,
                  dparam=args.dparam,
                  lr=args.lr,
                  l1_coef=args.l1_coef,
                  l2_coef=args.l2_coef,
                  # vis_screen=args.vis_screen,
                  save_path=args.save_path,
                  data_path=args.data_path,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  epochs=args.epochs
                  )

trainer._train_gan()
