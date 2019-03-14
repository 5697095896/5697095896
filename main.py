import argparse
from solver import Solver
from solver_n_msg_chain_rnn import SolverNMsgChainRNN
from solver_n_msg_multiple_decoder import SolverNMsgMultipleDecoders
from solver_n_msg_chain import SolverNMsgChain
from solver_n_msg_conditional import SolverNMsgCond
from solver_n_msg_deepsteg import SolverNMsgMultipleDecodersDeepSteg

def main(config):
    if config.model_type == 'n_msg':
        solver = SolverNMsgMultipleDecoders(config)
    elif config.model_type == 'n_msg_deepsteg':
        solver = SolverNMsgMultipleDecodersDeepSteg(config)
    elif config.model_type == 'n_msg_cond':
        solver = SolverNMsgCond(config)
    elif config.model_type == 'n_msg_chain':
        solver = SolverNMsgChain(config)
    elif config.model_type == '1_msg':
        solver = Solver2Msg(config)
    else:
        print("dataset type not supported!")
        return -1
    if config.mode == 'train':
        solver.train()
        solver.test()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'sample':
        solver.sample_examples()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train steganography')
    parser.add_argument('--enc_c_lr', type=float, default=0.001, help='')
    parser.add_argument('--enc_m_lr', type=float, default=0.001, help='')
    parser.add_argument('--dec_c_lr', type=float, default=0.001, help='')
    parser.add_argument('--dec_m_lr', type=float, default=0.001, help='')
    parser.add_argument('--lambda_carrier_loss', type=float, default=1.0, help='')
    parser.add_argument('--lambda_msg_loss', type=float, default=1.0, help='')

    parser.add_argument('--num_iters', type=int, default=100000, help='')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'abs', 'mix', 'sched'], help='')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'sample'], help='')
    parser.add_argument('--train_path', required=True, type=str, help='')
    parser.add_argument('--val_path', required=True, type=str, help='')
    parser.add_argument('--test_path', required=True, type=str, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--n_pairs', type=int, default=100000, help='')
    parser.add_argument('--n_messages', type=int, default=1, help='')
    parser.add_argument('--dataset', type=str, default='gcommand', help='select dataset', choices=['gcommand', 'timit', 'yoho'])
    parser.add_argument('--model_type', type=str, default='n_msg', help='select model type', choices=['2_msg', 'n_msg', 'n_msg_chain', 'n_msg_cond', 'n_msg_deepsteg'])
    parser.add_argument('--carrier_detach', default=-1, type=int, help='')
    parser.add_argument('--add_stft_noise', default=-1, type=int, help='')
    parser.add_argument('--adv', action='store_true',default=False, help='')
    parser.add_argument('--flip_msg', action='store_true',default=False, help='')
    parser.add_argument('--block_type', type=str, default='normal', choices=['normal', 'bn', 'in', 'relu'], help='')

    parser.add_argument('--num_workers', type=int, default=0, help='')
    parser.add_argument('--wandb', action='store_true', help='')
    parser.add_argument('--tensorboard', action='store_true', help='')
    parser.add_argument('--load_ckpt', type=str, default=None, help='')
    parser.add_argument('--run_dir', type=str, default='.', help='')
    parser.add_argument('--save_model_every', type=int, default=None, help='')
    parser.add_argument('--sample_every', type=int, default=None, help='')
    parser.add_argument('--plot_weights_every', type=int, default=500, help='')
    args = parser.parse_args()

    main(args)
