import torch


def generate_scipts_params():
    lrs = [0.00184, 0.000747, 2.9e-05, 2.06e-05, 1.47e-05, 2.08e-05, 1.73e-05, 0.000231, 0.00103, 0.000262]
    taus = [0.584, 0.8, 0.044, 0.622, 0.3, 0.054, 0.24, 2.32, 1.22, 0.036]
    wds = [8e-05, 0.000173, 0.000309, 8.35e-05, 1.55e-05, 0.000279, 3.97e-05, 1.88e-05, 4.06e-05, 9.17e-05]


    for lr, tau, wd in zip(lrs, taus, wds):
        exp_script = "python3 train.py --data var_5_50_nosm_20000 --archer lstm_1_60_60_0.1 --sampler nonprojk_soft_gumbel_rescale --dec mlp_1_60_60 --comp kipf_mlp_5_60_60_0.1  --batchsize 100  --optimizer_base adamw --optimizer_enc adamw --initial_lr_enc {} --initial_lr_base {} --tau {} --wd_base {} --wd_enc {} --num_epochs 200 --checkpoint_dir_load sst_lr_{}_tau_{}_wd_{}_50_epoch".format(lr, lr, tau, wd, wd, lr, tau, wd)
        print(exp_script)
        print()


def generate_scripts_seed():
    seeds = list(range(1, 11))

    lr = 0.00103
    tau = 1.22
    wd = 4.06e-05
    for seed in seeds:
        exp_script = "python3 train.py --data var_5_50_nosm_20000 --archer lstm_1_60_60_0.1 --sampler nonprojk_soft_gumbel_rescale --dec mlp_1_60_60 --comp kipf_mlp_5_60_60_0.1  --batchsize 100  --optimizer_base adamw --optimizer_enc adamw --initial_lr_enc {} --initial_lr_base {} --tau {} --wd_base {} --wd_enc {} --num_epochs 200 --seed {}".format(lr, lr, tau, wd, wd, seed)
        print(exp_script)
        print()


def generate_scripts_seed_relax():
    seeds = list(range(1, 11))

    lr = 0.00103
    tau = 1.22
    wd = 4.06e-05
    for seed in seeds:
        exp_script = "python3 train.py --data var_5_50_nosm_20000 --archer lstm_1_60_60_0.1 --sampler nonprojk_soft_gumbel_rescale --dec mlp_1_60_60 --comp kipf_mlp_5_60_60_0.1  --batchsize 100  --optimizer_base adamw --optimizer_enc adamw --initial_lr_enc {} --initial_lr_base {} --tau {} --wd_base {} --wd_enc {} --num_epochs 200 --seed {}".format(lr, lr, tau, wd, wd, seed)
        print(exp_script)
        print()


def generate_params(n):
    ns_kv = {0 : 2, 1 : 4, 2 : 5, 3 : 8}

    lr_sampler = torch.distributions.Uniform(torch.tensor([-5.]), torch.tensor([-3.]))
    wd_sampler = torch.distributions.Uniform(torch.tensor([-5.]), torch.tensor([-3.]))

    lr = 10 ** lr_sampler.sample((n,))
    wd = 10 ** wd_sampler.sample((n,))

    lr = [lr[i].item() for i in range(n)]
    wd = [wd[i].item() for i in range(n)]

    num_samples = [ns_kv[torch.randint(4, (1,)).item()] for i in range(n)]
    return lr, wd, num_samples


def generate_params_relax(n):
    lr_base_sampler = torch.distributions.Uniform(torch.tensor([-5.]), torch.tensor([-3.]))
    lr_critic_sampler = torch.distributions.Uniform(torch.tensor([-5.]), torch.tensor([-3.]))
    wd_base_sampler = torch.distributions.Uniform(torch.tensor([-5.]), torch.tensor([-3.]))
    wd_critic_sampler = torch.distributions.Uniform(torch.tensor([-5.]), torch.tensor([-3.]))


    lr_base = 10 ** lr_base_sampler.sample((n,))
    lr_critic = 10 ** lr_critic_sampler.sample((n,))
    wd_base = 10 ** wd_base_sampler.sample((n,))
    wd_critic = 10 ** wd_critic_sampler.sample((n,))

    lr_base = [lr_base[i].item() for i in range(n)]
    lr_critic = [lr_critic[i].item() for i in range(n)]
    wd_base = [wd_base[i].item() for i in range(n)]
    wd_critic = [wd_critic[i].item() for i in range(n)]

    return lr_base, lr_critic, wd_base, wd_critic


def generate_scripts_reinforce(n):
    torch.manual_seed(37)
    lrs, wds, num_samples = generate_params(n)

    for lr, wd, ns in zip(lrs, wds, num_samples):
        bs = 100 // ns
        exp_script = "python3 train.py --data var_5_50_nosm_20000 --archer lstm_1_60_60_0.1 --sampler edmonds --dec mlp_1_60_60 --comp kipf_mlp_5_60_60_0.1 --estimator reinforce --reinforce --mean_plus --plus_samples {} --batchsize {} --optimizer_base adamw --optimizer_enc adamw --initial_lr_enc {:.3} --initial_lr_base {:.3} --wd_base {:.3} --wd_enc {:.3} --num_epochs 200 --max_iter 200000 --eval_every 1000 --checkpoint_dir_load reinforce_lr_{:.3}_samples_{}_wd_{:.3}_seed_42_50_epoch".format(ns - 1, bs, lr, lr, wd, wd, lr, ns, wd)
        print(exp_script)
        print()


def generate_scripts_relax(n):
    torch.manual_seed(81)
    lrs_base, lrs_critic, wds_base, wds_critic = generate_params_relax(n)

    for lr_base, lr_critic, wd_base, wd_critic in zip(lrs_base, lrs_critic, wds_base, wds_critic):
        exp_script = "python3 train.py --data var_5_50_nosm_20000 --archer lstm_1_60_60_0.1 --critic lstm_1_60_60_0.1 --sampler edmonds --dec mlp_1_60_60 --comp kipf_mlp_5_60_60_0.1 --estimator relax --reinforce --batchsize 100 --optimizer_base adamw --optimizer_enc adamw --initial_lr_base {:.3} --initial_lr_enc {:.3} --initial_lr_critic {:.3} --wd_base {:.3} --wd_enc {:.3} --wd_critic {:.3} --num_epochs 200 --max_iter 200000 --eval_every 1000 --checkpoint_dir_load relax_lrb_{:.3}_lrc_{:.3}_wdb_{:.3}_wdc_{:.3}_seed_42_50_epoch".format(lr_base, lr_base, lr_critic, wd_base, wd_base, wd_critic, lr_base, lr_critic, wd_base, wd_critic)
        print(exp_script)
        print()


def generate_scripts_reinforce_e(n):
    torch.manual_seed(38)
    lrs, wds, num_samples = generate_params(n)

    for lr, wd, ns in zip(lrs, wds, num_samples):
        bs = 100 // ns
        exp_script = "python3 train.py --data var_5_50_nosm_20000 --archer lstm_1_60_60_0.1 --sampler edmonds --dec mlp_1_60_60 --comp kipf_mlp_5_60_60_0.1 --estimator reinforce_exp --reinforce --mean_plus --plus_samples {} --batchsize {} --optimizer_base adamw --optimizer_enc adamw --initial_lr_enc {:.3} --initial_lr_base {:.3} --wd_base {:.3} --wd_enc {:.3} --num_epochs 200 --max_iter 200000 --eval_every 1000 --checkpoint_dir_load reinforce_exp_lr_{:.3}_samples_{}_wd_{:.3}_seed_42_50_epoch".format(ns - 1, bs, lr, lr, wd, wd, lr, ns, wd)
        print(exp_script)
        print()



def generate_seed_reinforce():
    seeds = list(range(1, 11))

    lr = 0.000186
    ns = 5
    wd = 4.67e-05
    bs = 100 // ns

    for seed in seeds:
        exp_script = "python3 train.py --data var_5_50_nosm_20000 --archer lstm_1_60_60_0.1 --sampler edmonds --dec mlp_1_60_60 --comp kipf_mlp_5_60_60_0.1 --estimator reinforce --reinforce --mean_plus --plus_samples {} --batchsize {} --optimizer_base adamw --optimizer_enc adamw --initial_lr_enc {:.3} --initial_lr_base {:.3} --wd_base {:.3} --wd_enc {:.3} --num_epochs 200 --max_iter 200000 --eval_every 1000 --seed {}".format(ns - 1, bs, lr, lr, wd, wd, seed)
        print(exp_script)
        print()


if __name__ == '__main__':
    generate_scripts_reinforce_e(20)
