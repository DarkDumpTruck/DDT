from collections import namedtuple
import os
import torch
from torch import optim, nn
from tqdm import tqdm
import numpy as np

# This is an autotuner for network speed.
torch.backends.cudnn.benchmark = True

NNArgs = namedtuple(
    "NNArgs",
    [
        "input_size",
        "v_size",
        "pi_size",
        "num_channels",
        "depth",
        "kernel_size",
        "lr_milestone",
        "lr",
        "cv",
    ],
    defaults=(
        40,
        0.001,
        1.5,
    ),
)


def conv(in_channels, out_channels, stride=1, kernel_size=3):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding="same",
        bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    return conv(in_channels, out_channels, stride, 1)


def conv3x3(in_channels, out_channels, stride=1):
    return conv(in_channels, out_channels, stride, 3)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, kernel_size=3):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(in_channels, growth_rate * bn_size, 1)
        self.bn2 = nn.BatchNorm2d(growth_rate * bn_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv(growth_rate * bn_size, growth_rate, kernel_size=kernel_size)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = torch.cat([x, out], 1)
        return out


class NNArch(nn.Module):
    def __init__(self, game, args):
        super(NNArch, self).__init__()

        in_channels, in_x, in_y = args.input_size
        in_channels -= 1

        self.layers = []
        for i in range(args.depth):
            self.layers.append(
                DenseBlock(
                    in_channels + args.num_channels * i,
                    args.num_channels,
                    kernel_size=args.kernel_size,
                )
            )
        self.conv_layers = nn.Sequential(*self.layers)

        final_size = in_channels + args.num_channels * args.depth
        self.v_conv = conv1x1(final_size, 32)
        self.pi_conv = conv1x1(final_size, 32)

        self.v_bn = nn.BatchNorm2d(32)
        self.v_relu = nn.ReLU(inplace=True)
        self.v_flatten = nn.Flatten()
        self.v_fc1 = nn.Linear(32 * in_x * in_y + 28, 256)
        self.v_fc1_relu = nn.ReLU(inplace=True)
        self.v_fc2 = nn.Linear(256, args.v_size)
        self.v_softmax = nn.LogSoftmax(1)

        self.pi_bn = nn.BatchNorm2d(32)
        self.pi_relu = nn.ReLU(inplace=True)
        self.pi_flatten = nn.Flatten()
        self.pi_fc1 = nn.Linear(in_x * in_y * 32 + 28, args.pi_size)
        self.pi_softmax = nn.LogSoftmax(1)

    # s = batch_size * num_channels * board_x * board_y
    def forward(self, s):
        s, l = torch.split(s, [s.shape[1] - 1, 1], dim=1)
        l = torch.flatten(l, start_dim=1)[:, :28]

        s = self.conv_layers(s)

        v = self.v_conv(s)
        v = self.v_bn(v)
        v = self.v_relu(v)
        v = self.v_flatten(v)
        v = torch.cat((v, l), 1)
        v = self.v_fc1(v)
        v = self.v_fc1_relu(v)
        v = self.v_fc2(v)
        v = self.v_softmax(v)

        pi = self.pi_conv(s)
        pi = self.pi_bn(pi)
        pi = self.pi_relu(pi)
        pi = self.pi_flatten(pi)
        pi = self.pi_fc1(torch.cat((pi, l), 1))
        pi = self.pi_softmax(pi)

        return v, pi


class NNWrapper:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.nnet = NNArch(game, args)
        self.optimizer = optim.SGD(
            self.nnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
        )

        def lr_lambda(epoch):
            if epoch < 5:
                return 3
            elif epoch > args.lr_milestone:
                # after milestone, lr_lambda starts to decrease from 0.5 to a minimum of 0.05
                return args.lr_milestone / 2 / min(epoch, args.lr_milestone * 10)
            else:
                return 1

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )
        # self.scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, milestones=args.lr_milestones, gamma=0.1)
        self.cv = args.cv
        self.nnet.cuda()

        # self.nnet = torch.compile(self.nnet)

    def losses(self, dataset):
        self.nnet.eval()
        l_v = 0
        l_pi = 0
        for batch in tqdm(dataset, desc="Calculating Sample Loss", leave=False):
            canonical, target_vs, target_pis = batch
            canonical = canonical.contiguous().cuda()
            target_vs = target_vs.contiguous().cuda()
            target_pis = target_pis.contiguous().cuda()

            out_v, out_pi = self.nnet(canonical)
            l_v += self.loss_v(target_vs, out_v).item()
            l_pi += self.loss_pi(target_pis, out_pi).item()
        return l_v / len(dataset), l_pi / len(dataset)

    def sample_loss(self, dataset, size):
        loss = np.zeros(size)
        self.nnet.eval()
        i = 0
        for batch in tqdm(dataset, desc="Calculating Sample Loss", leave=False):
            canonical, target_vs, target_pis = batch
            canonical = canonical.contiguous().cuda()
            target_vs = target_vs.contiguous().cuda()
            target_pis = target_pis.contiguous().cuda()

            out_v, out_pi = self.nnet(canonical)
            l_v = self.sample_loss_v(target_vs, out_v)
            l_pi = self.sample_loss_pi(target_pis, out_pi)
            total_loss = l_pi + l_v
            for sample_loss in total_loss:
                loss[i] = sample_loss
                i += 1
        return loss

    def train(self, batches, steps_to_train, run, epoch, total_train_steps):
        self.nnet.train()

        v_loss = 0
        pi_loss = 0
        current_step = 0
        pbar = tqdm(
            total=steps_to_train, unit="batches", desc="Training NN", leave=False
        )
        past_states = []
        while current_step < steps_to_train:
            for batch in batches:
                if (
                    steps_to_train // 4 > 0
                    and current_step % (steps_to_train // 4) == 0
                    and current_step != 0
                ):
                    # Snapshot model weights
                    past_states.append(dict(self.nnet.named_parameters()))
                if current_step == steps_to_train:
                    break
                canonical, target_vs, target_pis = batch
                canonical = canonical.contiguous().cuda()
                target_vs = target_vs.contiguous().cuda()
                target_pis = target_pis.contiguous().cuda()

                # reset grad
                self.optimizer.zero_grad()

                # forward + backward + optimize
                out_v, out_pi = self.nnet(canonical)
                l_v = self.loss_v(target_vs, out_v)
                l_pi = self.loss_pi(target_pis, out_pi)
                total_loss = l_pi + l_v
                total_loss.backward()
                self.optimizer.step()

                run.track(
                    l_v.item(),
                    name="loss",
                    epoch=epoch,
                    step=total_train_steps + current_step,
                    context={"type": "value"},
                )
                run.track(
                    l_pi.item(),
                    name="loss",
                    epoch=epoch,
                    step=total_train_steps + current_step,
                    context={"type": "policy"},
                )
                run.track(
                    l_v.item() + l_pi.item(),
                    name="loss",
                    epoch=epoch,
                    step=total_train_steps + current_step,
                    context={"type": "total"},
                )

                # record loss and update progress bar.
                pi_loss += l_pi.item()
                v_loss += l_v.item()
                current_step += 1
                pbar.set_postfix(
                    {
                        "v loss": v_loss / current_step,
                        "pi loss": pi_loss / current_step,
                        "total": (v_loss + pi_loss) / current_step,
                    }
                )
                pbar.update()

        # Perform expontential averaging of network weights.
        past_states.append(dict(self.nnet.named_parameters()))
        merged_states = past_states[0]
        for state in past_states[1:]:
            for k in merged_states.keys():
                merged_states[k].data = (
                    merged_states[k].data * 0.75 + state[k].data * 0.25
                )
        nnet_dict = self.nnet.state_dict()
        nnet_dict.update(merged_states)
        self.nnet.load_state_dict(nnet_dict)

        self.scheduler.step()
        print(f"Current learn rate={self.scheduler.get_lr()}")
        pbar.close()
        return v_loss / steps_to_train, pi_loss / steps_to_train

    def predict(self, canonical, debug=False):
        v, pi = self.process(canonical.unsqueeze(0), debug)
        return v[0], pi[0]

    def process(self, batch, debug=False):
        batch = batch.contiguous().cuda()
        self.nnet.eval()
        with torch.no_grad():
            v, pi = self.nnet(batch, debug=debug)
            res = (torch.exp(v), torch.exp(pi))
        return res

    def sample_loss_pi(self, targets, outputs):
        return -1 * torch.sum(targets * outputs, axis=1)

    def sample_loss_v(self, targets, outputs):
        return -self.cv * torch.sum(targets * outputs, axis=1)

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        # return torch.sum((targets - outputs) ** 2) / targets.size()[0]
        return -self.cv * torch.sum(targets * outputs) / targets.size()[0]

    def save_checkpoint(
        self, folder=os.path.join("data", "checkpoint"), filename="checkpoint.pt"
    ):
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)
        torch.save(
            {
                "state_dict": self.nnet.state_dict(),
                "opt_state": self.optimizer.state_dict(),
                "sch_state": self.scheduler.state_dict(),
                "args": self.args,
                "game": self.game,
            },
            filepath,
        )

    @staticmethod
    def load_checkpoint(
        Game, folder=os.path.join("data", "checkpoint"), filename="checkpoint.pt"
    ):
        if folder != "":
            filepath = os.path.join(folder, filename)
        else:
            filepath = filename
        if not os.path.exists(filepath):
            raise Exception(f"No model in path {filepath}")
        checkpoint = torch.load(filepath)
        assert (
            checkpoint["game"] == Game
        ), f'Mismatching game type when loading model: got: {checkpoint["game"].__name__} want: {Game.__name__}'
        net = NNWrapper(checkpoint["game"], checkpoint["args"])
        net.nnet.load_state_dict(checkpoint["state_dict"])
        net.optimizer.load_state_dict(checkpoint["opt_state"])
        net.scheduler.load_state_dict(checkpoint["sch_state"])
        return net
