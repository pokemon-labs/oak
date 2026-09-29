import oak

# import oak.search

# s = "jynx blizzard lovelykiss psychic rest; chansey icebeam sing softboiled thunderbolt; cloyster blizzard clamp explosion hyperbeam; rhydon bodyslam earthquake rockslide substitute; starmie blizzard recover thunderbolt thunderwave; tauros blizzard bodyslam earthquake hyperbeam"
# t = "alakazam psychic recover seismictoss thunderwave; chansey reflect seismictoss softboiled thunderwave; exeggutor explosion psychic sleeppowder stunspore; lapras blizzard hyperbeam sing thunderbolt; snorlax bodyslam earthquake hyperbeam selfdestruct; tauros blizzard bodyslam earthquake hyperbeam"

# battle, durations, result = oak.parse_battle(f"{t} | {s}")

# pucb = oak.search.UCB(1.0)
# mucb = oak.search.MatrixUCB(pucb, 1.0, 512, 999999999)
# network = oak.search.Network()
# network.read_parameters("/home/user/RL/sunday/2000.battle.net")
# p1_cache = oak.search.SideCache()
# node = oak.search.MatrixUCBNode()
# output = oak.search.run(battle, durations, oak.search.Iterations(32000), mucb, node, oak.search.MonteCarlo())
# print(output.empirical_matrix)
# print(output.visit_matrix)
# for row in output.visit_matrix:
#     print(row)


import numpy as np
import argparse
import random
import math

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int)
parser.add_argument("--cols", type=int)
parser.add_argument("--iterations", type=int)
parser.add_argument("--interval", type=int, default=1)
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--c", type=float, default=1.0)
parser.add_argument("--double", action="store_true")
parser.add_argument("--live", action="store_true")
parser.add_argument("--decay", type=float, default=0.0)
parser.add_argument("--prior-temp", type=float, default=0.0)

parser.add_argument("--sd", type=float, default=0.5)
args = parser.parse_args()


def construct_matrix_and_do_matrix_ucb(rows, cols, iterations, trials):
    total_exploitability = 0
    max_exploitability = 0
    total_value_error = 0

    for trial in range(trials):
        means = np.random.rand(rows, cols)
        sd = np.random.rand(rows, cols) * args.sd
        bandits = {}
        for i in range(rows):
            for j in range(cols):
                bandits[(i, j)] = random.gauss(means[i, j], sd[i, j])

        visits = np.zeros((rows, cols), dtype=np.int64)
        total_values = np.zeros((rows, cols), dtype=np.double)

        def solve(c=1.0, iteration=0):
            log_T = math.log(iterations if not args.live else iteration + 1)
            weight = math.log(2 * rows * cols)

            row_payoffs = np.zeros((rows, cols))
            col_payoffs = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    exploration = c * math.sqrt(
                        2 * (2 * log_T + weight) / (visits[i, j] + 1)
                    )
                    if visits[i, j]:
                        row_payoffs[i, j] = total_values[i, j] / visits[i, j]
                        col_payoffs[i, j] = 1 - row_payoffs[i, j]
                    else:
                        row_payoffs[i, j] = 0.5
                        col_payoffs[i, j] = 0.5
                    row_payoffs[i, j] += exploration
                    col_payoffs[i, j] += exploration

            st1, st2, row_payoff, col_payoff = oak.solve_bimatrix(
                row_payoffs, col_payoffs, 256
            )
            # print(row_payoffs)
            # print(col_payoffs)
            return st1, st2, row_payoff

        def solve_double(c=1.0, iteration=0):
            log_T = math.log(iterations if not args.live else iteration + 1)
            weight = math.log(2 * rows * cols)

            row_payoffs = np.zeros((rows, cols))
            col_payoffs = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    exploration = c * math.sqrt(
                        2 * (2 * log_T + weight) / (visits[i, j] + 1)
                    )
                    if visits[i, j]:
                        row_payoffs[i, j] = total_values[i, j] / visits[i, j]
                        col_payoffs[i, j] = row_payoffs[i, j]
                    else:
                        row_payoffs[i, j] = 0.5
                        col_payoffs[i, j] = 0.5
                    row_payoffs[i, j] += exploration
                    col_payoffs[i, j] -= exploration

            st1, _, row_payoff = oak.solve_matrix(row_payoffs, 256)
            _, st2, row_payoff = oak.solve_matrix(col_payoffs, 256)
            # print(row_payoffs)
            # print(col_payoffs)
            return st1, st2, row_payoff

        row_nash, col_nash, nash_value = oak.solve_matrix(means, 256)

        s1 = np.pow(row_nash, args.prior_temp)
        s1 /= s1.sum()
        s2 = np.pow(col_nash, args.prior_temp)
        s2 /= s2.sum()

        for iteration in range(iterations):
            i = random.choices(list(range(rows)), weights=s1)[0]
            j = random.choices(list(range(cols)), weights=s2)[0]
            value = random.gauss(means[i, j], sd[i, j])
            visits[i, j] += 1
            total_values[i, j] += value
            if (iteration + 1) % args.interval == 0:
                # print(s1, s2)
                if args.double:
                    t1, t2, row_payoff, _ = solve_double(args.c, iteration)
                else:
                    t1, t2, row_payoff = solve(args.c, iteration)
                s1 = (1 - args.decay) * t1 + args.decay * s1
                s2 = (1 - args.decay) * t2 + args.decay * s2
        # print(visits)
        # print(means)
        s1, s2, value_estimate = solve(0, iteration)
        row_options = np.matmul(means, s2)
        col_options = np.matmul(s1, means)
        exploitability = row_options.max() - col_options.min()
        total_exploitability += exploitability
        max_exploitability = max(max_exploitability, exploitability)
        total_value_error += abs(nash_value - value_estimate)


    print(f"avg expl: {total_exploitability / trials}")
    print(f"max expl: {max_exploitability}")
    print(f"avg value error: {total_value_error / trials}")


def main():
    construct_matrix_and_do_matrix_ucb(
        args.rows, args.cols, args.iterations, args.trials
    )


def test():
    row = np.zeros((2, 2))
    col = np.zeros((2, 2))
    row[0, 0] = 0.5
    # col[0, 0] = .5
    row[1, 1] = 1
    col[0, 1] = 1
    col[1, 0] = 1

    print(row)
    print(col)

    a, b, c, d = oak.solve_bimatrix(row, col, 8)
    # a,b,c = oak.solve_matrix(row, 8)
    print(a, b)


if __name__ == "__main__":
    main()
    # test()
