# Script to reformat a team dump I used but gitignore'd


def fix_mon(mon):
    x = mon.split("|")
    x.insert(0, "")
    x.insert(2, "")
    x.insert(3, "-")
    for _ in range(7):
        x.append("")
    x = "|".join(x)
    return x


def fix_team(team):
    mons = team.split("]")
    mons = [fix_mon(mon) for mon in mons]
    mons = "]".join(mons)
    return mons


in_file = open("gen1ou.tsv", "r")
out_file = open("rby.tsv", "w")

lines = in_file.read()
max_lines = 100
for line in lines.strip().split("\n")[:max_lines]:
    r, team = line.split("\t")
    # out_file.write(r + '\t' + fix_team(team))
    out_file.write(fix_team(team) + "\n")
