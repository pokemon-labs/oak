import sys

# Util for converting rby-ou.tsv (not public) into a header (teams.h)

# fmt:off
libpkmn_species = ["None",       "Bulbasaur",  "Ivysaur",    "Venusaur",   "Charmander",
    "Charmeleon", "Charizard",  "Squirtle",   "Wartortle",  "Blastoise",
    "Caterpie",   "Metapod",    "Butterfree", "Weedle",     "Kakuna",
    "Beedrill",   "Pidgey",     "Pidgeotto",  "Pidgeot",    "Rattata",
    "Raticate",   "Spearow",    "Fearow",     "Ekans",      "Arbok",
    "Pikachu",    "Raichu",     "Sandshrew",  "Sandslash",  "NidoranF",
    "Nidorina",   "Nidoqueen",  "NidoranM",   "Nidorino",   "Nidoking",
    "Clefairy",   "Clefable",   "Vulpix",     "Ninetales",  "Jigglypuff",
    "Wigglytuff", "Zubat",      "Golbat",     "Oddish",     "Gloom",
    "Vileplume",  "Paras",      "Parasect",   "Venonat",    "Venomoth",
    "Diglett",    "Dugtrio",    "Meowth",     "Persian",    "Psyduck",
    "Golduck",    "Mankey",     "Primeape",   "Growlithe",  "Arcanine",
    "Poliwag",    "Poliwhirl",  "Poliwrath",  "Abra",       "Kadabra",
    "Alakazam",   "Machop",     "Machoke",    "Machamp",    "Bellsprout",
    "Weepinbell", "Victreebel", "Tentacool",  "Tentacruel", "Geodude",
    "Graveler",   "Golem",      "Ponyta",     "Rapidash",   "Slowpoke",
    "Slowbro",    "Magnemite",  "Magneton",   "Farfetchd",  "Doduo",
    "Dodrio",     "Seel",       "Dewgong",    "Grimer",     "Muk",
    "Shellder",   "Cloyster",   "Gastly",     "Haunter",    "Gengar",
    "Onix",       "Drowzee",    "Hypno",      "Krabby",     "Kingler",
    "Voltorb",    "Electrode",  "Exeggcute",  "Exeggutor",  "Cubone",
    "Marowak",    "Hitmonlee",  "Hitmonchan", "Lickitung",  "Koffing",
    "Weezing",    "Rhyhorn",    "Rhydon",     "Chansey",    "Tangela",
    "Kangaskhan", "Horsea",     "Seadra",     "Goldeen",    "Seaking",
    "Staryu",     "Starmie",    "MrMime",     "Scyther",    "Jynx",
    "Electabuzz", "Magmar",     "Pinsir",     "Tauros",     "Magikarp",
    "Gyarados",   "Lapras",     "Ditto",      "Eevee",      "Vaporeon",
    "Jolteon",    "Flareon",    "Porygon",    "Omanyte",    "Omastar",
    "Kabuto",     "Kabutops",   "Aerodactyl", "Snorlax",    "Articuno",
    "Zapdos",     "Moltres",    "Dratini",    "Dragonair",  "Dragonite",
    "Mewtwo",     "Mew"]

libpkmn_moves = [    "None",         "Pound",        "KarateChop",  "DoubleSlap",
    "CometPunch",   "MegaPunch",    "PayDay",      "FirePunch",
    "IcePunch",     "ThunderPunch", "Scratch",     "ViseGrip",
    "Guillotine",   "RazorWind",    "SwordsDance", "Cut",
    "Gust",         "WingAttack",   "Whirlwind",   "Fly",
    "Bind",         "Slam",         "VineWhip",    "Stomp",
    "DoubleKick",   "MegaKick",     "JumpKick",    "RollingKick",
    "SandAttack",   "Headbutt",     "HornAttack",  "FuryAttack",
    "HornDrill",    "Tackle",       "BodySlam",    "Wrap",
    "TakeDown",     "Thrash",       "DoubleEdge",  "TailWhip",
    "PoisonSting",  "Twineedle",    "PinMissile",  "Leer",
    "Bite",         "Growl",        "Roar",        "Sing",
    "Supersonic",   "SonicBoom",    "Disable",     "Acid",
    "Ember",        "Flamethrower", "Mist",        "WaterGun",
    "HydroPump",    "Surf",         "IceBeam",     "Blizzard",
    "Psybeam",      "BubbleBeam",   "AuroraBeam",  "HyperBeam",
    "Peck",         "DrillPeck",    "Submission",  "LowKick",
    "Counter",      "SeismicToss",  "Strength",    "Absorb",
    "MegaDrain",    "LeechSeed",    "Growth",      "RazorLeaf",
    "SolarBeam",    "PoisonPowder", "StunSpore",   "SleepPowder",
    "PetalDance",   "StringShot",   "DragonRage",  "FireSpin",
    "ThunderShock", "Thunderbolt",  "ThunderWave", "Thunder",
    "RockThrow",    "Earthquake",   "Fissure",     "Dig",
    "Toxic",        "Confusion",    "Psychic",     "Hypnosis",
    "Meditate",     "Agility",      "QuickAttack", "Rage",
    "Teleport",     "NightShade",   "Mimic",       "Screech",
    "DoubleTeam",   "Recover",      "Harden",      "Minimize",
    "Smokescreen",  "ConfuseRay",   "Withdraw",    "DefenseCurl",
    "Barrier",      "LightScreen",  "Haze",        "Reflect",
    "FocusEnergy",  "Bide",         "Metronome",   "MirrorMove",
    "SelfDestruct", "EggBomb",      "Lick",        "Smog",
    "Sludge",       "BoneClub",     "FireBlast",   "Waterfall",
    "Clamp",        "Swift",        "SkullBash",   "SpikeCannon",
    "Constrict",    "Amnesia",      "Kinesis",     "SoftBoiled",
    "HighJumpKick", "Glare",        "DreamEater",  "PoisonGas",
    "Barrage",      "LeechLife",    "LovelyKiss",  "SkyAttack",
    "Transform",    "Bubble",       "DizzyPunch",  "Spore",
    "Flash",        "Psywave",      "Splash",      "AcidArmor",
    "Crabhammer",   "Explosion",    "FurySwipes",  "Bonemerang",
    "Rest",         "RockSlide",    "HyperFang",   "Sharpen",
    "Conversion",   "TriAttack",    "SuperFang",   "Slash",
    "Substitute",   "Struggle"]
# fmt:on

species_map = {}
for species in libpkmn_species:
    species_map[species.lower()] = species
move_map = {}
for move in libpkmn_moves:
    move_map[move.lower()] = move


def convert_to_cpp_initializer(line):
    _, team_data = line.split("\t")
    team_entries = team_data.split("]")

    team_output = []
    for entry in team_entries:
        if not entry:  # skip any empty entries
            continue
        species, moves = entry.split("|")
        species = species_map[species]
        moves_list = moves.split(",")
        formatted_moves = ", ".join([move_map[move] for move in moves_list])
        team_output.append(f"Set{{Species::{species}, {{{formatted_moves}}}}}")

    cpp_output = []
    cpp_output.append(
        f"std::array<Set, 6>{{\n    " + ",\n    ".join(team_output) + "\n},"
    )

    return "".join(cpp_output)


def convert(path, n=1):
    print(f"constexpr std::array<std::array<Set, 6>, {n}> teams {{")
    with open(path, "r") as file:
        lines = file.readlines()
    lines = lines[:n]
    code = ""
    for line in lines:
        print(convert_to_cpp_initializer(line.strip()))
    print("};")


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: provide path to the tsv team dump and the number of lines to convert."
        )
        return 1

    try:
        path = str(sys.argv[1])
        n = int(sys.argv[2])
    except ValueError:
        print("Error: All arguments must be integers.")
        return 1

    header = """
#pragma once

#include <array>

#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>

#include <libpkmn/pkmn.h>

namespace SampleTeams {
using Data::Moves;
using Data::Species;
using enum Moves;
using Set = PKMN::PokemonInit;
"""
    print(header)

    convert(path, n)

    footer = """
} // namespace SampleTeams
"""
    print(footer)


main()
