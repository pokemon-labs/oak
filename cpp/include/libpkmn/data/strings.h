#pragma once

#include <pkmn.h>

#include <array>
#include <iostream>
#include <sstream>
#include <string>

#include <libpkmn/data/moves.h>
#include <libpkmn/data/species.h>

namespace PKMN {

namespace Data {

constexpr size_t MAX_MOVE_LEN = 12;

constexpr std::array<std::array<char, MAX_MOVE_LEN + 1>, 166> MOVE_CHAR_ARRAY{
    {"None",         "Pound",        "KarateChop",  "DoubleSlap",
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
     "Substitute",   "Struggle"}};

constexpr size_t MAX_SPECIES_LEN = 11;

constexpr std::array<std::array<char, MAX_SPECIES_LEN + 1>, 152>
    SPECIES_CHAR_ARRAY{
        {"None",       "Bulbasaur",  "Ivysaur",    "Venusaur",   "Charmander",
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
         "Mewtwo",     "Mew"}};

constexpr std::array<std::array<char, 9>, 15> TYPE_CHAR_ARRAY = {
    "Normal", "Fighting", "Flying",  "Poison", "Ground",
    "Rock",   "Bug",      "Ghost",   "Fire",   "Water",
    "Grass",  "Electric", "Psychic", "Ice",    "Dragon"};

}; // namespace Data

inline const char *species_char_array(const auto species) noexcept {
  return Data::SPECIES_CHAR_ARRAY[static_cast<uint8_t>(species)].data();
}

inline const char *move_char_array(const auto move) noexcept {
  return Data::MOVE_CHAR_ARRAY[static_cast<uint8_t>(move)].data();
}

inline std::string species_string(const auto species) noexcept {
  return std::string{
      Data::SPECIES_CHAR_ARRAY[static_cast<uint8_t>(species)].data()};
}

inline std::string move_string(const auto move) noexcept {
  auto id = static_cast<uint8_t>(move);
  if (id != 0xFF) [[likely]] {
    return std::string{Data::MOVE_CHAR_ARRAY[id].data()};
  } else {
    return "SKIP_TURN";
  }
}

inline std::string types_string(const auto types) noexcept {
  return std::string{
             Data::TYPE_CHAR_ARRAY[static_cast<uint8_t>(types) % 16].data()} +
         '/' +
         std::string{
             Data::TYPE_CHAR_ARRAY[static_cast<uint8_t>(types) / 16].data()};
}

} // namespace PKMN