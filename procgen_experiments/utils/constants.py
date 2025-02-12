EPSILON = 1e-9

# Max achieveable scores From https://arxiv.org/abs/1912.01588
#             Hard      Easy
# Environment Rmin Rmax Rmin Rmax
# CoinRun 5 10 5 10
# StarPilot 1.5 35 2.5 64
# CaveFlyer 2 13.4 3.5 12
# Dodgeball 1.5 19 1.5 19
# FruitBot -.5 27.2 -1.5 32.4
# Chaser .5 14.2 .5 13
# Miner 1.5 20 1.5 13
# Jumper 1 10 3 10
# Leaper 1.5 10 3 10
# Maze 4 10 5 10
# BigFish 0 40 1 40
# Heist 2 10 3.5 10
# Climber 1 12.6 2 12.6
# Plunder 3 30 4.5 30
# Ninja 2 10 3.5 10
# BossFight .5 13 .5 13

# consider only last 2 columns of the table above
PROCGEN_MINMAX_SCORES = {
  'BigfishEasy-v0': (1, 40),
  'BossfightEasy-v0': (.5, 13),
  'CaveflyerEasy-v0': (3.5, 12),
  'ChaserEasy-v0': (.5, 13),
  'ClimberEasy-v0': (2, 12.6),
  'CoinrunEasy-v0': (5, 10),
  'DodgeballEasy-v0': (1.5, 19),
  'FruitbotEasy-v0': (-1.5, 32.4),
  'HeistEasy-v0': (3.5, 10),
  'JumperEasy-v0': (3, 10),
  'LeaperEasy-v0': (3, 10),
  'MazeEasy-v0': (5, 10),
  'MinerEasy-v0': (1.5, 13),
  'NinjaEasy-v0': (3.5, 10),
  'PlunderEasy-v0': (4.5, 30),
  'StarpilotEasy-v0': (2.5, 64),
}

# Not sure about max rewards for games without "OK"
PROCGEN_MINMAX_REWARDS = {
  'BigfishEasy-v0': (0, 11), #OK
  'BossfightEasy-v0': (0, 11), #OK
  'CaveflyerEasy-v0': (0, 12),
  'ChaserEasy-v0': (0, 13),
  'ClimberEasy-v0': (0, 12.6),
  'CoinrunEasy-v0': (0, 10), #OK
  'DodgeballEasy-v0': (0, 19),
  'FruitbotEasy-v0': (-4, 20), #OK
  'HeistEasy-v0': (0, 10),
  'JumperEasy-v0': (0, 10),
  'LeaperEasy-v0': (0, 10),
  'MazeEasy-v0': (0, 10), #OK
  'MinerEasy-v0': (0, 13),
  'NinjaEasy-v0': (0, 10),
  'PlunderEasy-v0': (0, 11), #OK
  'StarpilotEasy-v0': (0, 2), #OK if you managed to hit an ennemy in two directions at once
}


PROCGEN_BASELINE_SCORES = {
  'ppo': {
    'train_scores': {
      'BigfishEasy-v0': 10.83,
      'BossfightEasy-v0': 7.69,
      'CaveflyerEasy-v0': 7.14,
      'ChaserEasy-v0': 3.02,
      'ClimberEasy-v0': 7.41,
      'CoinrunEasy-v0': 8.75,
      'DodgeballEasy-v0': 4.63,
      'FruitbotEasy-v0': 26.37,
      'HeistEasy-v0': 7.59,
      'JumperEasy-v0': 8.50,
      'LeaperEasy-v0': 5.17,
      'MazeEasy-v0': 9.39,
      'MinerEasy-v0': 9.87,
      'NinjaEasy-v0': 5.84,
      'PlunderEasy-v0': 5.85,
      'StarpilotEasy-v0': 24.14,
    },
    'test_scores': {
      'BigfishEasy-v0': 5.09,
      'BossfightEasy-v0': 6.98,
      'CaveflyerEasy-v0': 5.14,
      'ChaserEasy-v0': 2.58,
      'ClimberEasy-v0': 4.85,
      'CoinrunEasy-v0': 8.07,
      'DodgeballEasy-v0': 1.97,
      'FruitbotEasy-v0': 25.15,
      'HeistEasy-v0': 2.78,
      'JumperEasy-v0': 5.63,
      'LeaperEasy-v0': 5.01,
      'MazeEasy-v0': 5.26,
      'MinerEasy-v0': 6.99,
      'NinjaEasy-v0': 4.81,
      'PlunderEasy-v0': 5.11,
      'StarpilotEasy-v0': 22.58,
    },
  },
  # group: PPO-sh-1B-v11, env_id: PlunderEasy-v023.210493098123436group: PPO-sh-1B-v11, env_id: ChaserEasy-v010.04778491648203
  'ppo700M': {
    'train_scores': {
      'BigfishEasy-v0': 30.2,
      'BossfightEasy-v0': 12,
      'CaveflyerEasy-v0': 11.2,
      'ChaserEasy-v0': 10.4,
      'ClimberEasy-v0': 12.1,
      'CoinrunEasy-v0': 10.,
      'DodgeballEasy-v0': 18.,
      'FruitbotEasy-v0': 31.,
      'HeistEasy-v0': 10.,
      'JumperEasy-v0': 9.,
      'LeaperEasy-v0': 10.,
      'MazeEasy-v0': 10.,
      'MinerEasy-v0': 12.3,
      'NinjaEasy-v0': 10.,
      'PlunderEasy-v0': 23.2,
      'StarpilotEasy-v0': 48.,
    },
    'test_scores': {
      'BigfishEasy-v0': 30.2,
      'BossfightEasy-v0': 12,
      'CaveflyerEasy-v0': 11.2,
      'ChaserEasy-v0': 10.4,
      'ClimberEasy-v0': 12.1,
      'CoinrunEasy-v0': 10.,
      'DodgeballEasy-v0': 18.,
      'FruitbotEasy-v0': 31.,
      'HeistEasy-v0': 10.,
      'JumperEasy-v0': 9.,
      'LeaperEasy-v0': 10.,
      'MazeEasy-v0': 10.,
      'MinerEasy-v0': 12.3,
      'NinjaEasy-v0': 10.,
      'PlunderEasy-v0': 23.2,
      'StarpilotEasy-v0': 48.,
    },
  }
}