Access required datasets:

[UCF50](https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50)

📂 Selected 15 classes (uniformly sampled): ['BaseballPitch', 'Biking', 'CleanAndJerk', 'Fencing', 'HorseR
ace', 'JavelinThrow', 'JumpingJack', 'MilitaryParade', 'PizzaTossing', 'PlayingTabla', 'PommelHorse', 'Pus
hUps', 'Rowing', 'Skiing', 'Swing']

Data folder hierarchy:

```
├── UCF50
│   ├── BaseballPitch
│   │   ├── v_BaseballPitch_g01_c01.avi
│   │   ├── v_BaseballPitch_g01_c02.avi
│   │   ├── v_BaseballPitch_g01_c03.avi
│   │   └── ...
│   ├── Basketball
│   │   ├── v_Basketball_g01_c01.avi
│   │   ├── v_Basketball_g01_c02.avi
│   │   ├── v_Basketball_g01_c03.avi
│   │   └── ...
│   ├── BenchPress
│   │   ├── v_BenchPress_g01_c01.avi
│   │   ├── v_BenchPress_g01_c02.avi
│   │   ├── v_BenchPress_g01_c03.avi
│   │   └── ...
│   ├── Biking
│   │   ├── v_Biking_g01_c01.avi
│   │   ├── v_Biking_g01_c02.avi
│   │   ├── v_Biking_g01_c03.avi
│   │   └── ...
│   ├── Billiards
│   │   ├── v_Billards_g01_c01.avi
│   │   ├── v_Billards_g01_c02.avi
│   │   ├── v_Billards_g01_c03.avi
│   │   └── ...
│   ├── BreastStroke
│   │   ├── v_BreastStroke_g01_c01.avi
│   │   ├── v_BreastStroke_g01_c02.avi
│   │   ├── v_BreastStroke_g01_c03.avi
│   │   └── ...
│   ├── CleanAndJerk
│   │   ├── v_CleanAndJerk_g01_c01.avi
│   │   ├── v_CleanAndJerk_g01_c02.avi
│   │   ├── v_CleanAndJerk_g01_c03.avi
│   │   └── ...
│   ├── Diving
│   │   ├── v_Diving_g01_c01.avi
│   │   ├── v_Diving_g01_c02.avi
│   │   ├── v_Diving_g01_c03.avi
│   │   └── ...
│   ├── Drumming
│   │   ├── v_Drumming_g01_c01.avi
│   │   ├── v_Drumming_g01_c02.avi
│   │   ├── v_Drumming_g01_c03.avi
│   │   └── ...
│   ├── Fencing
│   │   ├── v_Fencing_g01_c01.avi
│   │   ├── v_Fencing_g01_c02.avi
│   │   ├── v_Fencing_g01_c03.avi
│   │   └── ...
│   ├── GolfSwing
│   │   ├── v_GolfSwing_g01_c01.avi
│   │   ├── v_GolfSwing_g01_c02.avi
│   │   ├── v_GolfSwing_g01_c03.avi
│   │   └── ...
│   ├── HighJump
│   │   ├── v_HighJump_g01_c01.avi
│   │   ├── v_HighJump_g01_c02.avi
│   │   ├── v_HighJump_g01_c03.avi
│   │   └── ...
│   ├── HorseRace
│   │   ├── v_HorseRace_g01_c01.avi
│   │   ├── v_HorseRace_g01_c02.avi
│   │   ├── v_HorseRace_g01_c03.avi
│   │   └── ...
│   ├── HorseRiding
│   │   ├── v_HorseRiding_g01_c01.avi
│   │   ├── v_HorseRiding_g01_c02.avi
│   │   ├── v_HorseRiding_g01_c03.avi
│   │   └── ...
│   ├── HulaHoop
│   │   ├── v_HulaHoop_g01_c01.avi
│   │   ├── v_HulaHoop_g01_c02.avi
│   │   ├── v_HulaHoop_g01_c03.avi
│   │   └── ...
│   ├── JavelinThrow
│   │   ├── v_JavelinThrow_g01_c01.avi
│   │   ├── v_JavelinThrow_g01_c02.avi
│   │   ├── v_JavelinThrow_g01_c03.avi
│   │   └── ...
│   ├── JugglingBalls
│   │   ├── v_JugglingBalls_g01_c01.avi
│   │   ├── v_JugglingBalls_g01_c02.avi
│   │   ├── v_JugglingBalls_g01_c03.avi
│   │   └── ...
│   ├── JumpRope
│   │   ├── v_JumpRope_g01_c01.avi
│   │   ├── v_JumpRope_g01_c02.avi
│   │   ├── v_JumpRope_g01_c03.avi
│   │   └── ...
│   ├── JumpingJack
│   │   ├── v_JumpingJack_g01_c01.avi
│   │   ├── v_JumpingJack_g01_c02.avi
│   │   ├── v_JumpingJack_g01_c03.avi
│   │   └── ...
│   ├── Kayaking
│   │   ├── v_Kayaking_g01_c01.avi
│   │   ├── v_Kayaking_g01_c02.avi
│   │   ├── v_Kayaking_g01_c03.avi
│   │   └── ...
│   ├── Lunges
│   │   ├── v_Lunges_g01_c01.avi
│   │   ├── v_Lunges_g01_c02.avi
│   │   ├── v_Lunges_g01_c03.avi
│   │   └── ...
│   ├── MilitaryParade
│   │   ├── v_MilitaryParade_g01_c01.avi
│   │   ├── v_MilitaryParade_g01_c02.avi
│   │   ├── v_MilitaryParade_g01_c03.avi
│   │   └── ...
│   ├── Mixing
│   │   ├── v_Mixing_g01_c01.avi
│   │   ├── v_Mixing_g01_c02.avi
│   │   ├── v_Mixing_g01_c03.avi
│   │   └── ...
│   ├── Nunchucks
│   │   ├── v_Nunchucks_g01_c01.avi
│   │   ├── v_Nunchucks_g01_c02.avi
│   │   ├── v_Nunchucks_g01_c03.avi
│   │   └── ...
│   ├── PizzaTossing
│   │   ├── v_PizzaTossing_g01_c01.avi
│   │   ├── v_PizzaTossing_g01_c02.avi
│   │   ├── v_PizzaTossing_g01_c03.avi
│   │   └── ...
│   ├── PlayingGuitar
│   │   ├── v_PlayingGuitar_g01_c01.avi
│   │   ├── v_PlayingGuitar_g01_c02.avi
│   │   ├── v_PlayingGuitar_g01_c03.avi
│   │   └── ...
│   ├── PlayingPiano
│   │   ├── v_PlayingPiano_g01_c01.avi
│   │   ├── v_PlayingPiano_g01_c02.avi
│   │   ├── v_PlayingPiano_g01_c03.avi
│   │   └── ...
│   ├── PlayingTabla
│   │   ├── v_PlayingTabla_g01_c01.avi
│   │   ├── v_PlayingTabla_g01_c02.avi
│   │   ├── v_PlayingTabla_g01_c03.avi
│   │   └── ...
│   ├── PlayingViolin
│   │   ├── v_PlayingViolin_g01_c01.avi
│   │   ├── v_PlayingViolin_g01_c02.avi
│   │   ├── v_PlayingViolin_g01_c03.avi
│   │   └── ...
│   ├── PoleVault
│   │   ├── v_PoleVault_g01_c01.avi
│   │   ├── v_PoleVault_g01_c02.avi
│   │   ├── v_PoleVault_g01_c03.avi
│   │   └── ...
│   ├── PommelHorse
│   │   ├── v_PommelHorse_g01_c01.avi
│   │   ├── v_PommelHorse_g01_c02.avi
│   │   ├── v_PommelHorse_g01_c03.avi
│   │   └── ...
│   ├── PullUps
│   │   ├── v_Pullup_g01_c01.avi
│   │   ├── v_Pullup_g01_c02.avi
│   │   ├── v_Pullup_g01_c03.avi
│   │   └── ...
│   ├── Punch
│   │   ├── v_Punch_g01_c01.avi
│   │   ├── v_Punch_g01_c02.avi
│   │   ├── v_Punch_g01_c03.avi
│   │   └── ...
│   ├── PushUps
│   │   ├── v_PushUps_g01_c01.avi
│   │   ├── v_PushUps_g01_c02.avi
│   │   ├── v_PushUps_g01_c03.avi
│   │   └── ...
│   ├── RockClimbingIndoor
│   │   ├── v_RockClimbingIndoor_g01_c01.avi
│   │   ├── v_RockClimbingIndoor_g01_c02.avi
│   │   ├── v_RockClimbingIndoor_g01_c03.avi
│   │   └── ...
│   ├── RopeClimbing
│   │   ├── v_RopeClimbing_g01_c01.avi
│   │   ├── v_RopeClimbing_g01_c02.avi
│   │   ├── v_RopeClimbing_g01_c03.avi
│   │   └── ...
│   ├── Rowing
│   │   ├── v_Rowing_g01_c01.avi
│   │   ├── v_Rowing_g01_c02.avi
│   │   ├── v_Rowing_g01_c03.avi
│   │   └── ...
│   ├── SalsaSpin
│   │   ├── v_SalsaSpin_g01_c01.avi
│   │   ├── v_SalsaSpin_g01_c02.avi
│   │   ├── v_SalsaSpin_g01_c03.avi
│   │   └── ...
│   ├── SkateBoarding
│   │   ├── v_SkateBoarding_g01_c01.avi
│   │   ├── v_SkateBoarding_g01_c02.avi
│   │   ├── v_SkateBoarding_g01_c03.avi
│   │   └── ...
│   ├── Skiing
│   │   ├── v_Skiing_g01_c01.avi
│   │   ├── v_Skiing_g01_c02.avi
│   │   ├── v_Skiing_g01_c03.avi
│   │   └── ...
│   ├── Skijet
│   │   ├── v_Skijet_g01_c01.avi
│   │   ├── v_Skijet_g01_c02.avi
│   │   ├── v_Skijet_g01_c03.avi
│   │   └── ...
│   ├── SoccerJuggling
│   │   ├── v_SoccerJuggling_g01_c01.avi
│   │   ├── v_SoccerJuggling_g01_c02.avi
│   │   ├── v_SoccerJuggling_g01_c03.avi
│   │   └── ...
│   ├── Swing
│   │   ├── v_Swing_g01_c01.avi
│   │   ├── v_Swing_g01_c02.avi
│   │   ├── v_Swing_g01_c03.avi
│   │   └── ...
│   ├── TaiChi
│   │   ├── v_TaiChi_g01_c01.avi
│   │   ├── v_TaiChi_g01_c02.avi
│   │   ├── v_TaiChi_g01_c03.avi
│   │   └── ...
│   ├── TennisSwing
│   │   ├── v_TennisSwing_g01_c01.avi
│   │   ├── v_TennisSwing_g01_c02.avi
│   │   ├── v_TennisSwing_g01_c03.avi
│   │   └── ...
│   ├── ThrowDiscus
│   │   ├── v_ThrowDiscus_g01_c01.avi
│   │   ├── v_ThrowDiscus_g01_c02.avi
│   │   ├── v_ThrowDiscus_g01_c03.avi
│   │   └── ...
│   ├── TrampolineJumping
│   │   ├── v_TrampolineJumping_g01_c01.avi
│   │   ├── v_TrampolineJumping_g01_c02.avi
│   │   ├── v_TrampolineJumping_g01_c03.avi
│   │   └── ...
│   ├── VolleyballSpiking
│   │   ├── v_VolleyballSpiking_g01_c01.avi
│   │   ├── v_VolleyballSpiking_g01_c02.avi
│   │   ├── v_VolleyballSpiking_g01_c03.avi
│   │   └── ...
│   ├── WalkingWithDog
│   │   ├── v_WalkingWithDog_g01_c01.avi
│   │   ├── v_WalkingWithDog_g01_c02.avi
│   │   ├── v_WalkingWithDog_g01_c03.avi
│   │   └── ...
│   └── YoYo
│       ├── v_YoYo_g01_c01.avi
│       ├── v_YoYo_g01_c02.avi
│       ├── v_YoYo_g01_c03.avi
│       └── ...
├── README.md
└── hierarchy.py
```