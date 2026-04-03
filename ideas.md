ideas.md

better folder management / currently there is a lot of data lying around

- raw video
- cropped video

- Fastvolleyball
  - track the ball
  - detect rallies
  - separate clips / split files
  - this would help with tracking score too

- player detection
  - pass 1 yolo?
  - player attribution pass 2
  - works with seed position + new colouring etc. seems to be ok
  - comparision with ground truth
  - figure out a good way to get correct actions + correct players

- action detection
  - currently with LLM ? maybe we can do something simpler with a classifier on the cropped video? or maybe we can do something with the ball tracking data? and maybe just get the player closest to the ball = the one doing the action?
  - here we might want chunking to keep context small

# viewer/editor

dual purpose, later = viewing only,
ealier = edit ground truth data

- should select one video then load ground truth data + the generated runs?

- not sure if video/ is still used anywhere

How is the player detection done vs. how it's done in my script?

How can we make the player detection more robust?
