# µGPT

An implemention of GPT for learning and fun.

Based on the the fantistic lectures by Andrej Karpathy:

  - [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY)

## Dataset

  [Family Guy Dialogues with various Lexicon Ratings](https://www.kaggle.com/datasets/eswarreddy12/family-guy-dialogues-with-various-lexicon-ratings)

  ```sh
  kaggle datasets download -d eswarreddy12/family-guy-dialogues-with-various-lexicon-ratings
  mkdir -p data
  unzip family-guy-dialogues-with-various-lexicon-ratings.zip -d data
  rm family-guy-dialogues-with-various-lexicon-ratings.zip
  ```
