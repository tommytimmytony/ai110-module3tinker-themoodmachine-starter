"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # Added after failure analysis
    "hopeful",   # "tired but kind of hopeful" scored only negative
    "proud",     # "proud of how far I've come" missed entirely
    "glad",
    "grateful",
    "enjoy",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # Added after failure analysis
    "worse",     # "could be worse" — understated negative
    "wrong",     # "nothing went wrong" — double negation test
    "garbage",   # "feeling like absolute garbage"
    "exhausted",
    "miserable",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    # New posts — varied styles, slang, emojis, ambiguity
    "omg that concert was INSANE 🔥🔥 best night ever",           # hype / all caps enthusiasm
    "woke up late, missed the bus, spilled my coffee. cool cool cool",  # sarcasm
    "lowkey stressed about finals but ngl kinda proud of how far i've come",  # mixed feelings
    "it's giving main character energy today 😌✨",               # positive slang
    "I absolutely love sitting in a 3-hour meeting that could've been an email 💀",  # sarcasm
    "not sad just... tired of everything idk",                    # subtle negative / ambiguous
    "the food was bad but the vibes were immaculate ngl",         # mixed
    "😭😭😭 why does everything happen at once",                  # emoji-heavy, negative
    # Round 3 — stress test: sarcasm, double negation, slang polysemy
    "I love Mondays",                                              # sarcasm — "love" fires positively
    "that movie was so bad it was good",                           # deliberate paradox
    "nothing went wrong today surprisingly",                       # double negation: nothing + wrong
    "could be worse I guess",                                      # understated negative
    "this slaps no cap",                                           # pure slang, no sentiment words
    "woke up feeling like absolute garbage 💀",                   # emoji + negative phrase
    "not gonna lie I actually had fun",                            # negator before non-sentiment word
    "proud of how far I've come even if I'm still tired",         # mixed — explicit
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"  — note: often sarcastic in internet slang
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    # Labels for new posts
    "positive",  # "omg that concert was INSANE..." — genuine excitement
    "negative",  # "woke up late, missed the bus..." — sarcastic "cool cool cool"
    "mixed",     # "lowkey stressed... kinda proud" — mixed feelings
    "positive",  # "it's giving main character energy" — positive slang
    "negative",  # "I absolutely love sitting in a 3-hour meeting..." — sarcasm = negative
    "negative",  # "not sad just... tired of everything" — subtle negative
    "mixed",     # "food was bad but vibes were immaculate" — explicit mixed
    "negative",  # "😭😭😭 why does everything happen at once" — negative
    # Round 3 labels
    "negative",  # "I love Mondays" — sarcasm
    "mixed",     # "so bad it was good" — deliberate paradox
    "positive",  # "nothing went wrong today" — double negation = positive
    "negative",  # "could be worse I guess" — understated, defeatist
    "positive",  # "this slaps no cap" — slang for great
    "negative",  # "feeling like absolute garbage 💀"
    "positive",  # "not gonna lie I actually had fun" — negator is prefix filler here
    "mixed",     # "proud... even if I'm still tired"
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
