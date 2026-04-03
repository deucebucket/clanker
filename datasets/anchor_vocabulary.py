"""Anchor vocabulary -- the 111 seed words with known polarity.
These are the foundation for the web architecture.
Next step: classify 500+ middle-tier words at ±10-15 force.
"""

ANCHORS = {
    # SOLID POSITIVE (25=strongest, 10=mild)
    'love': 25, 'joy': 25, 'happy': 25, 'happiness': 25,
    'beautiful': 20, 'wonderful': 20, 'amazing': 20, 'awesome': 20,
    'grateful': 20, 'proud': 20, 'celebrate': 20, 'triumph': 20,
    'hope': 15, 'peace': 15, 'safe': 15, 'kind': 15,
    'trust': 15, 'comfort': 15, 'gentle': 15, 'warm': 15,
    'good': 15, 'enjoy': 15, 'excited': 15, 'thrilled': 15,
    'confident': 15, 'inspired': 15, 'relieved': 15,
    'nice': 10, 'calm': 10, 'pleasant': 10, 'amused': 10,
    'fun': 10, 'funny': 10, 'sweet': 10, 'lovely': 15,
    'brilliant': 20, 'excellent': 15, 'perfect': 15, 'fantastic': 20,
    'delightful': 15, 'gorgeous': 15, 'thankful': 15, 'blessed': 15,
    'joyful': 20, 'sober': 10,

    # SOLID NEGATIVE (25=strongest, 10=mild)
    'hate': -25, 'hatred': -25, 'agony': -25, 'anguish': -25,
    'torture': -25, 'torment': -25, 'abuse': -25,
    'murder': -25, 'suicide': -25, 'death': -20, 'die': -20,
    'kill': -20, 'killed': -20, 'killing': -20,
    'grief': -25, 'despair': -25, 'terror': -25, 'horror': -20,
    'misery': -25, 'suffering': -20, 'pain': -20,
    'worthless': -25, 'hopeless': -25, 'helpless': -20,
    'devastated': -20, 'heartbroken': -20, 'betrayed': -20,
    'depressed': -20, 'suicidal': -20,
    'hurt': -15, 'sad': -15, 'angry': -15, 'scared': -15,
    'lonely': -15, 'guilty': -15, 'ashamed': -15,
    'anxious': -15, 'miserable': -20, 'terrified': -20,
    'frustrated': -12, 'disgusted': -15,
    'embarrassed': -10, 'annoyed': -10, 'jealous': -12,
    'crying': -15, 'sobbing': -20,
    'ugly': -15, 'terrible': -15, 'horrible': -15, 'awful': -12,
    'cruel': -20, 'vicious': -20, 'evil': -25,
    'dead': -15, 'dying': -20,
    'rape': -25, 'assault': -20, 'violence': -20,
    'relapsed': -15, 'overdosed': -15,
    'poverty': -15, 'disease': -15, 'cancer': -20,
    'war': -15, 'disaster': -20, 'catastrophe': -20,
    'cheated': -20, 'cheating': -20, 'lied': -15, 'lying': -15,
    'stolen': -12, 'betrayal': -20,
}
