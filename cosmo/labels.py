"""Editable label lists. Change these freely; nothing needs retraining.

CATEGORIES is used twice: as text queries for the garment detector, and as
the candidates for zero-shot category labels on each cropped piece.
"""

CATEGORIES = [
    "t-shirt", "shirt", "sweater", "hoodie", "jacket", "coat", "blazer",
    "dress", "skirt", "jeans", "trousers", "sweatpants", "shorts",
    "sneakers", "boots", "loafers", "heels", "sandals",
    "bag", "hat", "cap", "watch", "glasses", "belt", "jewelry",
]

# Broader buckets for the style-profile sentences ("Most of what you save is tops").
KIND = {
    "t-shirt": "tops", "shirt": "tops", "sweater": "tops", "hoodie": "tops",
    "jacket": "outerwear", "coat": "outerwear", "blazer": "outerwear",
    "dress": "dresses and skirts", "skirt": "dresses and skirts",
    "jeans": "bottoms", "trousers": "bottoms", "sweatpants": "bottoms", "shorts": "bottoms",
    "sneakers": "shoes", "boots": "shoes", "loafers": "shoes", "heels": "shoes",
    "sandals": "shoes",
    "bag": "accessories", "hat": "accessories", "cap": "accessories",
    "watch": "accessories", "glasses": "accessories", "belt": "accessories",
    "jewelry": "accessories",
}

FOOTWEAR = {c for c, k in KIND.items() if k == "shoes"}

COLORS = [
    "black", "white", "grey", "beige", "brown", "navy", "blue", "light blue",
    "green", "red", "burgundy", "pink", "yellow", "multicolor",
]

# Material / texture of a single piece. Only textures you can actually see in
# a photo are listed. The PLAIN_MATERIALS entries are catch-alls: when one of
# them wins, we say nothing about material instead of guessing a fibre.
MATERIALS = [
    "denim", "knit", "chunky knit", "fleece", "leather", "suede", "corduroy",
    "nylon", "satin", "mesh",
    "plain woven fabric", "plain jersey fabric",
]
PLAIN_MATERIALS = {"plain woven fabric", "plain jersey fabric"}

# Categories where material is meaningless (a watch is not "fine knit").
NO_MATERIAL = {"watch", "glasses", "jewelry"}

AESTHETICS = [
    "streetwear", "minimalist", "dark academia", "athleisure", "preppy",
    "vintage", "y2k", "formal",
]

# Zero-shot check: is this crop actually clothing? A crop counts as clothing
# if its best match is any prompt in CLOTHING_PROMPTS.
CLOTHING_PROMPTS = [
    "a photo of clothing or accessories", "a pair of shoes", "a wristwatch",
    "a pair of glasses", "jewelry", "a bag",
]
NOT_CLOTHING_PROMPTS = [
    "a screenshot of text", "a photo of food", "a landscape photo",
    "a phone", "headphones or earbuds", "a perfume bottle", "a camera",
]

# Several phrasings per label; their text vectors are averaged, which makes
# the label less sensitive to any one wording.
TEMPLATES = {
    "category": ["a photo of {}, a type of clothing", "a product photo of {}", "{}"],
    "material": ["a photo of clothing made of {}", "close-up of {} fabric", "{} texture"],
    "color": ["a photo of {} clothing", "a {} garment", "the color {}"],
    "aesthetic": ["an outfit in {} style", "a photo of {} fashion", "{} aesthetic"],
}
