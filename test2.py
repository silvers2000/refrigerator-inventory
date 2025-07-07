import fiftyone.utils.openimages as fouo

# 1) Get the full class list
all_classes = fouo.get_classes()

# 2) Your “desired” fruit names (the ones you care about)
desired = [
    "Apple", "Apricot", "Avocado", "Banana", "Blackberry", "Blueberry",
    "Cantaloupe", "Cherry", "Clementine", "Coconut", "Date", "Durian",
    "Fig", "Grape", "Grapefruit", "Guava", "Honeydew melon", "Jackfruit",
    "Kiwi fruit", "Lemon", "Lime", "Lychee", "Mandarin", "Mango", "Melon",
    "Orange", "Papaya", "Passion fruit", "Peach", "Pear", "Pineapple",
    "Plum", "Pomegranate", "Raspberry", "Strawberry", "Tangerine",
    "Tomato", "Watermelon", "Winter melon",
]

# 3) Keep only those that actually appear in OpenImages
valid_fruits = [f for f in desired if f in all_classes]

print(f"\n{len(valid_fruits)} valid fruit classes:\n", valid_fruits)
