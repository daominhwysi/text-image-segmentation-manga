import os
import random
from pathlib import Path

# --- Configuration ---
VALID_EXTENSIONS = {".otf", ".ttf"}


def is_font_file(filename):
    return filename.lower().endswith(tuple(VALID_EXTENSIONS))


class FontHierarchicalSampler:
    def __init__(self, file_paths):
        """
        Builds a virtual directory tree from a list of file paths.
        structure = {
            'type': 'dir',
            'path': 'full/path',
            'children': {
                'foldername': { ... node ... },
                'filename.ttf': { 'type': 'file', 'path': ... }
            }
        }
        """
        self.tree = {"type": "root", "children": {}}
        self._build_tree(file_paths)

    def _build_tree(self, paths):
        for path_str in paths:
            path = Path(path_str)
            parts = path.parts  # e.g. ('fonts', 'subdir', 'font.ttf')

            current_level = self.tree

            # Iterate through the path parts to build the tree
            for i, part in enumerate(parts):
                is_file = i == len(parts) - 1

                # If we are at the file level
                if is_file:
                    current_level["children"][part] = {
                        "type": "file",
                        "path": str(path),
                        "name": part,
                    }
                else:
                    # If we are at a directory level, ensure it exists
                    if part not in current_level["children"]:
                        current_level["children"][part] = {
                            "type": "dir",
                            "children": {},
                            "name": part,
                        }
                    # Move deeper
                    current_level = current_level["children"][part]

    def get_random_font(self):
        """
        Public method to start the recursive search from the root.
        """
        if not self.tree["children"]:
            return None
        return self._recursive_select(self.tree)

    def _recursive_select(self, node):
        """
        Mimics the original logic:
        1. List all immediate candidates (files and subfolders).
        2. Pick one with 1/n probability.
        3. If file -> return. If dir -> recurse.
        """
        # Get immediate children nodes
        candidates = list(node["children"].values())

        if not candidates:
            return None

        # THE CORE LOGIC (Same as original):
        # Probability = 1 / n (where n is number of siblings)
        selected_node = random.choice(candidates)

        # print(f"-> Selected: {selected_node['name']} (from {len(candidates)} options)")

        if selected_node["type"] == "file":
            return selected_node["path"]
        else:
            return self._recursive_select(selected_node)


# --- Setup Functions ---


def scan_all_fonts(root_directory):
    """
    Scans the IO once to get every single valid font path.
    """
    all_fonts = []
    root_path = Path(root_directory)

    if not root_path.exists():
        print(f"Error: Directory '{root_directory}' not found.")
        return []

    # os.walk is generally faster than recursive glob for large trees
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if is_font_file(file):
                all_fonts.append(os.path.join(root, file))

    return all_fonts


def initialize_font_samplers(root_folder, split_ratio=0.9):
    """
    1. Scans disk.
    2. Splits data.
    3. Returns two Sampler objects (Train and Test).
    """
    print("Scanning file system...")
    all_paths = scan_all_fonts(root_folder)
    print(f"Found {len(all_paths)} fonts total.")

    # Shuffle for random split
    random.shuffle(all_paths)

    split_index = int(len(all_paths) * split_ratio)
    train_paths = all_paths[:split_index]
    test_paths = all_paths[split_index:]

    print(f"Building Trees -> Train: {len(train_paths)}, Test: {len(test_paths)}")

    train_sampler = FontHierarchicalSampler(train_paths)
    test_sampler = FontHierarchicalSampler(test_paths)

    return train_sampler, test_sampler


# --- Usage Example ---

if __name__ == "__main__":
    # 1. Setup Phase (Do this once at program start)
    # create a dummy folder "fonts" with some files to test if you run this script directly
    ROOT_DIR = "fonts"

    # If you don't have a fonts folder, this will return empty samplers
    train_gen, test_gen = initialize_font_samplers(ROOT_DIR, split_ratio=0.8)

    # 2. Runtime Phase (Looping extensively without IO lag)

    # Example: Get a font for Training
    print("\n--- Training Selection ---")
    font_path = train_gen.get_random_font()
    if font_path:
        print(f"WINNER TRAIN: {font_path}")
    else:
        print("No fonts found in train set.")

    # Example: Get a font for Testing
    print("\n--- Testing Selection ---")
    font_path = test_gen.get_random_font()
    if font_path:
        print(f"WINNER TEST: {font_path}")
    else:
        print("No fonts found in test set.")
