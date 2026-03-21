from pathlib import Path
from typing import List, Optional
import os

def get_linux_style_dataset_list(
    dataset_root: str,
    exclude_prefixes: Optional[List[str]] = None,
    include_prefixes: Optional[List[str]] = None
) -> List[str]:

    """
    Return all matching paths under the `setups` subdirectory in Linux-style format
    using '/' as the path separator.

    Args:
        dataset_root (str): Root directory path, for example 'H:/Valid_datasets/CompenHR_datasets'.
        exclude_prefixes (List[str], optional): Exclude directory names with these prefixes.
        include_prefixes (List[str], optional): Only include directory names with these prefixes
            (higher priority than `exclude_prefixes`).
    Returns:
        List[str]: A list of paths like ['setups/Block1', 'setups/Fruits_Vegetables2', ...].
    """

    exclude_prefixes = exclude_prefixes or []
    setups_dir = Path(dataset_root) / 'setups'

    if not setups_dir.exists():
        print(f"[Warning] Directory does not exist: {setups_dir}")
        return []

    all_dirs = [name for name in os.listdir(setups_dir) if (setups_dir / name).is_dir()]

    if include_prefixes:
        filtered_dirs = [
            name for name in all_dirs if any(name.startswith(p) for p in include_prefixes)
        ]
    else:
        filtered_dirs = [
            name for name in all_dirs if not any(name.startswith(p) for p in exclude_prefixes)
        ]
    data_list = [os.path.join("setups", name).replace("\\", "/") for name in filtered_dirs]

    # Print output
    print("Dataset list:")
    for path in data_list:
        print(f"  {path}")
    print(f"\nTotal: {len(data_list)} setup directories")

    return data_list


# SIComp_dataset1_root = r'../../../data/SIComp_train_dataset1'
SIComp_dataset1_root = r'H:/mnt/data/lihaibo/data/SIComp_train_dataset1'
SIComp_dataset1_lists = [
    r"setups/Cherry/pos1",
    r"setups/Cherry/pos2",
    r"setups/Cherry/pos3",
    r"setups/Clouds/pos1",
    r"setups/Clouds/pos2",
    r"setups/Clouds/pos3",
    r"setups/Clouds/pos4",
    r"setups/Clouds/pos5",
    r"setups/coastline/pos1",
    r"setups/coastline/pos2",
    r"setups/coastline/pos3",
    r"setups/coastline/pos4",
    r"setups/coastline/pos5",
    r"setups/Colorful_stripes/pos1",
    r"setups/Colorful_stripes/pos2",
    r"setups/Cube/pos1",
    r"setups/Flowers/pos1",
    r"setups/Flowers/pos2",
    r"setups/Flowers/pos3",
    r"setups/Lake/pos1",
    r"setups/Lake/pos2",
    r"setups/Lotus_Leaf/pos1",
    r"setups/Lotus_Leaf/pos2",
    r"setups/Marble/pos1",
    r"setups/Marble/pos2",
    r"setups/Marble/pos3",
    r"setups/Moon/pos1",
    r"setups/Moon/pos2",
    r"setups/Moon/pos3",
    r"setups/Moon/pos4",
    r"setups/Paint/pos1",
    r"setups/Paint/pos2",
    r"setups/Paint/pos3",
    r"setups/Pillow/pos1",
    r"setups/Pillow/pos2",
    r"setups/Pillow/pos3",
    r"setups/Plastic_board/pos1",
    r"setups/Plastic_board/pos2",
    r"setups/Plastic_board/pos3",
    r"setups/Rainbow/pos1",
    r"setups/Rainbow/pos2",
    r"setups/Rainbow/pos3",
    r"setups/Rainbow/pos4",
    r"setups/Sea_flowers/pos1",
    r"setups/Sea_flowers/pos2",
    r"setups/Sea_flowers/pos3",
    r"setups/Sea_flowers/pos4",
    r"setups/Stripes/pos1",
    r"setups/Stripes/pos2",
    r"setups/Stripes/pos3",
    r"setups/Stripes/pos4",
    r"setups/Wood_Grain/pos1",
    r"setups/Wood_Grain/pos2",
    r"setups/Wood_Grain/pos3",
    r"setups/Wood_Grain/pos4",
    r"setups/wooden_board/pos1",
    r"setups/wooden_board/pos2",
    r"setups/wooden_board/pos3",
    r"setups/Wool/pos1",
    r"setups/Wool/pos2",
    r"setups/Wool/pos3",
    r"setups/Wool/pos4",
]

# SIComp_dataset2_root = r'../../../data/SIComp_train_dataset2'
SIComp_dataset2_root = r'H:/mnt/data/lihaibo/data/SIComp_train_dataset2'
SIComp_dataset2_lists = [
    r"setups/Blue_iron_gate/pos1",
    r"setups/Blue_iron_gate/pos2",
    r"setups/Blue_iron_gate/pos3",
    r"setups/Brick_wall/pos1",
    r"setups/Brick_wall/pos2",
    r"setups/Brick_wall/pos3",
    r"setups/Brick_wall/pos4",
    r"setups/Brick_wall/pos5",
    r"setups/Brick_wall/pos6",
    r"setups/defective_wall/pos1",
    r"setups/defective_wall/pos2",
    r"setups/defective_wall/pos3",
    r"setups/defective_wall/pos2",
    r"setups/defective_wall/pos3",
    r"setups/Graffiti/pos1",
    r"setups/Graffiti/pos2",
    r"setups/Graffiti/pos1",
    r"setups/Graffiti/pos2",
    r"setups/Mirror/pos1",
    r"setups/Mirror/pos1",
    r"setups/Playground_Wall/pos1",
    r"setups/Playground_Wall/pos2",
    r"setups/Playground_Wall/pos3",
    r"setups/scrawl/pos1",
    r"setups/scrawl/pos2",
    r"setups/scrawl/pos3",
    r"setups/scrawl/pos4",
    r"setups/scrawl/pos5",
    r"setups/scrawl/pos6",
    r"setups/wall/pos1",
    r"setups/wall/pos2",
    r"setups/wood/pos1",
    r"setups/wood/pos2",
    r"setups/wood/pos3",
    r"setups/wood/pos4",
    r"setups/wood/pos5",
    r"setups/wood/pos6",
    r"setups/wood/pos7",
    r"setups/wood/pos8",
    r"setups/wood/pos9",
]

# CompenNet_plus_plus_root = r'../../../data/compenNet_plus_plus/data'
CompenNet_plus_plus_root =  r'H:/mnt/data/lihaibo/data/CompenNet_plus_plus/data'
CompenNet_plus_plus_lists = [
    r'light1/pos1/lavender_np',
    r'light1/pos2/cubes_np',
    r'light2/pos1/lavender_np',
    r'light2/pos1/stripes_np',
    r'light2/pos4/lavender_np',
    r'light2/pos6/curves_np',
    r'light3/pos1/bubbles_np',
    r'light3/pos2/curves_np',
    r'light3/pos2/water_np',
]

# CompenHR_root = r'../../../data/CompenHR'
CompenHR_root = r'H:/mnt/data/lihaibo/data/CompenHR'
CompenHR_lists = [
    r"setups/bubble/1",
    r"setups/bubble/2",
    r"setups/cloud/1",
    r"setups/cloud/2",
    r"setups/cloud/3",
    r"setups/cube/1",
    r"setups/cube/2",
    r"setups/cube/3",
    r"setups/flower_np/1",
    r"setups/lavender/1",
    r"setups/lavender/2",
    r"setups/lavender/3",
    r"setups/leaf_np/1",
    r"setups/lemon_np/1",
    r"setups/rock_np/1",
    r"setups/stripes/1",
    r"setups/stripes/2",
    r"setups/stripes/3",
    r"setups/stripes_np/1",
    r"setups/water/1",
    r"setups/water/2",
]

# Validate_root = r"../../../data/Validate_dataset"
Validate_root = r'H:/mnt/data/lihaibo/data/Validate_dataset'
Validate_lists = get_linux_style_dataset_list(dataset_root=Validate_root)