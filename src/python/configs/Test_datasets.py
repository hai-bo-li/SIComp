from pathlib import Path
from typing import List, Optional
import os



def get_linux_style_dataset_list(
        dataset_root: str,
        exclude_prefixes: Optional[List[str]] = None,
        include_prefixes: Optional[List[str]] = None
) -> List[str]:
    """
    Return all matching paths under the `setups` subdirectory in Linux-style format,
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


valid_data_root = os.getenv("DATASET_ROOT", "xxx")
data_name = os.getenv("DATA_NAME", "")
valid_prefixes = [data_name] if data_name else []
valid_data_lists = get_linux_style_dataset_list(dataset_root=valid_data_root, include_prefixes=valid_prefixes)

print(f"Dataset Root: {valid_data_root}")
print(f"Prefixes: {valid_prefixes}")
