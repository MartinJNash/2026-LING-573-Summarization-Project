from pathlib import Path
from datasets import Dataset

def read_gs_training_data():
    parent = "data/multiclinsum_gs_train_en"
    yield from read_folder(parent)

def read_test_training_data():
    parent = "data/multiclinsum_test_en"
    yield from read_folder(parent)

def read_folder(root: str):
    """Reads MultiClinSum data from parent folder"""

    root_path = Path(root)
    fulltext = root_path / "fulltext"
    summaries = root_path / "summaries"

    for fulltext_file_path in sorted(fulltext.iterdir()):
        if fulltext_file_path.is_file():
            summary_file_path = summaries / (fulltext_file_path.stem + "_sum.txt")

            with open(fulltext_file_path, 'r+') as f:
                original_text = f.read()

            with open(summary_file_path, 'r+') as f:
                summary_text = f.read()

            yield {
                "input": original_text,
                "target": summary_text
            }

if __name__ == "__main__":
    print("start")
    ds = Dataset.from_generator(read_gs_training_data)
    print(ds)
    print("done")
