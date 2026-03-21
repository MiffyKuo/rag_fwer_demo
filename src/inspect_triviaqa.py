from datasets import load_dataset
from pprint import pprint

# 確認 TriviaQA 長甚麼樣子
def main():
    print("Loading a very small TriviaQA sample for inspection...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="train[:2]")

    print("\nNumber of rows:", len(ds))
    print("\nColumn names:")
    print(ds.column_names)

    print("\nFirst example:")
    row = ds[0]
    pprint(row)

if __name__ == "__main__":
    main()