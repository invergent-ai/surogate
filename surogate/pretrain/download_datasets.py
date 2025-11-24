from datasets import Dataset, IterableDataset, load_dataset as hf_load_dataset

def cli_main():
    hf_load_dataset('uonlp/CulturaX', 'ro', split='train')

if __name__ == '__main__':
    cli_main()